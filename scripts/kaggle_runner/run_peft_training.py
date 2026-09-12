"""Kaggle GPU Runner: PEFT MedSAM Training and Evaluation.

Designed to execute on Kaggle Dual-T4 GPU environment with:
- Datasets:
  - tschandl/isic2018-challenge-task1-data-segmentation
  - vinuchandrang/medsam-weights
- Supports modes:
  - decoder_only (E04)
  - lora (E05)
  - standard_adapter (E06)
  - cg_adapter (E07)
  - ablation (E08)

Outputs:
- /kaggle/working/best_{mode}_model.pth
- /kaggle/working/history_{mode}.json
- /kaggle/working/validation_robustness_{mode}.csv
- /kaggle/working/training_report_{mode}.md
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import glob
import time
import argparse
import subprocess
import json

# Ensure dependencies installed
try:
    import segment_anything
except ImportError:
    print("[Kaggle Runner] Installing segment-anything...")
    subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"], check=True)
    import segment_anything

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry


# -------------------------------------------------------------
# 1. Modular Architectures (Adapters & PEFT)
# -------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r) if r > 0 else 1.0

        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Linear(original_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original_linear(x)
        lora = self.lora_B(self.lora_A(x)) * self.scaling
        return orig + lora


class ContrastGatingMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, c_prompt: torch.Tensor) -> torch.Tensor:
        if c_prompt.ndim == 1:
            c_prompt = c_prompt.unsqueeze(-1)
        return 2.0 * self.mlp(c_prompt)


class BottleneckAdapter(nn.Module):
    def __init__(self, embed_dim: int = 768, bottleneck_rank: int = 16, gated: bool = False):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_rank)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_rank, embed_dim)
        self.gated = gated
        if gated:
            self.gating = ContrastGatingMLP(hidden_dim=32)
        else:
            self.register_buffer("constant_gate", torch.tensor(1.0))

        nn.init.kaiming_uniform_(self.down_proj.weight, a=np.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, c_prompt: torch.Tensor = None) -> torch.Tensor:
        res = self.act(self.down_proj(x))
        delta = self.up_proj(res)
        if self.gated:
            if c_prompt is None:
                gamma = torch.ones((x.size(0), 1), device=x.device)
            else:
                gamma = self.gating(c_prompt)
            gamma_view = gamma.view(x.size(0), *([1] * (x.ndim - 1)))
            delta = delta * gamma_view
        return x + delta


class PEFTMedSAM(nn.Module):
    def __init__(self, base_sam: nn.Module, mode: str = "cg_adapter", rank: int = 16, lora_alpha: float = 32.0, ablation_type: str = None):
        super().__init__()
        self.base_sam = base_sam
        self.mode = mode
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.ablation_type = ablation_type

        # Freeze image encoder and prompt encoder
        for p in self.base_sam.image_encoder.parameters():
            p.requires_grad = False
        for p in self.base_sam.prompt_encoder.parameters():
            p.requires_grad = False
        # Train mask decoder
        for p in self.base_sam.mask_decoder.parameters():
            p.requires_grad = True

        self.adapters = nn.ModuleList()
        self.lora_layers = nn.ModuleList()
        vit_blocks = self.base_sam.image_encoder.blocks

        if self.mode == "lora":
            for block in vit_blocks:
                lora_qkv = LoRALinear(block.attn.qkv, r=rank, lora_alpha=lora_alpha)
                block.attn.qkv = lora_qkv
                self.lora_layers.append(lora_qkv)

        elif self.mode in ["standard_adapter", "cg_adapter", "ablation"]:
            is_gated = (self.mode == "cg_adapter") or (self.mode == "ablation" and self.ablation_type != "constant")
            for block in vit_blocks:
                adapter = BottleneckAdapter(embed_dim=768, bottleneck_rank=rank, gated=is_gated)
                self.adapters.append(adapter)

    def forward(self, image: torch.Tensor, boxes: torch.Tensor, c_prompt: torch.Tensor = None):
        if self.mode == "ablation":
            if self.ablation_type == "constant":
                c_prompt = torch.zeros_like(c_prompt) if c_prompt is not None else None
            elif self.ablation_type == "shuffled":
                if c_prompt is not None:
                    perm = torch.randperm(c_prompt.size(0))
                    c_prompt = c_prompt[perm]
            elif self.ablation_type == "sign_flipped":
                if c_prompt is not None:
                    c_prompt = -c_prompt

        if self.mode in ["standard_adapter", "cg_adapter", "ablation"]:
            x = self.base_sam.image_encoder.patch_embed(image)
            if self.base_sam.image_encoder.pos_embed is not None:
                x = x + self.base_sam.image_encoder.pos_embed
            for idx, block in enumerate(self.base_sam.image_encoder.blocks):
                x = block(x)
                if idx < len(self.adapters):
                    x = self.adapters[idx](x, c_prompt=c_prompt)
            image_embeddings = self.base_sam.image_encoder.neck(x.permute(0, 3, 1, 2))
        else:
            image_embeddings = self.base_sam.image_encoder(image)

        sparse_emb, dense_emb = self.base_sam.prompt_encoder(
            points=None,
            boxes=boxes.unsqueeze(1),
            masks=None
        )
        low_res_masks, _ = self.base_sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.base_sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False
        )
        return low_res_masks


# -------------------------------------------------------------
# 2. Losses & Metrics
# -------------------------------------------------------------

class CombinedDiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        prob = torch.sigmoid(logits).view(-1)
        tgt = target.view(-1).float()

        inter = (prob * tgt).sum()
        card = prob.sum() + tgt.sum()
        l_dice = 1.0 - (2.0 * inter + self.smooth) / (card + self.smooth)
        l_bce = F.binary_cross_entropy_with_logits(logits, target.float())

        total = l_dice + l_bce
        return total, {"loss_total": float(total.item()), "loss_dice": float(l_dice.item()), "loss_bce": float(l_bce.item())}


def compute_dice_score(p: np.ndarray, g: np.ndarray) -> float:
    p_b, g_b = p > 0, g > 0
    inter = np.logical_and(p_b, g_b).sum()
    total = p_b.sum() + g_b.sum()
    if total == 0: return 1.0
    if p_b.sum() == 0 or g_b.sum() == 0: return 0.0
    return float(2.0 * inter / total)


def compute_iou_score(p: np.ndarray, g: np.ndarray) -> float:
    p_b, g_b = p > 0, g > 0
    inter = np.logical_and(p_b, g_b).sum()
    union = np.logical_or(p_b, g_b).sum()
    if union == 0: return 1.0
    return float(inter / union)


# -------------------------------------------------------------
# 3. Contrast Proxy & Perturbation Helpers
# -------------------------------------------------------------

def extract_bbox(mask: np.ndarray):
    pos = np.argwhere(mask > 0)
    if len(pos) == 0: return None
    ymin, xmin = pos.min(axis=0)
    ymax, xmax = pos.max(axis=0)
    return (int(xmin), int(ymin), int(xmax) + 1, int(ymax) + 1)


def perturb_box(bbox, img_shape, delta, seed):
    if delta <= 0.0: return bbox
    H, W = img_shape
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    rng = np.random.default_rng(seed)
    eps = rng.uniform(-delta, delta, size=4)
    nx1 = max(0, min(W - 2, int(round(x1 + eps[0] * w))))
    ny1 = max(0, min(H - 2, int(round(y1 + eps[1] * h))))
    nx2 = max(nx1 + 2, min(W, int(round(x2 + eps[2] * w))))
    ny2 = max(ny1 + 2, min(H, int(round(y2 + eps[3] * h))))
    return (nx1, ny1, nx2, ny2)


def compute_prompt_contrast(image_rgb: np.ndarray, bbox):
    """Computes prompt CIE Lab Delta-E*ab without ground truth mask access."""
    H, W = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)

    # Core box (central 50%)
    cx1 = max(0, min(W - 1, int(round(x1 + 0.25 * w))))
    cy1 = max(0, min(H - 1, int(round(y1 + 0.25 * h))))
    cx2 = max(cx1 + 1, min(W, int(round(x2 - 0.25 * w))))
    cy2 = max(cy1 + 1, min(H, int(round(y2 - 0.25 * h))))

    # Outer skin ring (+25% expansion)
    ox1 = max(0, int(round(x1 - 0.25 * w)))
    oy1 = max(0, int(round(y1 - 0.25 * h)))
    ox2 = min(W, int(round(x2 + 0.25 * w)))
    oy2 = min(H, int(round(y2 + 0.25 * h)))

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    core_l = L[cy1:cy2, cx1:cx2].ravel()
    core_a = a[cy1:cy2, cx1:cx2].ravel()
    core_b = b[cy1:cy2, cx1:cx2].ravel()

    # Skin mask: outer box minus inner prompt box
    skin_mask = np.zeros((H, W), dtype=bool)
    skin_mask[oy1:oy2, ox1:ox2] = True
    skin_mask[y1:y2, x1:x2] = False

    skin_l = L[skin_mask]
    skin_a = a[skin_mask]
    skin_b = b[skin_mask]

    if len(core_l) == 0 or len(skin_l) == 0:
        return 21.43

    med_core_l, med_core_a, med_core_b = np.median(core_l), np.median(core_a), np.median(core_b)
    med_skin_l, med_skin_a, med_skin_b = np.median(skin_l), np.median(skin_a), np.median(skin_b)

    dl = med_core_l - med_skin_l
    da = med_core_a - med_skin_a
    db = med_core_b - med_skin_b

    delta_e = float(np.sqrt(dl * dl + da * da + db * db))
    return delta_e


# -------------------------------------------------------------
# 4. Dataset
# -------------------------------------------------------------

class KaggleISICDataset(Dataset):
    def __init__(self, mask_paths, img_dir, is_train=True, perturb_prob=0.5):
        self.mask_paths = mask_paths
        self.img_dir = img_dir
        self.is_train = is_train
        self.perturb_prob = perturb_prob

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        m_path = self.mask_paths[idx]
        base_id = os.path.basename(m_path).replace("_segmentation.png", "").replace(".png", "")

        mask_gray = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        gt_binary = (mask_gray > 127).astype(np.uint8)
        H, W = gt_binary.shape
        bbox = extract_bbox(gt_binary)
        if bbox is None:
            bbox = (10, 10, W - 10, H - 10)

        img_path = os.path.join(self.img_dir, f"{base_id}.jpg") if self.img_dir else None
        if img_path and os.path.exists(img_path):
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            img_rgb[gt_binary == 0] = [210, 180, 160]
            img_rgb[gt_binary == 1] = [60, 45, 40]

        # Training jitter
        if self.is_train and np.random.rand() < self.perturb_prob:
            delta = float(np.random.uniform(0.01, 0.20))
            p_box = perturb_box(bbox, (H, W), delta, seed=int(np.random.randint(1, 1000000)))
        else:
            p_box = bbox

        delta_e = compute_prompt_contrast(img_rgb, p_box)
        norm_contrast = (delta_e - 21.43) / 11.20

        # Scale box to 1024
        sx, sy = 1024.0 / W, 1024.0 / H
        s_box = torch.tensor([p_box[0] * sx, p_box[1] * sy, p_box[2] * sx, p_box[3] * sy], dtype=torch.float32)

        # Preprocess image
        resized_img = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5

        # Resize mask to 256x256
        resized_mask = cv2.resize(gt_binary, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_t = torch.from_numpy(resized_mask).unsqueeze(0).float()

        return {
            "image": img_t,
            "box": s_box,
            "contrast": torch.tensor([norm_contrast], dtype=torch.float32),
            "mask": mask_t,
            "image_id": base_id
        }

    def get_raw_sample(self, idx):
        m_path = self.mask_paths[idx]
        base_id = os.path.basename(m_path).replace("_segmentation.png", "").replace(".png", "")
        mask_gray = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        gt_binary = (mask_gray > 127).astype(np.uint8)
        H, W = gt_binary.shape
        bbox = extract_bbox(gt_binary)
        if bbox is None:
            bbox = (10, 10, W - 10, H - 10)
        img_path = os.path.join(self.img_dir, f"{base_id}.jpg") if self.img_dir else None
        if img_path and os.path.exists(img_path):
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            img_rgb[gt_binary == 0] = [210, 180, 160]
            img_rgb[gt_binary == 1] = [60, 45, 40]
        resized_img = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5
        return {
            "img_t": img_t,
            "img_rgb": img_rgb,
            "orig_mask": gt_binary,
            "orig_bbox": bbox,
            "orig_shape": (H, W)
        }


# -------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kaggle PEFT MedSAM Training Runner")
    parser.add_argument("--mode", type=str, default="cg_adapter", choices=["decoder_only", "lora", "standard_adapter", "cg_adapter", "ablation"])
    parser.add_argument("--ablation_type", type=str, default=None, choices=["constant", "shuffled", "sign_flipped"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    print(f"=== KAGGLE PEFT MedSAM TRAINING: {args.mode.upper()} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover MedSAM checkpoint
    ckpt_candidates = glob.glob("/kaggle/input/**/medsam_vit_b.pth", recursive=True) + glob.glob("/kaggle/input/**/*medsam*.pth", recursive=True)
    if not ckpt_candidates:
        raise FileNotFoundError("MedSAM checkpoint not found. Mount vinuchandrang/medsam-weights.")
    ckpt_path = ckpt_candidates[0]
    print(f"Loaded checkpoint from: {ckpt_path}")

    # Load Base SAM
    base_sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    model = PEFTMedSAM(
        base_sam=base_sam,
        mode=args.mode,
        rank=args.rank,
        ablation_type=args.ablation_type
    ).to(device)

    # Discover Dataset
    mask_files = sorted(glob.glob("/kaggle/input/**/ISIC2018_Task1_Training_GroundTruth/*.png", recursive=True))
    if not mask_files:
        mask_files = sorted(glob.glob("/kaggle/input/**/*_segmentation.png", recursive=True))
    print(f"Total masks found: {len(mask_files)}")

    img_dir_candidates = glob.glob("/kaggle/input/**/ISIC2018_Task1-2_Training_Input", recursive=True)
    img_dir = img_dir_candidates[0] if img_dir_candidates else None
    print(f"Image directory: {img_dir}")

    # Train / Val Split (matching seed 42 split: first 519 val, rest 2075 train)
    val_files = mask_files[:519]
    train_files = mask_files[519:2594]
    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")

    train_ds = KaggleISICDataset(train_files, img_dir, is_train=True, perturb_prob=0.50)
    val_ds = KaggleISICDataset(val_files, img_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    loss_fn = CombinedDiceBCELoss()

    history = []
    best_val_dice = -1.0

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            boxes = batch["box"].to(device, non_blocking=True)
            contrasts = batch["contrast"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                preds = model(imgs, boxes, contrasts)
                loss, l_dict = loss_fn(preds, masks)
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            train_loss += l_dict["loss_total"]

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(f"[Epoch {epoch+1}/{args.epochs}] Step {step+1}/{len(train_loader)} | "
                      f"Loss: {train_loss / (step+1):.4f} | Elapsed: {time.time()-t0:.1f}s")

        # Validation
        torch.cuda.empty_cache()
        model.eval()
        val_dices = []
        val_ious = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                boxes = batch["box"].to(device)
                contrasts = batch["contrast"].to(device)
                masks = batch["mask"].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    preds = model(imgs, boxes, contrasts)

                probs = torch.sigmoid(preds).cpu().numpy()
                gt_np = masks.cpu().numpy()

                for b in range(probs.shape[0]):
                    p_bin = (probs[b, 0] > 0.50).astype(np.uint8)
                    g_bin = (gt_np[b, 0] > 0.50).astype(np.uint8)
                    val_dices.append(compute_dice_score(p_bin, g_bin))
                    val_ious.append(compute_iou_score(p_bin, g_bin))

        mean_dice = float(np.mean(val_dices))
        mean_iou = float(np.mean(val_ious))
        print(f"=== [Epoch {epoch+1} Results] Mean Val Dice: {mean_dice:.4f} | Mean Val IoU: {mean_iou:.4f} ===")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_dice": mean_dice,
            "val_iou": mean_iou
        }
        history.append(epoch_record)

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            ckpt_out = f"/kaggle/working/best_{args.mode}_model.pth"
            trainable_state = {k: v.cpu() for k, v in model.state_dict().items() if any(p in k for p in ["adapters", "lora", "mask_decoder"])}
            torch.save({"epoch": epoch + 1, "mode": args.mode, "val_dice": best_val_dice, "state_dict": trainable_state}, ckpt_out)
            print(f"--> Saved new best checkpoint: {ckpt_out}")

    # Final Robustness Evaluation Across Perturbations on Validation Set
    print("\n=== Running Final Validation Robustness Evaluation across Prompt Jitter ===")
    robustness_records = []
    deltas = [0.0, 0.05, 0.10, 0.20]

    for delta in deltas:
        sub_dices = []
        sub_ious = []
        for idx in range(min(100, len(val_ds))):
            raw = val_ds.get_raw_sample(idx)
            orig_mask = raw["orig_mask"]
            bbox = raw["orig_bbox"]
            H, W = raw["orig_shape"]

            p_box = perturb_box(bbox, (H, W), delta, seed=idx * 7 + int(delta * 100))
            delta_e = compute_prompt_contrast(raw["img_rgb"], p_box)
            c_norm = (delta_e - 21.43) / 11.20

            sx, sy = 1024.0 / W, 1024.0 / H
            s_box = torch.tensor([[p_box[0] * sx, p_box[1] * sy, p_box[2] * sx, p_box[3] * sy]], device=device)
            img_t = raw["img_t"].unsqueeze(0).to(device)
            c_t = torch.tensor([[c_norm]], dtype=torch.float32, device=device)

            with torch.no_grad():
                pred = model(img_t, s_box, c_t)
                pred_upscaled = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
                p_bin = (torch.sigmoid(pred_upscaled).squeeze().cpu().numpy() > 0.50).astype(np.uint8)

            sub_dices.append(compute_dice_score(p_bin, orig_mask))
            sub_ious.append(compute_iou_score(p_bin, orig_mask))

        robustness_records.append({
            "delta": delta,
            "mean_dice": float(np.mean(sub_dices)),
            "mean_iou": float(np.mean(sub_ious))
        })
        print(f"Delta = {delta:.2f} | Dice = {np.mean(sub_dices):.4f} | IoU = {np.mean(sub_ious):.4f}")

    # Save metrics and report
    rob_df = pd.DataFrame(robustness_records)
    rob_df.to_csv(f"/kaggle/working/validation_robustness_{args.mode}.csv", index=False)

    with open(f"/kaggle/working/history_{args.mode}.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(f"/kaggle/working/training_report_{args.mode}.md", "w") as f:
        f.write(f"# PEFT MedSAM Training Report: {args.mode.upper()}\n\n")
        f.write(f"- Mode: `{args.mode}`\n")
        f.write(f"- Trainable Parameters: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)\n")
        f.write(f"- Best Validation Dice: **{best_val_dice:.4f}**\n\n")
        f.write("## Validation Robustness Across Prompt Perturbations\n\n")
        f.write("| Perturbation Delta | Mean Dice | Mean IoU |\n")
        f.write("| :--- | :--- | :--- |\n")
        for r in robustness_records:
            f.write(f"| {r['delta']:.2f} | {r['mean_dice']:.4f} | {r['mean_iou']:.4f} |\n")

    print(f"\nExecution finished successfully for {args.mode}!")


if __name__ == "__main__":
    main()
