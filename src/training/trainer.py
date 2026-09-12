"""Trainer Engine for PEFT MedSAM Adaptation.

Supports all PEFT variants:
- Decoder-only (E04)
- LoRA (E05)
- Standard Bottleneck Adapter (E06)
- Contrast-Gated Adapter (E07)
- Causal Ablations (E08)

Features:
- Mixed precision training (torch.amp.autocast)
- Gradient accumulation and gradient clipping
- Combined Dice + BCE loss optimization
- Validation tracking (Dice, IoU, loss)
- Best checkpoint saving and detailed metrics logging
"""

import os
import json
import time
from typing import Dict, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.peft_medsam import PEFTMedSAM
from src.training.loss import CombinedDiceBCELoss


def compute_binary_dice(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """Compute binary Dice coefficient."""
    p_b = pred_binary > 0
    g_b = gt_binary > 0
    inter = np.logical_and(p_b, g_b).sum()
    total = p_b.sum() + g_b.sum()
    if total == 0:
        return 1.0
    if p_b.sum() == 0 or g_b.sum() == 0:
        return 0.0
    return float(2.0 * inter / total)


def compute_binary_iou(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """Compute binary IoU."""
    p_b = pred_binary > 0
    g_b = gt_binary > 0
    inter = np.logical_and(p_b, g_b).sum()
    union = np.logical_or(p_b, g_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


class MedSAMTrainer:
    """Trainer for PEFT MedSAM models."""

    def __init__(
        self,
        model: PEFTMedSAM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 10,
        grad_accum_steps: int = 2,
        max_grad_norm: float = 1.0,
        amp_enabled: bool = True,
        checkpoint_dir: str = "checkpoints",
        device: Optional[torch.device] = None,
        log_interval: int = 20
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        os.makedirs(checkpoint_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.amp_enabled = amp_enabled and (self.device.type == "cuda")

        self.loss_fn = CombinedDiceBCELoss(lambda_dice=1.0, lambda_bce=1.0)

        # Optimize only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found in model!")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs * len(train_loader),
            eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)

        self.history: List[Dict[str, Any]] = []
        self.best_val_dice = -1.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice_loss = 0.0
        total_bce_loss = 0.0
        n_batches = len(self.train_loader)

        self.optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            boxes = batch["box"].to(self.device, non_blocking=True)
            contrasts = batch["contrast"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                pred_logits = self.model(images, boxes, contrasts)

                # Interpolate if predicted size doesn't match target mask
                if pred_logits.shape[-2:] != masks.shape[-2:]:
                    pred_logits = F.interpolate(
                        pred_logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                loss, loss_dict = self.loss_fn(pred_logits, masks)
                loss_scaled = loss / self.grad_accum_steps

            self.scaler.scale(loss_scaled).backward()

            total_loss += loss_dict["loss_total"]
            total_dice_loss += loss_dict["loss_dice"]
            total_bce_loss += loss_dict["loss_bce"]

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == n_batches:
                self.scaler.unscale_(self.optimizer)
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            if (step + 1) % self.log_interval == 0 or (step + 1) == n_batches:
                avg_step_loss = total_loss / (step + 1)
                lr_curr = self.optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch+1}/{self.epochs}] Step {step+1}/{n_batches} | "
                      f"Loss: {avg_step_loss:.4f} | LR: {lr_curr:.6f} | "
                      f"Elapsed: {time.time()-t0:.1f}s")

        return {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, n_batches),
            "train_dice_loss": total_dice_loss / max(1, n_batches),
            "train_bce_loss": total_bce_loss / max(1, n_batches),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "time_sec": round(time.time() - t0, 2)
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}

        self.model.eval()
        val_loss = 0.0
        dices = []
        ious = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                boxes = batch["box"].to(self.device)
                contrasts = batch["contrast"].to(self.device)
                masks = batch["mask"].to(self.device)

                with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                    pred_logits = self.model(images, boxes, contrasts)

                    if pred_logits.shape[-2:] != masks.shape[-2:]:
                        pred_logits = F.interpolate(
                            pred_logits,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False
                        )

                    loss, _ = self.loss_fn(pred_logits, masks)
                    val_loss += loss.item()

                probs = torch.sigmoid(pred_logits).cpu().numpy()
                gt_np = masks.cpu().numpy()

                for b in range(probs.shape[0]):
                    pred_bin = (probs[b, 0] > 0.50).astype(np.uint8)
                    gt_bin = (gt_np[b, 0] > 0.50).astype(np.uint8)
                    dices.append(compute_binary_dice(pred_bin, gt_bin))
                    ious.append(compute_binary_iou(pred_bin, gt_bin))

        return {
            "val_loss": float(val_loss / max(1, len(self.val_loader))),
            "val_dice": float(np.mean(dices)) if dices else 0.0,
            "val_iou": float(np.mean(ious)) if ious else 0.0
        }

    def train(self) -> List[Dict[str, Any]]:
        """Execute full training and validation pipeline."""
        print(f"=== Starting Training: Mode={self.model.mode} | Device={self.device} | Epochs={self.epochs} ===")
        param_counts = self.model.count_parameters()
        print(f"Trainable Parameters: {param_counts['trainable_parameters']:,} / {param_counts['total_parameters']:,} "
              f"({param_counts['trainable_fraction_percent']:.2f}%)")

        for epoch in range(self.epochs):
            epoch_metrics = self.train_epoch(epoch)

            if self.val_loader is not None:
                val_metrics = self.evaluate()
                epoch_metrics.update(val_metrics)
                print(f"[Epoch {epoch+1} Val] Loss: {val_metrics['val_loss']:.4f} | "
                      f"Dice: {val_metrics['val_dice']:.4f} | IoU: {val_metrics['val_iou']:.4f}")

                val_dice = val_metrics.get("val_dice", 0.0)
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self._save_checkpoint(is_best=True, epoch=epoch + 1, metrics=epoch_metrics)
                    print(f"--> Saved new BEST checkpoint (Dice: {val_dice:.4f})")

            self.history.append(epoch_metrics)
            self._save_checkpoint(is_best=False, epoch=epoch + 1, metrics=epoch_metrics)

        # Save history JSON
        hist_path = os.path.join(self.checkpoint_dir, f"history_{self.model.mode}.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {hist_path}")

        return self.history

    def _save_checkpoint(self, is_best: bool, epoch: int, metrics: Dict[str, Any]):
        """Save model weights and training state."""
        # Extract trainable state dict
        trainable_state = {
            k: v.cpu() for k, v in self.model.state_dict().items()
            if any(p in k for p in ["adapters", "lora", "mask_decoder", "mock_adapter", "mock_lora"])
        }

        ckpt = {
            "epoch": epoch,
            "mode": self.model.mode,
            "rank": self.model.rank,
            "lora_alpha": self.model.lora_alpha,
            "ablation_type": self.model.ablation_type,
            "metrics": metrics,
            "state_dict": trainable_state
        }

        if is_best:
            fn = f"best_{self.model.mode}_model.pth"
        else:
            fn = f"latest_{self.model.mode}_model.pth"

        path = os.path.join(self.checkpoint_dir, fn)
        torch.save(ckpt, path)
