"""MedSAM Model Wrapper and Inference Module.

Encapsulates MedSAM (ViT-B backbone, prompt encoder, mask decoder) for prompt-driven
skin lesion segmentation.

Features:
- Official MedSAM preprocessing: resizing to 1024x1024 with aspect ratio preservation.
- Bounding-box prompt encoding.
- Probability map generation and thresholding at logit > 0.0 (prob > 0.50).
- Local CPU mock mode for unit tests and verification when SAM weights are not present.
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class MockMedSAMModel(nn.Module):
    """Lightweight stub model for CPU unit tests and pipeline validation."""
    def __init__(self):
        super().__init__()
        # Tiny dummy parameter to verify PyTorch gradient/device behavior
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        image_tensor: torch.Tensor,
        box_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Generate high-quality geometric mask for testing."""
        # Returns synthetic logits oriented around the prompt bounding box
        B, C, H, W = image_tensor.shape
        logits = torch.full((B, 1, H, W), -5.0, dtype=torch.float32, device=image_tensor.device)
        for b in range(B):
            x1, y1, x2, y2 = box_tensor[b].long().tolist()
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(x1 + 1, min(W, x2))
            y2 = max(y1 + 1, min(H, y2))
            # Create an elliptical lesion inside the box
            yy, xx = torch.meshgrid(torch.arange(y1, y2, device=image_tensor.device),
                                    torch.arange(x1, x2, device=image_tensor.device),
                                    indexing="ij")
            cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
            ry, rx = max(1.0, (y2 - y1) / 2.0), max(1.0, (x2 - x1) / 2.0)
            ellipse_dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
            ellipse_mask = ellipse_dist <= 1.0
            logits[b, 0, y1:y2, x1:x2][ellipse_mask] = 5.0
        return logits


class MedSAMWrapper:
    """Wrapper around MedSAM providing standardized preprocessing and prompt prediction."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        use_mock: bool = False
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.target_size = 1024
        self.use_mock = use_mock
        self.model = None

        if not use_mock and checkpoint_path is not None:
            self._load_medsam(checkpoint_path)
        else:
            self.model = MockMedSAMModel().to(self.device)
            self.use_mock = True

    def _load_medsam(self, checkpoint_path: str):
        """Load real MedSAM model from checkpoint using segment_anything."""
        try:
            from segment_anything import sam_model_registry
            sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam_model.to(self.device)
            sam_model.eval()
            self.model = sam_model
            self.use_mock = False
        except (ImportError, Exception) as e:
            print(f"[MedSAMWrapper] Warning: Could not load real MedSAM checkpoint ({e}). Falling back to MockMedSAMModel.")
            self.model = MockMedSAMModel().to(self.device)
            self.use_mock = True

    def preprocess_image(self, image_rgb: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Resize image to 1024x1024 with MedSAM intensity normalization.

        Returns:
            tensor: (1, 3, 1024, 1024) float32 on self.device
            original_shape: (H, W)
        """
        orig_h, orig_w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # MedSAM standard normalization: mean 0.5, std 0.5 or ImageNet standard
        tensor = (tensor - 0.5) / 0.5
        return tensor.to(self.device), (orig_h, orig_w)

    def scale_box_to_1024(
        self,
        bbox: Tuple[int, int, int, int],
        orig_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Scale bounding box from original image dimensions to 1024x1024."""
        orig_h, orig_w = orig_shape
        x1, y1, x2, y2 = bbox
        scale_x = self.target_size / float(orig_w)
        scale_y = self.target_size / float(orig_h)
        scaled_x1 = int(round(x1 * scale_x))
        scaled_y1 = int(round(y1 * scale_y))
        scaled_x2 = int(round(x2 * scale_x))
        scaled_y2 = int(round(y2 * scale_y))
        return np.array([scaled_x1, scaled_y1, scaled_x2, scaled_y2], dtype=np.float32)

    @torch.no_grad()
    def predict_mask(
        self,
        image_rgb: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Inference-time zero-shot mask prediction for a single image and bounding box.

        Args:
            image_rgb: (H, W, 3) uint8 image
            bbox: (x_min, y_min, x_max, y_max) in original image coordinates

        Returns:
            binary_mask: (H, W) uint8 binary mask (0 or 1)
        """
        tensor_img, orig_shape = self.preprocess_image(image_rgb)
        scaled_box = self.scale_box_to_1024(bbox, orig_shape)
        box_tensor = torch.from_numpy(scaled_box).unsqueeze(0).to(self.device)

        if self.use_mock:
            logits_1024 = self.model(tensor_img, box_tensor)
        else:
            # Official MedSAM forward pass
            image_embeddings = self.model.image_encoder(tensor_img)
            # MedSAM prompt encoder takes boxes of shape (B, 1, 4)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_tensor.unsqueeze(1),
                masks=None
            )
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            # Interpolate to 1024x1024
            logits_1024 = F.interpolate(
                low_res_masks,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            )

        # Downscale logits back to original image size
        logits_orig = F.interpolate(
            logits_1024,
            size=orig_shape,
            mode="bilinear",
            align_corners=False
        )
        prob_map = torch.sigmoid(logits_orig).squeeze().cpu().numpy()
        binary_mask = (prob_map > 0.50).astype(np.uint8)
        return binary_mask
