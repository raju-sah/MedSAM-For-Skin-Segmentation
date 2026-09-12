"""Loss functions for MedSAM lesion segmentation adaptation.

Implements:
- DiceLoss: Smooth soft Dice loss computed with sigmoid probabilities
- CombinedDiceBCELoss: Weighted sum of Dice loss and Binary Cross-Entropy with logits
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss with smooth term and sigmoid activation."""

    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            pred: Predicted tensor of arbitrary shape (e.g. B, 1, H, W).
                  If from_logits is True, pred contains raw logits.
            target: Ground truth binary mask (0 or 1) of same shape as pred.

        Returns:
            Scalar Dice loss in [0, 1].
        """
        if self.from_logits:
            prob = torch.sigmoid(pred)
        else:
            prob = pred

        prob = prob.view(-1)
        target = target.view(-1).float()

        intersection = (prob * target).sum()
        cardinality = prob.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedDiceBCELoss(nn.Module):
    """Combined Soft Dice and Binary Cross-Entropy Loss."""

    def __init__(self, lambda_dice: float = 1.0, lambda_bce: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.dice_loss = DiceLoss(smooth=smooth, from_logits=True)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            pred_logits: Raw model prediction logits (B, 1, H, W).
            target: Ground truth binary mask (B, 1, H, W) or (B, H, W).

        Returns:
            Tuple of (total_loss_tensor, loss_dict_with_floats)
        """
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()

        l_dice = self.dice_loss(pred_logits, target)
        l_bce = F.binary_cross_entropy_with_logits(pred_logits, target)

        total_loss = self.lambda_dice * l_dice + self.lambda_bce * l_bce

        loss_metrics = {
            "loss_total": float(total_loss.item()),
            "loss_dice": float(l_dice.item()),
            "loss_bce": float(l_bce.item())
        }

        return total_loss, loss_metrics
