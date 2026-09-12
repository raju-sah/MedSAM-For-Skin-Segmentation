"""Unit tests for Phase 5 Training Pipeline (Losses, Dataset, Trainer)."""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.loss import DiceLoss, CombinedDiceBCELoss
from src.training.dataset import ISIC2018Dataset
from src.training.trainer import MedSAMTrainer, compute_binary_dice, compute_binary_iou
from src.models.peft_medsam import PEFTMedSAM


class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dice_loss_bounds_and_differentiability(self):
        """Test DiceLoss on synthetic logits and targets."""
        loss_fn = DiceLoss(smooth=1e-5, from_logits=True)

        target = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        target[:, :, 8:24, 8:24] = 1.0

        # Perfect prediction: very high positive logits where target=1, negative where target=0
        pred_perfect = torch.ones_like(target) * -10.0
        pred_perfect[:, :, 8:24, 8:24] = 10.0
        pred_perfect.requires_grad = True

        loss_perfect = loss_fn(pred_perfect, target)
        self.assertLess(loss_perfect.item(), 0.05)

        # Opposite prediction: inverted
        pred_bad = -pred_perfect.detach()
        loss_bad = loss_fn(pred_bad, target)
        self.assertGreater(loss_bad.item(), 0.90)

        # Backward test
        loss_perfect.backward()
        self.assertIsNotNone(pred_perfect.grad)
        self.assertFalse(torch.isnan(pred_perfect.grad).any())

    def test_combined_loss(self):
        """Test CombinedDiceBCELoss outputs and dict format."""
        loss_fn = CombinedDiceBCELoss(lambda_dice=1.0, lambda_bce=1.0)
        pred = torch.randn((2, 1, 16, 16), requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        total_loss, metrics = loss_fn(pred, target)
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_dice", metrics)
        self.assertIn("loss_bce", metrics)
        self.assertGreater(total_loss.item(), 0.0)

        total_loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_dataset_generation_and_tensor_shapes(self):
        """Test ISIC2018Dataset item formatting and contrast computation."""
        dataset = ISIC2018Dataset(
            manifest_csv="non_existent_manifest.csv",  # triggers fallback
            split="train",
            max_samples=4,
            perturb_prob=1.0,  # force perturbation
            target_size=(1024, 1024),
            mask_size=(256, 256)
        )
        self.assertEqual(len(dataset), 4)

        item = dataset[0]
        self.assertEqual(item["image"].shape, (3, 1024, 1024))
        self.assertEqual(item["box"].shape, (4,))
        self.assertEqual(item["contrast"].shape, (1,))
        self.assertEqual(item["mask"].shape, (1, 256, 256))
        self.assertTrue(torch.isfinite(item["contrast"]).all())
        self.assertGreaterEqual(item["delta"], 0.0)

    def test_trainer_on_mock_model(self):
        """Test MedSAMTrainer execution on MockMedSAMModel for 1 epoch."""
        train_ds = ISIC2018Dataset(
            manifest_csv="non_existent.csv",
            split="train",
            max_samples=4,
            target_size=(1024, 1024),
            mask_size=(256, 256)
        )
        val_ds = ISIC2018Dataset(
            manifest_csv="non_existent.csv",
            split="val",
            max_samples=2,
            target_size=(1024, 1024),
            mask_size=(256, 256)
        )

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = PEFTMedSAM(base_medsam=None, mode="cg_adapter", bottleneck_rank=16)

        trainer = MedSAMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            grad_accum_steps=1,
            checkpoint_dir=self.temp_dir,
            amp_enabled=False,
            device=torch.device("cpu"),
            log_interval=1
        )

        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertIn("train_loss", history[0])
        self.assertIn("val_dice", history[0])
        self.assertIn("val_iou", history[0])

        # Verify checkpoint file was written
        ckpt_files = os.listdir(self.temp_dir)
        self.assertTrue(any(f.endswith(".pth") for f in ckpt_files))


if __name__ == "__main__":
    unittest.main()
