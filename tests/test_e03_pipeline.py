"""Comprehensive unit tests for E03 Zero-Shot MedSAM evaluation pipeline.

Tests:
1. Perturbation reproducibility and bounds enforcement
2. Segmentation metrics (Dice, IoU, HD95, NSD)
3. MedSAMWrapper preprocessing, scaling, and mock prediction
4. End-to-end zero-shot evaluation pipeline execution
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil

from src.data.perturbation import perturb_bounding_box, generate_prompt_realizations
from src.metrics.segmentation_metrics import (
    compute_dice,
    compute_iou,
    compute_hd95,
    compute_nsd,
    evaluate_segmentation_pair
)
from src.models.medsam_wrapper import MedSAMWrapper
from src.eval.zero_shot_eval import run_zero_shot_evaluation


class TestE03Pipeline(unittest.TestCase):

    def test_perturbation_reproducibility(self):
        """Verify that identical seeds produce identical bounding box perturbations."""
        bbox = (30, 40, 120, 150)
        img_shape = (200, 300)
        box1 = perturb_bounding_box(bbox, img_shape, delta=0.10, seed=42)
        box2 = perturb_bounding_box(bbox, img_shape, delta=0.10, seed=42)
        box3 = perturb_bounding_box(bbox, img_shape, delta=0.10, seed=99)

        self.assertEqual(box1, box2)
        self.assertNotEqual(box1, box3)

    def test_perturbation_bounds_enforcement(self):
        """Verify that perturbations never violate image dimensions or create degenerate boxes."""
        bbox = (2, 2, 8, 8)
        img_shape = (50, 50)
        for seed in range(50):
            p_box = perturb_bounding_box(bbox, img_shape, delta=0.20, seed=seed)
            x1, y1, x2, y2 = p_box
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 50)
            self.assertLessEqual(y2, 50)
            self.assertGreaterEqual(x2 - x1, 2)
            self.assertGreaterEqual(y2 - y1, 2)

    def test_segmentation_metrics_identical_masks(self):
        """Verify metric outputs for identical foreground masks."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:50, 20:50] = 1
        res = evaluate_segmentation_pair(mask, mask)
        self.assertEqual(res["dice"], 1.0)
        self.assertEqual(res["iou"], 1.0)
        self.assertEqual(res["hd95"], 0.0)
        self.assertEqual(res["nsd"], 1.0)

    def test_segmentation_metrics_disjoint_masks(self):
        """Verify metric outputs for disjoint masks."""
        m1 = np.zeros((80, 80), dtype=np.uint8)
        m2 = np.zeros((80, 80), dtype=np.uint8)
        m1[10:30, 10:30] = 1
        m2[50:70, 50:70] = 1
        res = evaluate_segmentation_pair(m1, m2)
        self.assertEqual(res["dice"], 0.0)
        self.assertEqual(res["iou"], 0.0)
        self.assertGreater(res["hd95"], 0.0)
        self.assertEqual(res["nsd"], 0.0)

    def test_medsam_wrapper_mock_prediction(self):
        """Verify MedSAMWrapper preprocessing and mock prediction."""
        model = MedSAMWrapper(use_mock=True)
        img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        bbox = (30, 25, 90, 85)

        mask = model.predict_mask(img, bbox)
        self.assertEqual(mask.shape, (120, 160))
        self.assertTrue(np.all(np.isin(mask, [0, 1])))
        self.assertGreater(np.sum(mask), 0)

    def test_zero_shot_eval_mock_pipeline(self):
        """Verify end-to-end zero-shot evaluation pipeline execution on sample subset."""
        temp_dir = tempfile.mkdtemp()
        try:
            res = run_zero_shot_evaluation(
                use_mock=True,
                max_samples=2,
                output_dir=temp_dir
            )
            self.assertTrue(os.path.exists(res["results_csv"]))
            self.assertTrue(os.path.exists(res["report_md"]))
            self.assertGreater(res["total_evaluations"], 0)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
