"""Unit tests for Generalized Multi-Modal Contrast Proxy Extraction."""

import unittest
import numpy as np

from src.data.generalized_contrast_proxy import (
    extract_prompt_regions,
    compute_dermatology_contrast,
    compute_endoscopy_contrast,
    compute_ultrasound_contrast,
    compute_generalized_contrast_proxy
)


class TestGeneralizedContrastProxy(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic RGB image with distinct inner and outer zones
        self.rgb_img = np.full((120, 160, 3), 180, dtype=np.uint8)
        self.rgb_img[40:80, 50:110] = [80, 40, 30]  # Dark pigmented inner target
        self.bbox = (50, 40, 110, 80)

        # Synthetic grayscale ultrasound image (hypoechoic lesion in speckled tissue)
        self.us_img = np.random.normal(140, 15, (120, 160)).astype(np.float32)
        self.us_img[35:75, 45:95] = np.random.normal(60, 10, (40, 50))  # Hypoechoic
        self.us_img = np.clip(self.us_img, 0, 255).astype(np.uint8)
        self.us_bbox = (45, 35, 95, 75)

    def test_extract_prompt_regions_shapes(self):
        """Verify inner and outer region extraction shapes and non-emptiness."""
        inner, outer = extract_prompt_regions(self.rgb_img, self.bbox)
        self.assertGreater(len(inner), 0)
        self.assertGreater(len(outer), 0)
        self.assertEqual(inner.shape[1], 3)
        self.assertEqual(outer.shape[1], 3)

    def test_dermatology_contrast_bounds(self):
        """Verify dermatology CIE Lab contrast calculation and non-zero response."""
        c = compute_dermatology_contrast(self.rgb_img, self.bbox)
        self.assertIsInstance(c, float)
        self.assertGreater(c, 0.0)
        self.assertLessEqual(c, 100.0)

    def test_identical_image_zero_contrast(self):
        """Verify that a uniform image yields zero contrast across all modalities."""
        flat_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        box = (20, 20, 80, 80)

        c_derm = compute_dermatology_contrast(flat_rgb, box)
        c_endo = compute_endoscopy_contrast(flat_rgb, box)
        c_us = compute_ultrasound_contrast(flat_rgb, box)

        self.assertAlmostEqual(c_derm, 0.0, places=3)
        self.assertAlmostEqual(c_endo, 0.0, places=3)
        self.assertAlmostEqual(c_us, 0.0, places=3)

    def test_endoscopy_contrast_hemoglobin_sensitivity(self):
        """Verify endoscopy contrast responds strongly to mucosal vascular erythema."""
        # Baseline pinkish mucosa
        endo_img = np.full((100, 100, 3), [190, 110, 110], dtype=np.uint8)
        # Hypervascular adenomatous polyp (elevated red channel)
        endo_img[30:70, 30:70] = [240, 60, 60]
        box = (30, 30, 70, 70)

        c_endo = compute_endoscopy_contrast(endo_img, box)
        self.assertGreater(c_endo, 5.0)
        self.assertLessEqual(c_endo, 100.0)

    def test_ultrasound_contrast_hypoechoic_lesion(self):
        """Verify ultrasound contrast captures hypoechogenicity differential."""
        c_us = compute_ultrasound_contrast(self.us_img, self.us_bbox)
        self.assertGreater(c_us, 5.0)
        self.assertLessEqual(c_us, 100.0)

    def test_generalized_dispatcher_routing(self):
        """Verify dispatcher properly routes modalities and raises on unknown string."""
        c1 = compute_generalized_contrast_proxy(self.rgb_img, self.bbox, modality="dermatology")
        c2 = compute_generalized_contrast_proxy(self.rgb_img, self.bbox, modality="endoscopy")
        c3 = compute_generalized_contrast_proxy(self.us_img, self.us_bbox, modality="ultrasound")

        self.assertGreater(c1, 0.0)
        self.assertGreater(c2, 0.0)
        self.assertGreater(c3, 0.0)

        with self.assertRaises(ValueError):
            compute_generalized_contrast_proxy(self.rgb_img, self.bbox, modality="magnetic_resonance_fmri")

    def test_degenerate_bounding_box_safety(self):
        """Verify proxy extraction handles degenerate or near-border bounding boxes gracefully."""
        border_box = (0, 0, 2, 2)
        c = compute_generalized_contrast_proxy(self.rgb_img, border_box, modality="dermatology")
        self.assertIsInstance(c, float)
        self.assertGreaterEqual(c, 0.0)


if __name__ == "__main__":
    unittest.main()
