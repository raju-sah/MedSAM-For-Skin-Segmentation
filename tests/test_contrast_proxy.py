"""Unit tests for contrast proxy extraction and numerical safety using standard unittest."""

import unittest
import numpy as np
from src.data.contrast_proxy import (
    get_lesion_core_proxy,
    get_perilesional_background_proxy_mask,
    rgb_to_lab,
    calculate_ita,
    compute_contrast_proxy,
    audit_proxy_composition,
    ContrastResult
)


class TestContrastProxy(unittest.TestCase):

    def test_core_proxy_geometry(self):
        """Test erosion geometry preserves center and respects bounds."""
        bbox = (100, 100, 200, 200)  # 100x100 box
        core = get_lesion_core_proxy(bbox, erosion_alpha=0.50, img_shape=(300, 300))
        c_xmin, c_ymin, c_xmax, c_ymax = core

        # Core should be 50x50 centered at (150, 150)
        self.assertEqual(c_xmin, 125)
        self.assertEqual(c_ymin, 125)
        self.assertEqual(c_xmax, 175)
        self.assertEqual(c_ymax, 175)
        self.assertEqual(c_xmax - c_xmin, 50)
        self.assertEqual(c_ymax - c_ymin, 50)

    def test_perilesional_skin_proxy_disjointness(self):
        """Test that perilesional skin proxy is strictly outside the bounding box."""
        img_shape = (200, 200)
        bbox = (50, 50, 150, 150)
        skin_mask = get_perilesional_background_proxy_mask(bbox, img_shape, margin_beta=0.25)

        # Pixels inside the bounding box must be False in skin_mask
        inside_box = skin_mask[50:150, 50:150]
        self.assertTrue(np.all(~inside_box), "Skin proxy must not overlap bounding box interior")

        # Pixels in the expanded margin must be True
        self.assertTrue(np.any(skin_mask), "Skin proxy must have positive area")
        self.assertTrue(skin_mask[30, 100])
        self.assertTrue(skin_mask[160, 100])

    def test_degenerate_bounding_box(self):
        """Test handling of inverted, zero-area, or out-of-bounds bounding boxes."""
        dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Inverted box
        res1 = compute_contrast_proxy(dummy_img, (50, 50, 30, 30))
        self.assertFalse(res1.is_valid)
        self.assertIn("Degenerate", res1.warning_message)

        # Zero area box
        res2 = compute_contrast_proxy(dummy_img, (50, 50, 50, 50))
        self.assertFalse(res2.is_valid)

    def test_ita_numerical_safety_singular_b(self):
        """Test ITA calculation with b* close to zero (preventing division by zero)."""
        l_vals = np.array([20.0, 50.0, 70.0, 80.0], dtype=np.float32)
        b_vals = np.array([0.0, 1e-6, -1e-6, 15.0], dtype=np.float32)

        ita, valid_mask, sing_count = calculate_ita(l_vals, b_vals, b_star_epsilon=1e-4)

        self.assertEqual(sing_count, 3)
        self.assertEqual(ita[0], -90.0)
        self.assertEqual(ita[1], 90.0)
        self.assertEqual(ita[2], 90.0)
        expected_standard = np.arctan2(80.0 - 50.0, 15.0) * (180.0 / np.pi)
        self.assertTrue(np.isclose(ita[3], expected_standard, atol=1e-2))

    def test_ita_luminance_filtering(self):
        """Test filtering of deep shadows (L* < 10) and specular glare (L* > 95)."""
        l_vals = np.array([5.0, 98.0, 60.0], dtype=np.float32)
        b_vals = np.array([10.0, 10.0, 10.0], dtype=np.float32)

        ita, valid_mask, _ = calculate_ita(l_vals, b_vals, min_l=10.0, max_l=95.0)
        self.assertEqual(len(ita), 1)
        self.assertFalse(valid_mask[0])
        self.assertFalse(valid_mask[1])
        self.assertTrue(valid_mask[2])

    def test_zero_leakage_api(self):
        """Verify that compute_contrast_proxy strictly takes (image, box) and no mask."""
        img = np.full((100, 100, 3), 180, dtype=np.uint8)
        img[30:70, 30:70] = 50

        bbox = (30, 30, 70, 70)
        res = compute_contrast_proxy(img, bbox)

        self.assertIsInstance(res, ContrastResult)
        self.assertTrue(res.is_valid)
        self.assertGreater(res.delta_l, 0)
        self.assertGreater(res.delta_ita, 0)
        self.assertGreater(res.delta_e_ab, 0)

    def test_audit_proxy_composition(self):
        """Test audit-only function with synthetic multi-class sDDI mask."""
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[30:70, 30:70] = 1  # Lesion
        mask[10:30, 10:90] = 4  # Skin
        mask[22:28, 22:28] = 2  # Marker in skin proxy halo (pad is 10px, so [20, 30] is in halo)

        bbox = (30, 30, 70, 70)
        core = get_lesion_core_proxy(bbox, erosion_alpha=0.5)
        skin_mask = get_perilesional_background_proxy_mask(bbox, (100, 100), margin_beta=0.25)

        purity = audit_proxy_composition(mask, core, skin_mask, is_sddi=True)

        self.assertEqual(purity["core_lesion_purity"], 1.0)
        self.assertGreater(purity["marker_contamination"], 0.0)
        self.assertLess(purity["skin_proxy_purity"], 1.0)


if __name__ == "__main__":
    unittest.main()
