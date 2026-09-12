"""Unit tests for dataset audit modules."""

import os
import unittest
import numpy as np
from src.data.sddi_audit import audit_sddi_split
from src.data.isic_audit import generate_isic_split_manifest


class TestAuditPipeline(unittest.TestCase):

    def test_sddi_split_audit_on_mock(self):
        """Test sDDI audit function correctly summarizes mock masks."""
        tmp_dir = "/tmp/mock_sddi"
        os.makedirs(tmp_dir, exist_ok=True)

        # Create 2 mock .npy files: 256x256
        m1 = np.zeros((256, 256), dtype=np.int32)
        m1[50:100, 50:100] = 1  # lesion
        m1[0:50, 0:256] = 4    # skin
        np.save(os.path.join(tmp_dir, "000001.npy"), m1)

        m2 = np.zeros((256, 256), dtype=np.int32)
        m2[40:80, 40:80] = 1   # lesion
        m2[10:20, 10:20] = 2   # marker
        np.save(os.path.join(tmp_dir, "000002.npy"), m2)

        res = audit_sddi_split(tmp_dir)

        self.assertEqual(res["count"], 2)
        self.assertIn(1, res["classes_found"])
        self.assertIn(2, res["classes_found"])
        self.assertIn(4, res["classes_found"])
        self.assertEqual(res["marker_frequency"], 0.5)

        # Cleanup mock
        os.remove(os.path.join(tmp_dir, "000001.npy"))
        os.remove(os.path.join(tmp_dir, "000002.npy"))
        os.rmdir(tmp_dir)

    def test_isic_manifest_generation_reproducibility(self):
        """Verify that split generation with same seed produces identical splits."""
        records = [
            {"image_id": f"ISIC_{i:07d}", "height": 1000, "width": 1000, "lesion_pixels": 100, "total_pixels": 10000, "lesion_area_fraction": 0.01}
            for i in range(100)
        ]

        out1 = "/tmp/mock_manifest1.csv"
        out2 = "/tmp/mock_manifest2.csv"

        info1 = generate_isic_split_manifest(records, out1, random_seed=42, train_ratio=0.80)
        info2 = generate_isic_split_manifest(records, out2, random_seed=42, train_ratio=0.80)

        self.assertEqual(info1["train_count"], 80)
        self.assertEqual(info1["val_count"], 20)

        with open(out1) as f1, open(out2) as f2:
            self.assertEqual(f1.read(), f2.read())

        # Cleanup
        os.remove(out1)
        os.remove(out2)


if __name__ == "__main__":
    unittest.main()
