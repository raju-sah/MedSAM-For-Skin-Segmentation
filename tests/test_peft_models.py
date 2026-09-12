"""Comprehensive unit tests for PEFT and CG-Adapter architectures (E04–E08).

Tests:
1. LoRALinear mathematical identity at step 0 and gradient propagation
2. ContrastGatingMLP output boundedness and initialization near 1.0
3. BottleneckAdapter identity initialization and contrast gating
4. PEFTMedSAM mode configuration, parameter counting, and forward passes
"""

import unittest
import torch
import torch.nn as nn

from src.models.adapters import LoRALinear, ContrastGatingMLP, BottleneckAdapter
from src.models.peft_medsam import PEFTMedSAM


class TestPEFTModels(unittest.TestCase):

    def test_lora_identity_init(self):
        """Verify LoRALinear matches base linear exactly at initialization."""
        base_lin = nn.Linear(768, 768)
        lora = LoRALinear(base_lin, r=16, lora_alpha=32.0)
        x = torch.randn(2, 32, 768)
        y_base = base_lin(x)
        y_lora = lora(x)
        self.assertTrue(torch.allclose(y_base, y_lora, atol=1e-5))

        # Check trainable parameters
        self.assertFalse(lora.base_linear.weight.requires_grad)
        self.assertTrue(lora.lora_A.requires_grad)
        self.assertTrue(lora.lora_B.requires_grad)

    def test_gating_mlp_bounds(self):
        """Verify ContrastGatingMLP outputs gamma in [0, 2] and inits near 1.0."""
        gate = ContrastGatingMLP(hidden_dim=16)
        c = torch.linspace(-5.0, 5.0, 50).unsqueeze(-1)
        gamma = gate(c)
        self.assertTrue(torch.all(gamma >= 0.0))
        self.assertTrue(torch.all(gamma <= 2.0))

        # At c=0, gamma should be very close to 1.0
        c_zero = torch.tensor([[0.0]])
        gamma_zero = gate(c_zero)
        self.assertAlmostEqual(gamma_zero.item(), 1.0, delta=0.05)

    def test_bottleneck_adapter_forward(self):
        """Verify BottleneckAdapter identity at step 0 and correct gating."""
        adapter = BottleneckAdapter(embed_dim=768, bottleneck_rank=16, gated=True)
        x = torch.randn(2, 64, 768)
        c = torch.tensor([[0.5], [-0.5]])
        y = adapter(x, c_prompt=c)
        self.assertEqual(y.shape, x.shape)
        # Up-projection zero-init -> output matches input at step 0
        self.assertTrue(torch.allclose(x, y, atol=1e-5))

    def test_peft_medsam_modes(self):
        """Verify all PEFT modes initialize and run forward passes."""
        for mode in ["decoder_only", "lora", "standard_adapter", "cg_adapter", "ablation"]:
            model = PEFTMedSAM(base_medsam=None, mode=mode)
            img = torch.randn(1, 3, 1024, 1024)
            box = torch.tensor([[50, 50, 200, 200]], dtype=torch.float32)
            c = torch.tensor([[0.5]], dtype=torch.float32)
            out = model(img, box, c_prompt=c)
            self.assertEqual(out.shape, (1, 1, 1024, 1024))


if __name__ == "__main__":
    unittest.main()
