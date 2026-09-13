# CG-MedSAM Release v1.0.0

Official checkpoint release for **CG-MedSAM**: Contrast-Gated Parameter-Efficient MedSAM for Skin-Tone-Robust Lesion Segmentation.

## Release Assets
- **PEFT Model Archive:** `cg_medsam_peft_weights_v1.0.tar.gz` (139.4 MB)
- **Archive SHA-256:** `5b639b2811709d7bb1f51af6092881f4bf91d95a006b5774f775949baef8b44c`
- **Git Commit:** `afedfc79989864d5c2db215b151b5704be61ce39`

## Model Checksums (SHA-256)
| Model | Role | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `best_cg_adapter_model.pth` | **Proposed (Ours)** | 16.72 MB | `a09fcf2808dc101c87ea255b79a9830b4cb4c1e2e7ad1bd36e9c6756cbf73b38` |
| `best_standard_adapter_model.pth` | Baseline Adapter | 16.70 MB | `8e8e4a04acc7aa05236b34ffceb1f1168d7d48e86b15c232117e41bcca28ead3` |
| `best_lora_model.pth` | LoRA Baseline | 101.15 MB | `e8c107d3c34e8122931909adcbb05410be42c1d79665fa61a735c6c7c82ea2ac` |
| `best_decoder_only_model.pth` | Decoder Baseline | 15.52 MB | `695430f33f9772da50972ed3a363671f0625a5081804336264be2d02f4ad896f` |

## Verification Command
```bash
python scripts/verify_checkpoints.py
```
