# LCR-IH

Local-Cropout-Robust Image Hiding via Redundant Auxiliary Embedding and Mask-Aware Secret Enhancement

This repository contains a public research code scaffold for the LCR-IH project.

## Project overview

This codebase is organized for image-hiding experiments under local crop attacks. It includes:

- training scripts
- evaluation scripts
- dataset loading utilities
- public configuration files
- metric utilities

## Repository structure

```text
LCR-IH/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── critic.py
├── config_finetune_dvi2k.py
├── config_finetune.py
├── config_finetune_coco.py
├── datasets.py
├── train_StegFormer_single_image_finetune.py
├── test_save_single_image_hiding_finetune_coco.py
├── src/
│   ├── __init__.py
│   └── ...
├── configs/
│   └── README.md
├── scripts/
│   └── README.md
├── docs/
│   └── installation.md
├── examples/
│   └── README.md
└── notebooks/
    └── demo.ipynb
```

## Data and paths

To protect user privacy, all machine-specific absolute paths have been removed from the public repository. Before running the code, please prepare your own data folders and configure the corresponding directories in the config files.

Recommended directory layout:

```text
LCR-IH/
data/
├── DIV2K_train/
├── DIV2K_valid/
├── COCO/
└── outputs/
```

You can set paths like:

```python
DIV2K_train_dir = "./data/DIV2K_train"
DIV2K_valid_dir = "./data/DIV2K_valid"
COCO_dir = "./data/COCO"
output_dir = "./outputs"
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

1. Prepare data in `./data/`.
2. Modify the config file paths in `config_finetune_dvi2k.py` or `config_finetune_coco.py`.
3. Train:

```bash
python train_StegFormer_single_image_finetune.py --config_module config_finetune_dvi2k
```

4. Evaluate:

```bash
python test_save_single_image_hiding_finetune_coco.py --config config_finetune_coco
```

## Notes

- This repository intentionally avoids embedding personal or machine-specific paths.
- Private paths such as cluster directories or local server directories should be configured locally by the user before running experiments.
- The repository is intended as a clean public-facing code scaffold for research sharing.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
