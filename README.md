# LCR-IH

Local-Cropout-Robust Image Hiding via Redundant Auxiliary Embedding and Mask-Aware Secret Enhancement

This repository contains the public code release scaffold for the LCR-IH project, including:

- training and evaluation scripts
- dataset utilities
- experiment configuration for DIV2K-based robustness tests
- project organization for open-sourcing a research codebase

## Paper summary

Local cropping is a common degradation in practical image transmission that may remove part of a stego-image and severely compromise hidden information. LCR-IH addresses this problem by combining redundant auxiliary embedding with mask-aware secret enhancement to improve robustness against local cropping while preserving visual quality.

## Repository structure

```text
LCR-IH/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config_finetune_dvi2k.py
├── config_finetune.py
├── config_finetune_coco.py
├── datasets.py
├── train_StegFormer_single_image_finetune.py
├── test_save_single_image_hiding_finetune_coco.py
├── src/
│   ├── __init__.py
│   └── ...
├── docs/
│   └── installation.md
├── examples/
│   └── README.md
└── notebooks/
    └── demo.ipynb
```

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure dataset paths and checkpoints in the config files.

3. Train the model:

```bash
python train_StegFormer_single_image_finetune.py --config_module config_finetune_dvi2k
```

4. Run evaluation:

```bash
python test_save_single_image_hiding_finetune_coco.py --config config_finetune_coco
```

## Notes

- This repo is ready to receive the full research codebase for model implementations and evaluation utilities.
- The code files included here are the main training/evaluation scripts you provided and are intended to be adapted to your local project structure.
- External files such as `stegformer_crop_model1.py` and `critic.py` must be present in the same runtime environment if they are not included in this repository yet.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this project in your research, please cite the relevant paper.
