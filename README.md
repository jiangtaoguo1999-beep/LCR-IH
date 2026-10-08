# LCR-IH

Local-Cropout-Robust Image Hiding via Redundant Auxiliary Embedding and Mask-Aware Secret Enhancement

This repository is the public release scaffold for the paper "Local-Cropout-Robust Image Hiding via Redundant Auxiliary Embedding and Mask-Aware Secret Enhancement".

The project focuses on image hiding under local cropping attacks, where part of the stego-image may be removed during transmission. The proposed framework combines redundant auxiliary embedding with a mask-aware secret enhancement module to improve recovery robustness and visual fidelity.

## Project overview

- Robust recovery from locally cropped stego-images
- Redundant auxiliary secret embedding
- Mask-aware enhancement for missing regions
- Improved structural consistency and local detail preservation

## Repository structure

```text
LCR-IH/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models/
│   ├── datasets/
│   ├── utils/
│   └── train.py
├── scripts/
│   ├── train.sh
│   └── eval.sh
├── configs/
│   └── default.yaml
├── examples/
│   └── README.md
├── docs/
│   └── installation.md
└── notebooks/
    └── demo.ipynb
```

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- numpy
- pillow
- opencv-python
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

1. Prepare the dataset.
2. Configure training or evaluation settings in `configs/default.yaml`.
3. Train the model:

```bash
python src/train.py --config configs/default.yaml
```

4. Evaluate on cropped stego-images:

```bash
python src/eval.py --config configs/default.yaml
```

## Citation

If you use this project in your research, please cite the paper:

```bibtex
@article{guo2026,
  title={Local-Cropout-Robust Image Hiding via Redundant Auxiliary Embedding and Mask-Aware Secret Enhancement},
  author={Jiangtao Guo, Buwei Tian, Xiaomeng Li, Jie Gui and Lu Dong},
  journal={},
  year={}
}
```

Please replace the BibTeX metadata with the final author list and paper details once the official publication metadata is available.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Notes

This repository is organized as a clean public release scaffold for the LCR-IH project. The implementation files and experiment scripts can be added under `src/`, `configs/`, and `scripts/` according to the research codebase. If you are releasing the full implementation, we recommend keeping the final training/evaluation pipeline, model definitions, dataset utilities, and reproducibility scripts in this structure.

## Contact

For questions or collaboration requests, please open an issue in this repository.
