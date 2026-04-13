# SNN Depth and Geometry Release

This repository contains the main code for our SNN-based monocular depth and geometry perception framework.

The release is organized into two parts:

- `depth_branch/`: supervised depth estimation code used for the depth branch.
- `geometry_branch/`: self-supervised SfM / VO front-end code used for the geometry branch.

This repository is intentionally cleaned for public release:

- training and evaluation scripts are kept;
- core model definitions are kept;
- dataset generation helpers are kept;
- intermediate experiment outputs, reviewer-specific scripts, and local analysis files are excluded.

## Repository Layout

```text
.
├── depth_branch/
│   ├── common.py
│   ├── make_kitti_selection_lists.py
│   ├── models.py
│   ├── train_ann_depth.py
│   ├── train_snn_depth.py
│   └── README.md
├── geometry_branch/
│   ├── benchmark_snn_frontends.py
│   ├── common.py
│   ├── compare_frontend_vo.py
│   ├── eval_snn_geometry_backend.py
│   ├── eval_snn_vo_ate.py
│   ├── make_kitti_sfm_triplets.py
│   ├── models.py
│   ├── run_lif_spike_mainline.py
│   ├── sfm_common.py
│   ├── slam_backend.py
│   ├── train_snn_sfm_kitti.py
│   └── README.md
├── docs/
│   └── DATA_PREPARATION.md
├── .gitignore
├── LICENSE
└── requirements.txt
```

## Environment

- Python `>=3.8`
- PyTorch with CUDA is recommended for training

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare datasets

See [DATA_PREPARATION.md](/home/larl/snn/snn_depth_geometry_release/docs/DATA_PREPARATION.md).

### 2. Depth branch

See [depth_branch/README.md](/home/larl/snn/snn_depth_geometry_release/depth_branch/README.md).

### 3. Geometry branch

See [geometry_branch/README.md](/home/larl/snn/snn_depth_geometry_release/geometry_branch/README.md).

## Notes

- Pretrained checkpoints are not included in this release.
- Dataset files are not included in this release.
- Paths in the training scripts are configurable through command-line arguments.
- The geometry branch currently supports ANN-initialized and SNN fine-tuned front-end training on KITTI odometry sequences.

## Reproducibility Scope

This release focuses on the core code used by the paper:

- depth model training and SNN fine-tuning;
- geometry front-end training;
- VO / ATE evaluation;
- geometry backend evaluation;
- inference and latency benchmarking.

It does not include:

- local reviewer response scripts;
- private experiment logs;
- intermediate output folders;
- thesis drafting files.

## Citation

If you use this code in your research, please cite the corresponding paper once the final bibliographic information is available.
