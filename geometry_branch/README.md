# Geometry Branch

This directory contains the main code for the geometry branch, namely the SNN-based monocular SfM / VO front-end.

## Files

- `models.py`: geometry front-end model definitions.
- `common.py`: logging and shared utilities.
- `sfm_common.py`: KITTI odometry data handling, photometric losses, warping, and geometry helpers.
- `slam_backend.py`: geometry backend utilities for depth reprojection and fusion.
- `train_snn_sfm_kitti.py`: main geometry front-end training script.
- `eval_snn_vo_ate.py`: VO / ATE evaluation.
- `eval_snn_geometry_backend.py`: backend geometry evaluation.
- `benchmark_snn_frontends.py`: latency and inference benchmark.
- `compare_frontend_vo.py`: comparison helper for multiple front-ends.
- `run_lif_spike_mainline.py`: convenience launcher for the main geometry configuration.
- `make_kitti_sfm_triplets.py`: helper to generate triplet lists.

## Expected Data

The geometry branch is designed for KITTI odometry sequences.

See [DATA_PREPARATION.md](/home/larl/snn/snn_depth_geometry_release/docs/DATA_PREPARATION.md).

## Main Training Command

```bash
python train_snn_sfm_kitti.py \
  --kitti-root /path/to/kitti_dataset/dataset \
  --ann-encoder-ckpt /path/to/best_ann_encoder.pth \
  --snn-depth-ckpt /path/to/best_snn_depth_model.pth \
  --output-dir ./outputs/snn_sfm \
  --auto-experiment-dir \
  --train-seqs 03,04,06 \
  --val-seqs 09 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --num-epochs 30 \
  --height 256 \
  --width 832 \
  --time-steps 4 \
  --input-encoding delta_latency \
  --lif-output-mode spike \
  --sparse-exec \
  --sparse-layers conv1,conv2 \
  --pose-input-normalization \
  --freeze-depth-epochs 2 \
  --amp
```

## Evaluate VO / ATE

```bash
python eval_snn_vo_ate.py \
  --kitti-root /path/to/kitti_dataset/dataset \
  --seq-id 09 \
  --ckpt /path/to/best_snn_sfm.pth
```

## Evaluate Geometry Backend

```bash
python eval_snn_geometry_backend.py \
  --kitti-root /path/to/kitti_dataset/dataset \
  --seq-id 09 \
  --ckpt /path/to/best_snn_sfm.pth
```

## Run Mainline Convenience Script

```bash
python run_lif_spike_mainline.py \
  --train-seqs 03,04,06 \
  --val-seqs 09
```

## Remarks

- The geometry branch currently targets front-end geometric perception rather than a complete SLAM system.
- Sparse execution, temporal encoding, and pose-consistency regularization are all exposed through command-line options.
- For public release, only the main branch code has been kept here; reviewer-specific scripts and local experiment wrappers are intentionally excluded.
