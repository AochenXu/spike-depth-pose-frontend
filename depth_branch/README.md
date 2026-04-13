# Depth Branch

This directory contains the main code for the depth perception branch of the project.

The depth branch focuses on the question of how spike-based visual encoding can be used for monocular depth estimation while preserving a practical training path through ANN pretraining and SNN fine-tuning.

## Files

- `models.py`: ANN/SNN depth models and shared building blocks.
- `common.py`: dataset loading, metrics, logging, visualization, and utility functions.
- `train_ann_depth.py`: ANN depth training script.
- `train_snn_depth.py`: SNN depth fine-tuning script.
- `make_kitti_selection_lists.py`: helper to build KITTI depth file lists.

## Expected Data

The depth branch expects paired image and depth file lists.

Typical usage is based on KITTI depth data:

- RGB image list file
- depth map list file

See [DATA_PREPARATION.md](../docs/DATA_PREPARATION.md).

## Training an ANN depth model

```bash
python train_ann_depth.py \
  --image-list-file /path/to/train_images.txt \
  --depth-list-file /path/to/train_depths.txt \
  --output-dir ./outputs/ann_depth \
  --auto-experiment-dir \
  --num-epochs 10 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --height 256 \
  --width 832 \
  --amp
```

## Fine-tuning the SNN depth model

```bash
python train_snn_depth.py \
  --image-list-file /path/to/train_images.txt \
  --depth-list-file /path/to/train_depths.txt \
  --ann-encoder-ckpt /path/to/best_ann_encoder.pth \
  --output-dir ./outputs/snn_depth \
  --auto-experiment-dir \
  --num-epochs 10 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --height 256 \
  --width 832 \
  --input-encoding analog \
  --time-steps 1 \
  --init-from-ann \
  --amp
```

## Main Outputs

Training produces:

- model checkpoints
- CSV history files
- JSON metric summaries
- optional visualizations and TensorBoard logs

## Remarks

- The SNN depth branch is designed to start from ANN encoder initialization.
- The released pipeline reflects the practical training path used in our study: first obtain a stable ANN depth model, then transfer it to the SNN branch.
- This branch is meant to expose the main code path used for depth learning rather than every intermediate experimental variant explored during development.
