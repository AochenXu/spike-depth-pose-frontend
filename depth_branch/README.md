# Depth Branch

This directory contains the main code for the depth branch.

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

See [DATA_PREPARATION.md](/home/larl/snn/snn_depth_geometry_release/docs/DATA_PREPARATION.md).

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
- The current public release keeps the main training path used in our project, while leaving room for future cleanup and modularization.
