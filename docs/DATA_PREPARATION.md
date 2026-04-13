# Data Preparation

This repository uses two data layouts.

## 1. Depth Branch Data

The depth branch expects two plain-text files:

- one file containing RGB image paths
- one file containing matching depth map paths

Each line should correspond to one sample, and the two files must be aligned line by line.

Example:

```text
/path/to/image_000001.png
/path/to/image_000002.png
...
```

```text
/path/to/depth_000001.png
/path/to/depth_000002.png
...
```

You can adapt [make_kitti_selection_lists.py](/home/larl/snn/snn_depth_geometry_release/depth_branch/make_kitti_selection_lists.py) to generate such lists from KITTI depth data.

## 2. Geometry Branch Data

The geometry branch expects the KITTI odometry dataset root in the following style:

```text
<kitti_root>/
└── sequences/
    ├── 00/
    ├── 01/
    ├── ...
    └── 10/
```

The code reads triplets `(t-1, t, t+1)` directly from KITTI odometry sequences.

Use [make_kitti_sfm_triplets.py](/home/larl/snn/snn_depth_geometry_release/geometry_branch/make_kitti_sfm_triplets.py) if you want to export explicit triplet lists.

## Camera Intrinsics

Both branches rely on KITTI camera intrinsics. The geometry branch reads them from sequence calibration files and rescales them internally according to the chosen image size.

## What Is Not Included

This repository does not ship:

- KITTI raw data
- KITTI odometry data
- KITTI depth ground truth
- pretrained checkpoints

Please prepare these resources locally before training or evaluation.
