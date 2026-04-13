import os
from typing import List, Optional, Tuple

import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def backproject_depth_numpy(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = np.arange(width, dtype=np.float32)[None, :].repeat(height, axis=0)
    v = np.arange(height, dtype=np.float32)[:, None].repeat(width, axis=1)
    z = depth.astype(np.float32)
    x = (u - cx) * z / max(fx, 1e-7)
    y = (v - cy) * z / max(fy, 1e-7)
    return np.stack([x, y, z], axis=-1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return points @ rot.T + trans[None, None, :]


def rasterize_depth(points_tgt: np.ndarray, intrinsics: np.ndarray, out_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = out_shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    z = points_tgt[..., 2]
    valid = z > 1e-6
    x = points_tgt[..., 0]
    y = points_tgt[..., 1]
    u = np.round(fx * (x / np.maximum(z, 1e-7)) + cx).astype(np.int32)
    v = np.round(fy * (y / np.maximum(z, 1e-7)) + cy).astype(np.int32)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    depth_out = np.full((height, width), np.inf, dtype=np.float32)
    count_out = np.zeros((height, width), dtype=np.int32)
    if not np.any(valid):
        depth_out[:] = 0.0
        return depth_out, count_out

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]
    flat_idx = v_valid * width + u_valid
    np.minimum.at(depth_out.reshape(-1), flat_idx, z_valid)
    np.add.at(count_out.reshape(-1), flat_idx, 1)
    depth_out[np.isinf(depth_out)] = 0.0
    return depth_out, count_out


def reproject_depth_map(depth_src: np.ndarray, pose_src_to_tgt: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_points = backproject_depth_numpy(depth_src, intrinsics)
    tgt_points = transform_points(src_points, pose_src_to_tgt)
    return rasterize_depth(tgt_points, intrinsics, depth_src.shape)


def fuse_depth_maps(depth_maps: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not depth_maps:
        raise ValueError("depth_maps must not be empty")
    stack = np.stack(depth_maps, axis=0).astype(np.float32)
    valid = stack > 0
    count = valid.sum(axis=0).astype(np.int32)
    stack[~valid] = np.nan
    fused = np.nanmedian(stack, axis=0)
    fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    return fused.astype(np.float32), count


def weighted_fuse_depth_maps(
    depth_maps: List[np.ndarray],
    weights: List[float],
    confidence_maps: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not depth_maps:
        raise ValueError("depth_maps must not be empty")
    if len(depth_maps) != len(weights):
        raise ValueError("depth_maps and weights must have the same length")
    if confidence_maps is not None and len(confidence_maps) != len(depth_maps):
        raise ValueError("confidence_maps and depth_maps must have the same length")

    stack = np.stack(depth_maps, axis=0).astype(np.float32)
    valid = stack > 0
    count = valid.sum(axis=0).astype(np.int32)

    weight_stack = np.asarray(weights, dtype=np.float32)[:, None, None] * valid.astype(np.float32)
    if confidence_maps is not None:
        conf = np.stack(confidence_maps, axis=0).astype(np.float32)
        conf = np.maximum(conf, 0.0)
        weight_stack *= conf

    weighted_sum = np.sum(stack * weight_stack, axis=0)
    weight_sum = np.sum(weight_stack, axis=0)
    fused = np.divide(weighted_sum, np.maximum(weight_sum, 1e-7), out=np.zeros_like(weighted_sum), where=weight_sum > 0)
    fused[weight_sum <= 0] = 0.0
    return fused.astype(np.float32), count


def compute_overlap_mae(depth_a: np.ndarray, depth_b: np.ndarray) -> Tuple[float, int]:
    valid = (depth_a > 0) & (depth_b > 0)
    overlap = int(valid.sum())
    if overlap == 0:
        return 0.0, 0
    mae = float(np.abs(depth_a[valid] - depth_b[valid]).mean())
    return mae, overlap


def depth_to_pointcloud(depth: np.ndarray, image: np.ndarray, intrinsics: np.ndarray, stride: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    points = backproject_depth_numpy(depth, intrinsics)
    valid = depth > 0
    if stride > 1:
        valid[::stride, ::stride] &= True
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask
    xyz = points[valid]
    rgb = image[valid]
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def save_pointcloud_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz, rgb):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
