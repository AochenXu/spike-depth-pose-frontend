import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class RateEncoder:
    def __init__(self, time_steps: int = 8, max_rate: float = 1.0):
        self.time_steps = int(time_steps)
        self.max_rate = float(max_rate)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        img = torch.clamp(img.float(), 0.0, 1.0) * self.max_rate
        random_tensor = torch.rand(
            self.time_steps,
            *img.shape,
            dtype=img.dtype,
        )
        return (random_tensor < img.unsqueeze(0)).float()


class LatencyEncoder:
    def __init__(self, time_steps: int = 8):
        self.time_steps = int(time_steps)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        img = torch.clamp(img.float(), 0.0, 1.0)
        if self.time_steps <= 1:
            return (img > 0.5).float().unsqueeze(0)
        spike_time = ((1.0 - img) * float(self.time_steps - 1)).round().long()
        spike_time = torch.clamp(spike_time, min=0, max=self.time_steps - 1)
        valid = img > 1e-4
        return torch.stack([((spike_time == t) & valid).float() for t in range(self.time_steps)], dim=0)


class DeltaLatencyEncoder:
    def __init__(self, time_steps: int = 8, delta_threshold: float = 0.03):
        self.time_steps = int(time_steps)
        self.delta_threshold = float(delta_threshold)

    def encode(self, img_prev: torch.Tensor, img_t: torch.Tensor) -> torch.Tensor:
        img_prev = torch.clamp(img_prev.float(), 0.0, 1.0)
        img_t = torch.clamp(img_t.float(), 0.0, 1.0)
        diff = torch.abs(img_t - img_prev)
        if self.time_steps <= 1:
            return (img_t * (diff > self.delta_threshold).float()).unsqueeze(0)
        diff_norm = diff / (diff.amax() + 1e-6)
        spike_time = ((1.0 - diff_norm) * float(self.time_steps - 1)).round().long()
        spike_time = torch.clamp(spike_time, min=0, max=self.time_steps - 1)
        valid = diff > self.delta_threshold
        return torch.stack([img_t * ((spike_time == t) & valid).float() for t in range(self.time_steps)], dim=0)


def load_intrinsics(calib_path: str) -> np.ndarray:
    with open(calib_path, "r") as f:
        lines = f.readlines()
    p2 = lines[2].strip().split()[1:]
    p2 = np.array(p2, dtype=np.float32).reshape(3, 4)
    return p2[:, :3]


def scale_intrinsics(K, orig_size=(1242, 375), new_size=None):
    if new_size is None:
        raise ValueError("new_size must be provided")
    ow, oh = orig_size
    nw, nh = new_size
    sx = float(nw) / float(ow)
    sy = float(nh) / float(oh)
    if isinstance(K, np.ndarray):
        K2 = K.astype(np.float32).copy()
    else:
        K2 = K.clone().float()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


class KITTIOdometryTriplet(Dataset):
    def __init__(
        self,
        kitti_root: str,
        seqs: Sequence[str],
        resize=(256, 832),
        time_steps: int = 8,
        spike_input: bool = False,
        spike_encoding: str = "rate",
    ):
        self.kitti_root = kitti_root
        self.seqs = list(seqs)
        self.resize = resize
        self.spike_input = spike_input
        self.spike_encoding = str(spike_encoding)
        self.rate_encoder = RateEncoder(time_steps=time_steps)
        self.latency_encoder = LatencyEncoder(time_steps=time_steps)
        self.delta_latency_encoder = DeltaLatencyEncoder(time_steps=time_steps)
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Dict[str, str]]:
        samples = []
        for seq_id in self.seqs:
            seq_dir = os.path.join(self.kitti_root, "sequences", seq_id)
            img_dir = os.path.join(seq_dir, "image_2")
            calib_path = os.path.join(seq_dir, "calib.txt")
            if not os.path.isdir(img_dir) or not os.path.exists(calib_path):
                continue
            img_names = sorted([name for name in os.listdir(img_dir) if name.endswith(".png")])
            for idx in range(1, len(img_names) - 1):
                samples.append(
                    {
                        "seq_id": seq_id,
                        "img_prev": os.path.join(img_dir, img_names[idx - 1]),
                        "img_t": os.path.join(img_dir, img_names[idx]),
                        "img_next": os.path.join(img_dir, img_names[idx + 1]),
                        "calib_path": calib_path,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        if self.resize is not None:
            img = img.resize(self.resize[::-1], Image.BILINEAR)
        img = np.asarray(img).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1), (orig_w, orig_h)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_prev, orig_size = self._load_image(sample["img_prev"])
        img_t, _ = self._load_image(sample["img_t"])
        img_next, _ = self._load_image(sample["img_next"])
        intrinsics = load_intrinsics(sample["calib_path"])
        intrinsics = scale_intrinsics(intrinsics, orig_size=orig_size, new_size=(self.resize[1], self.resize[0]))
        intrinsics = torch.from_numpy(intrinsics).float()

        out = {
            "img_prev": img_prev,
            "img_t": img_t,
            "img_next": img_next,
            "intrinsics": intrinsics,
            "seq_id": sample["seq_id"],
        }
        if self.spike_input:
            if self.spike_encoding == "rate":
                out["spike_t"] = self.rate_encoder.encode(img_t)
            elif self.spike_encoding == "latency":
                out["spike_t"] = self.latency_encoder.encode(img_t)
            elif self.spike_encoding == "delta_latency":
                out["spike_t"] = self.delta_latency_encoder.encode(img_prev, img_t)
            else:
                raise ValueError(f"Unsupported spike_encoding: {self.spike_encoding}")
        return out


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    batch = axis_angle.shape[0]
    theta = torch.norm(axis_angle, dim=1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / theta
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    K = torch.zeros(batch, 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[:, 0, 1] = -kz
    K[:, 0, 2] = ky
    K[:, 1, 0] = kz
    K[:, 1, 2] = -kx
    K[:, 2, 0] = -ky
    K[:, 2, 1] = kx
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).repeat(batch, 1, 1)
    theta = theta.view(batch, 1, 1)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def pose_vec_to_matrix(pose_vec: torch.Tensor) -> torch.Tensor:
    """Convert [B,6] pose vectors to homogeneous transform matrices [B,4,4]."""
    batch = pose_vec.shape[0]
    transform = torch.eye(4, device=pose_vec.device, dtype=pose_vec.dtype).unsqueeze(0).repeat(batch, 1, 1)
    transform[:, :3, :3] = axis_angle_to_rotation_matrix(pose_vec[:, 3:])
    transform[:, :3, 3] = pose_vec[:, :3]
    return transform


def invert_pose_matrix(transform: torch.Tensor) -> torch.Tensor:
    """Invert homogeneous transforms [B,4,4]."""
    rot = transform[:, :3, :3]
    trans = transform[:, :3, 3:]
    rot_inv = rot.transpose(1, 2)
    trans_inv = -(rot_inv @ trans)
    out = torch.eye(4, device=transform.device, dtype=transform.dtype).unsqueeze(0).repeat(transform.shape[0], 1, 1)
    out[:, :3, :3] = rot_inv
    out[:, :3, 3:] = trans_inv
    return out


def pose_consistency_loss(forward_pose: torch.Tensor, backward_pose: torch.Tensor) -> torch.Tensor:
    """Penalize forward/backward pose pairs that do not compose to identity."""
    forward_tf = pose_vec_to_matrix(forward_pose)
    backward_tf = pose_vec_to_matrix(backward_pose)
    composed = forward_tf @ backward_tf
    identity = torch.eye(4, device=composed.device, dtype=composed.dtype).unsqueeze(0)
    return torch.abs(composed - identity).mean()


def backproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = depth.shape
    device = depth.device
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).repeat(batch, 1, 1)

    fx = intrinsics[:, 0, 0].view(batch, 1, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1)

    u = torch.arange(0, width, device=device).view(1, 1, width).expand(batch, height, width)
    v = torch.arange(0, height, device=device).view(1, height, 1).expand(batch, height, width)
    z = depth.squeeze(1)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack([x, y, z], dim=1).view(batch, 3, -1)


def warp_image(img_src: torch.Tensor, depth_ref: torch.Tensor, pose: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = img_src.shape
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).repeat(batch, 1, 1)

    points = backproject_depth(depth_ref, intrinsics)
    rotation = axis_angle_to_rotation_matrix(pose[:, 3:])
    translation = pose[:, :3].view(batch, 3, 1)
    transformed = rotation @ points + translation

    x = transformed[:, 0, :]
    y = transformed[:, 1, :]
    z = transformed[:, 2, :].clamp_min(1e-7)

    fx = intrinsics[:, 0, 0].view(batch, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1)

    u = (fx * (x / z) + cx).view(batch, height, width)
    v = (fy * (y / z) + cy).view(batch, height, width)

    x_norm = 2.0 * (u / max(1, width - 1)) - 1.0
    y_norm = 2.0 * (v / max(1, height - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(img_src, grid, padding_mode="border", align_corners=True)


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return torch.clamp((1 - ssim_n / (ssim_d + 1e-7)) / 2, 0, 1)


def compute_photometric_terms(target: torch.Tensor, warped_sources: Sequence[torch.Tensor], identity_sources: Sequence[torch.Tensor]):
    photo_errors = [0.15 * torch.abs(target - pred).mean(1, keepdim=True) + 0.85 * ssim(target, pred) for pred in warped_sources]
    photo_stack = torch.stack(photo_errors, dim=0)
    photo_min, _ = photo_stack.min(dim=0)

    id_errors = [0.15 * torch.abs(target - src).mean(1, keepdim=True) + 0.85 * ssim(target, src) for src in identity_sources]
    id_stack = torch.stack(id_errors, dim=0)
    identity_min, _ = id_stack.min(dim=0)
    mask = (photo_min + 1e-5 < identity_min).float()
    loss = (photo_min * mask).sum() / (mask.sum() + 1e-6)
    return loss, {
        "photo_map": photo_min,
        "auto_mask_ratio": float(mask.mean().item()),
    }


def depth_smoothness_loss(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    depth = depth / (depth.mean(dim=(2, 3), keepdim=True) + 1e-6)
    grad_depth_x = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    grad_depth_y = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    grad_img_x = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), 1, keepdim=True)
    return (grad_depth_x * torch.exp(-grad_img_x)).mean() + (grad_depth_y * torch.exp(-grad_img_y)).mean()


def far_depth_penalty(
    depth: torch.Tensor,
    max_depth: float,
    start_ratio: float = 0.9,
) -> torch.Tensor:
    """Discourage depth maps from collapsing into the far-depth ceiling.

    The penalty stays inactive for the bulk of the valid depth range and only
    grows when predictions approach the configured maximum depth.
    """

    max_depth = float(max_depth)
    start_ratio = float(start_ratio)
    if max_depth <= 0.0:
        raise ValueError("max_depth must be positive")
    start_ratio = min(max(start_ratio, 0.0), 0.999)
    start_depth = max_depth * start_ratio
    denom = max(max_depth - start_depth, 1e-6)
    excess = torch.relu((depth - start_depth) / denom)
    return (excess ** 2).mean()
