import csv
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_list_file(list_file: str) -> List[str]:
    with open(list_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def write_json(path: str, data: Dict) -> None:
    def _to_jsonable(obj):
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        return obj

    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(_to_jsonable(data), f, indent=2, sort_keys=True)


def write_csv_row(path: str, row: Dict[str, float]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def create_summary_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return NullSummaryWriter()


def parse_depth_selection_id(path: str) -> str:
    stem = Path(path).stem
    parts = stem.split("_")
    if "drive" in parts:
        drive_idx = parts.index("drive")
        if drive_idx + 2 < len(parts):
            return "_".join(parts[: drive_idx + 3])
    return stem


def paired_grouped_split(
    image_paths: Sequence[str],
    depth_paths: Sequence[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    assert len(image_paths) == len(depth_paths), "image/depth list length mismatch"
    samples = list(zip(image_paths, depth_paths))
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for img_path, depth_path in samples:
        key = parse_depth_selection_id(img_path)
        groups.setdefault(key, []).append((img_path, depth_path))

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_groups = len(keys)
    n_train = max(1, int(n_groups * train_ratio))
    n_val = max(1, int(n_groups * val_ratio)) if n_groups >= 3 else max(0, n_groups - n_train)
    n_train = min(n_train, n_groups - n_val)
    n_test = max(0, n_groups - n_train - n_val)

    train_keys = keys[:n_train]
    val_keys = keys[n_train : n_train + n_val]
    test_keys = keys[n_train + n_val : n_train + n_val + n_test]

    if not test_keys and val_keys:
        test_keys = val_keys[-1:]
        val_keys = val_keys[:-1]

    def flatten(group_keys: Sequence[str]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for key in group_keys:
            out.extend(groups[key])
        return out

    return {
        "train": flatten(train_keys),
        "val": flatten(val_keys),
        "test": flatten(test_keys),
    }


class KITTIDepthDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, str]], resize: Optional[Tuple[int, int]] = (256, 832)):
        self.samples = list(samples)
        self.resize = resize

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.resize is not None:
            img = img.resize(self.resize[::-1], Image.BILINEAR)
        img = np.asarray(img).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def _load_depth(self, path: str) -> torch.Tensor:
        depth = Image.open(path)
        if self.resize is not None:
            depth = depth.resize(self.resize[::-1], Image.NEAREST)
        depth = np.asarray(depth).astype(np.float32) / 256.0
        depth[depth <= 0] = 0.0
        return torch.from_numpy(depth).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, depth_path = self.samples[idx]
        return {
            "image": self._load_image(image_path),
            "depth": self._load_depth(depth_path),
            "image_path": image_path,
            "depth_path": depth_path,
        }


def _adjacent_image_candidates(image_path: str, offset: int) -> List[str]:
    path = Path(image_path)
    stem = path.stem
    candidates: List[str] = []
    if stem.isdigit():
        width = len(stem)
        adj = int(stem) + offset
        if adj >= 0:
            candidates.append(str(path.with_name(f"{adj:0{width}d}{path.suffix}")))
    match = re.match(r"^(.*_image_)(\d+)(_image_\d+)$", stem)
    if match:
        prefix, frame_id, suffix = match.groups()
        width = len(frame_id)
        adj = int(frame_id) + offset
        if adj >= 0:
            candidates.append(str(path.with_name(f"{prefix}{adj:0{width}d}{suffix}{path.suffix}")))
    return candidates


def resolve_neighbor_image_path(image_path: str, offset: int = -1) -> str:
    candidates = _adjacent_image_candidates(image_path, offset)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    fallback_candidates = _adjacent_image_candidates(image_path, -offset)
    for candidate in fallback_candidates:
        if os.path.exists(candidate):
            return candidate
    return image_path


class KITTITemporalDepthDataset(KITTIDepthDataset):
    """Depth dataset with a neighboring RGB frame for temporal spike encoding."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, depth_path = self.samples[idx]
        image_prev_path = resolve_neighbor_image_path(image_path, offset=-1)
        image_next_path = resolve_neighbor_image_path(image_path, offset=1)
        return {
            "image_prev": self._load_image(image_prev_path),
            "image": self._load_image(image_path),
            "image_next": self._load_image(image_next_path),
            "depth": self._load_depth(depth_path),
            "image_path": image_path,
            "image_prev_path": image_prev_path,
            "image_next_path": image_next_path,
            "depth_path": depth_path,
        }


def depth_l1_loss(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        mask = (gt > 0).float()
    else:
        mask = mask.float() * (gt > 0).float()
    valid = mask.sum()
    if valid < 1:
        return pred.new_tensor(0.0)
    return (torch.abs(pred - gt) * mask).sum() / valid


@dataclass
class DepthMetrics:
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta1: float
    delta2: float
    delta3: float
    valid_pixels: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> DepthMetrics:
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()
    mask = (gt > min_depth) & (gt < max_depth)
    valid = int(mask.sum().item())
    if valid == 0:
        return DepthMetrics(*(0.0 for _ in range(7)), valid_pixels=0)

    pred = torch.clamp(pred[mask], min=min_depth, max=max_depth)
    gt = torch.clamp(gt[mask], min=min_depth, max=max_depth)

    thresh = torch.max(gt / pred, pred / gt)
    delta1 = float((thresh < 1.25).float().mean().item())
    delta2 = float((thresh < 1.25 ** 2).float().mean().item())
    delta3 = float((thresh < 1.25 ** 3).float().mean().item())

    diff = pred - gt
    abs_rel = float((torch.abs(diff) / gt).mean().item())
    sq_rel = float(((diff ** 2) / gt).mean().item())
    rmse = float(torch.sqrt((diff ** 2).mean()).item())
    rmse_log = float(torch.sqrt(((torch.log(pred) - torch.log(gt)) ** 2).mean()).item())
    return DepthMetrics(abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3, valid_pixels=valid)


def aggregate_depth_metrics(metric_list: Iterable[DepthMetrics]) -> Dict[str, float]:
    metrics = list(metric_list)
    if not metrics:
        return {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3", "valid_pixels"]}
    keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
    out = {k: float(np.mean([getattr(m, k) for m in metrics])) for k in keys}
    out["valid_pixels"] = int(np.sum([m.valid_pixels for m in metrics]))
    return out


def save_depth_visualization(depth_tensor: torch.Tensor, save_path: str) -> None:
    depth = depth_tensor.detach().cpu().squeeze().numpy()
    valid = depth > 0
    if valid.sum() == 0:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
    else:
        d_min = np.percentile(depth[valid], 5)
        d_max = np.percentile(depth[valid], 95)
        depth_clip = np.clip(depth, d_min, d_max)
        depth_norm = (depth_clip - d_min) / (d_max - d_min + 1e-6)
        depth_vis = (depth_norm * 255.0).astype(np.uint8)
    ensure_dir(os.path.dirname(save_path) or ".")
    Image.fromarray(depth_vis).save(save_path)


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
    pieces = []
    for key in ordered:
        if key in metrics:
            pieces.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(pieces)


def measure_inference_ms(model: torch.nn.Module, sample: torch.Tensor, steps: int = 20) -> float:
    if sample.device.type != "cuda":
        return 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(sample)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            model(sample)
        torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) * 1000.0 / steps
