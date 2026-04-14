import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models import MonoDepthSNN_Spike
from sfm_common import axis_angle_to_rotation_matrix


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_image(path: str, resize_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = resize_hw
    img = Image.open(path).convert("RGB")
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_gt_poses(path: str) -> np.ndarray:
    poses = []
    with open(path, "r") as f:
        for line in f:
            vals = np.fromstring(line.strip(), sep=" ", dtype=np.float32)
            if vals.size != 12:
                continue
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :4] = vals.reshape(3, 4)
            poses.append(mat)
    return np.stack(poses, axis=0)


def normalize_poses(poses: np.ndarray) -> np.ndarray:
    ref_inv = np.linalg.inv(poses[0])
    return np.stack([ref_inv @ pose for pose in poses], axis=0)


def pose_vec_to_matrix(pose_vec: torch.Tensor) -> np.ndarray:
    pose_vec = pose_vec.view(1, 6)
    trans = pose_vec[:, :3]
    rot = axis_angle_to_rotation_matrix(pose_vec[:, 3:])[0]
    mat = torch.eye(4, dtype=pose_vec.dtype, device=pose_vec.device)
    mat[:3, :3] = rot
    mat[:3, 3] = trans[0]
    return mat.detach().cpu().numpy()


def chain_relative_poses(rel_poses: Sequence[np.ndarray]) -> np.ndarray:
    poses = [np.eye(4, dtype=np.float32)]
    current = np.eye(4, dtype=np.float32)
    for rel in rel_poses:
        current = current @ rel
        poses.append(current.copy())
    return np.stack(poses, axis=0)


def scale_align_translation(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    denom = float(np.sum(pred_xyz * pred_xyz))
    if denom < 1e-8:
        return pred_xyz
    scale = float(np.sum(gt_xyz * pred_xyz) / denom)
    return pred_xyz * scale


def compute_ate(pred_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
    pred_xyz = pred_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz_aligned = scale_align_translation(pred_xyz, gt_xyz)
    diff = pred_xyz_aligned - gt_xyz
    rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    mean_err = float(np.mean(np.linalg.norm(diff, axis=1)))
    return {"ate_rmse": rmse, "ate_mean": mean_err}


def build_window_starts(total_images: int, window_size: int, window_stride: int, max_windows: int) -> List[int]:
    if window_size <= 0:
        window_size = total_images
    window_size = min(window_size, total_images)
    if window_stride <= 0:
        window_stride = window_size
    starts = list(range(0, max(1, total_images - window_size + 1), window_stride))
    if not starts:
        starts = [0]
    if max_windows > 0:
        starts = starts[:max_windows]
    return starts


def save_trajectory_plot(pred_poses: np.ndarray, gt_poses: np.ndarray, out_path: str, title: str) -> None:
    pred_xyz = pred_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = scale_align_translation(pred_xyz, gt_xyz)

    plt.figure(figsize=(7, 5))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="GT", linewidth=2)
    plt.plot(pred_xyz[:, 0], pred_xyz[:, 2], label="Pred", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def resolve_ckpt_path(args) -> str:
    if args.experiment_dir:
        return os.path.join(args.experiment_dir, "best_snn_sfm.pth")
    return args.ckpt_path


def build_model(config: Dict[str, object], args) -> MonoDepthSNN_Spike:
    return MonoDepthSNN_Spike(
        tau=config.get("tau", args.tau),
        time_steps=config.get("time_steps", args.time_steps),
        v_threshold=config.get("v_threshold", args.v_threshold),
        input_encoding=config.get("input_encoding", args.input_encoding),
        lif_output_mode=config.get("lif_output_mode", "mixed"),
        learnable_threshold=config.get("learnable_threshold", False),
        sparse_exec=config.get("sparse_exec", False),
        sparse_layers=config.get("sparse_layers", ["conv1", "conv2"]),
        sparse_activity_threshold=config.get("sparse_activity_threshold", args.sparse_activity_threshold),
        sparse_fallback_ratio=config.get("sparse_fallback_ratio", args.sparse_fallback_ratio),
        delta_anchor_weight=config.get("delta_anchor_weight", 0.2),
        decoder_channel_scale=config.get("decoder_channel_scale", 1.0),
        min_depth=config.get("min_depth", 0.5),
        max_depth=config.get("max_depth", 80.0),
        pose_hidden_channels=config.get("pose_hidden_channels", 256),
        pose_mlp_hidden=config.get("pose_mlp_hidden", 128),
        pose_input_normalization=config.get("pose_input_normalization", False),
        hybrid_static_branch=config.get("hybrid_static_branch", False),
        hybrid_static_weight=config.get("hybrid_static_weight", 0.5),
        hybrid_pose_diff=config.get("hybrid_pose_diff", False),
    )


def evaluate_window(
    model: MonoDepthSNN_Spike,
    config: Dict[str, object],
    image_paths: Sequence[Path],
    device: torch.device,
    resize_hw: Tuple[int, int],
) -> Dict[str, object]:
    rel_poses = []
    avg_spikes = []
    avg_active = []
    avg_used_sparse = []

    with torch.no_grad():
        for idx in range(len(image_paths) - 1):
            cur_path = str(image_paths[idx])
            next_path = str(image_paths[idx + 1])
            prev_path = str(image_paths[idx - 1]) if idx > 0 else cur_path
            img_prev = load_image(prev_path, resize_hw).unsqueeze(0).to(device)
            img_t = load_image(cur_path, resize_hw).unsqueeze(0).to(device)
            img_next = load_image(next_path, resize_hw).unsqueeze(0).to(device)

            kwargs = {
                "num_steps": config.get("time_steps", 1),
                "input_encoding": config.get("input_encoding", "analog"),
            }
            if config.get("input_encoding") in {"delta_latency", "delta_latency_anchor"}:
                kwargs["img_prev"] = img_prev
            model(img_t, **kwargs)

            avg_spikes.append(float(model.get_spike_stats().get("avg_spike_rate", 0.0)))
            if hasattr(model, "get_sparse_stats"):
                sparse_stats = model.get_sparse_stats()
                avg_active.append(float(sparse_stats.get("avg_active_ratio", 1.0)))
                avg_used_sparse.append(float(sparse_stats.get("avg_used_sparse", 0.0)))

            pose_next = model.predict_pose_pair(img_t, img_next)
            rel_poses.append(pose_vec_to_matrix(pose_next[0]))

    pred_poses = chain_relative_poses(rel_poses)
    return {
        "pred_poses": pred_poses,
        "avg_spike_rate": float(np.mean(avg_spikes)) if avg_spikes else 0.0,
        "avg_active_ratio": float(np.mean(avg_active)) if avg_active else 1.0,
        "avg_used_sparse": float(np.mean(avg_used_sparse)) if avg_used_sparse else 0.0,
    }


def evaluate_full_trajectory(
    model: MonoDepthSNN_Spike,
    config: Dict[str, object],
    image_paths: Sequence[Path],
    gt_poses: np.ndarray,
    device: torch.device,
    resize_hw: Tuple[int, int],
) -> Dict[str, object]:
    window_eval = evaluate_window(model, config, image_paths, device, resize_hw)
    pred_poses = window_eval["pred_poses"]
    gt_eval = gt_poses[: pred_poses.shape[0]]
    metrics = compute_ate(pred_poses, gt_eval)
    metrics.update(
        {
            "num_frames": int(pred_poses.shape[0]),
            "num_windows": 1,
            "avg_spike_rate": window_eval["avg_spike_rate"],
            "avg_active_ratio": window_eval["avg_active_ratio"],
            "avg_used_sparse": window_eval["avg_used_sparse"],
            "protocol": "full_trajectory",
        }
    )
    return {
        "metrics": metrics,
        "pred_poses": pred_poses,
        "gt_poses": gt_eval,
        "window_rows": [],
    }


def evaluate_windowed(
    model: MonoDepthSNN_Spike,
    config: Dict[str, object],
    image_paths: Sequence[Path],
    gt_poses: np.ndarray,
    device: torch.device,
    resize_hw: Tuple[int, int],
    window_size: int,
    window_stride: int,
    max_windows: int,
) -> Dict[str, object]:
    starts = build_window_starts(len(image_paths), window_size, window_stride, max_windows)
    ate_rmse_all = []
    ate_mean_all = []
    spike_all = []
    active_all = []
    used_sparse_all = []
    window_rows = []
    first_pred = None
    first_gt = None

    for start in starts:
        indices = list(range(start, min(len(image_paths), start + window_size)))
        if len(indices) < 3:
            continue
        window_paths = [image_paths[i] for i in indices]
        gt_eval = normalize_poses(gt_poses[indices])
        window_eval = evaluate_window(model, config, window_paths, device, resize_hw)
        pred_poses = window_eval["pred_poses"]
        gt_match = gt_eval[: pred_poses.shape[0]]
        ate = compute_ate(pred_poses, gt_match)
        ate_rmse_all.append(ate["ate_rmse"])
        ate_mean_all.append(ate["ate_mean"])
        spike_all.append(float(window_eval["avg_spike_rate"]))
        active_all.append(float(window_eval["avg_active_ratio"]))
        used_sparse_all.append(float(window_eval["avg_used_sparse"]))
        window_rows.append(
            {
                "start_index": int(start),
                "num_frames": int(pred_poses.shape[0]),
                "ate_rmse": float(ate["ate_rmse"]),
                "ate_mean": float(ate["ate_mean"]),
                "avg_spike_rate": float(window_eval["avg_spike_rate"]),
                "avg_active_ratio": float(window_eval["avg_active_ratio"]),
                "avg_used_sparse": float(window_eval["avg_used_sparse"]),
            }
        )
        if first_pred is None:
            first_pred = pred_poses
            first_gt = gt_match

    if not window_rows:
        raise RuntimeError("No valid windows were evaluated. Try increasing --max-frames or decreasing --window-size.")

    metrics = {
        "ate_rmse": float(np.mean(ate_rmse_all)),
        "ate_mean": float(np.mean(ate_mean_all)),
        "num_frames": int(np.mean([row["num_frames"] for row in window_rows])),
        "num_windows": int(len(window_rows)),
        "avg_spike_rate": float(np.mean(spike_all)),
        "avg_active_ratio": float(np.mean(active_all)),
        "avg_used_sparse": float(np.mean(used_sparse_all)),
        "protocol": "windowed",
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "max_windows": int(max_windows),
        "max_frames": int(len(image_paths)),
    }
    return {
        "metrics": metrics,
        "pred_poses": first_pred,
        "gt_poses": first_gt,
        "window_rows": window_rows,
    }


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = resolve_ckpt_path(args)
    if not ckpt_path:
        raise ValueError("Please provide --ckpt-path or --experiment-dir.")
    if not args.kitti_root:
        raise ValueError("Please provide --kitti-root.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    model = build_model(config, args)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    seq_dir = Path(args.kitti_root) / "sequences" / args.seq_id
    img_dir = seq_dir / "image_2"
    pose_path = Path(args.kitti_root) / "poses" / f"{args.seq_id}.txt"
    image_paths = sorted(img_dir.glob("*.png"))
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    gt_all = load_gt_poses(str(pose_path))[: len(image_paths)]

    resize_hw = (args.height, args.width)
    if args.protocol == "full_trajectory":
        gt_norm = normalize_poses(gt_all)
        result = evaluate_full_trajectory(model, config, image_paths, gt_norm, device, resize_hw)
        plot_title = f"VO Trajectory (XZ) seq{args.seq_id} full trajectory"
    else:
        result = evaluate_windowed(
            model,
            config,
            image_paths,
            gt_all,
            device,
            resize_hw,
            args.window_size,
            args.window_stride,
            args.max_windows,
        )
        plot_title = f"VO Trajectory (XZ) seq{args.seq_id} first window"

    metrics = dict(result["metrics"])
    metrics["checkpoint"] = ckpt_path
    metrics["seq_id"] = args.seq_id

    ensure_dir(args.output_dir)
    with open(os.path.join(args.output_dir, f"vo_ate_seq{args.seq_id}.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, f"vo_ate_seq{args.seq_id}_windows.json"), "w") as f:
        json.dump({"seq_id": args.seq_id, "protocol": metrics["protocol"], "windows": result["window_rows"]}, f, indent=2)
    np.save(os.path.join(args.output_dir, f"pred_traj_seq{args.seq_id}.npy"), result["pred_poses"])
    np.save(os.path.join(args.output_dir, f"gt_traj_seq{args.seq_id}.npy"), result["gt_poses"])
    save_trajectory_plot(
        result["pred_poses"],
        result["gt_poses"],
        os.path.join(args.output_dir, f"traj_seq{args.seq_id}.png"),
        plot_title,
    )

    print(
        f"[vo] protocol={metrics['protocol']} seq={args.seq_id} "
        f"windows={metrics['num_windows']} frames={metrics['num_frames']} "
        f"ate_rmse={metrics['ate_rmse']:.4f} ate_mean={metrics['ate_mean']:.4f} "
        f"avg_spike={metrics['avg_spike_rate']:.4f}"
    )


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate SNN SfM front-end with full-trajectory or windowed VO/ATE on KITTI poses.")
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--seq-id", default="09")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--output-dir", default=str(script_dir / "outputs" / "vo_eval"))
    parser.add_argument("--protocol", choices=["windowed", "full_trajectory"], default="windowed")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-stride", type=int, default=50)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=1)
    parser.add_argument("--v-threshold", type=float, default=0.35)
    parser.add_argument("--input-encoding", choices=["rate", "analog", "latency", "delta_latency", "delta_latency_anchor"], default="analog")
    parser.add_argument("--sparse-activity-threshold", type=float, default=0.4)
    parser.add_argument("--sparse-fallback-ratio", type=float, default=0.6)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
