import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models import MonoDepthSNN_Spike
from sfm_common import axis_angle_to_rotation_matrix, load_intrinsics, scale_intrinsics


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_image(path: str, resize) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize(resize[::-1], Image.BILINEAR)
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


def chain_relative_poses(rel_poses: List[np.ndarray]) -> np.ndarray:
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
    return {
        "ate_rmse": rmse,
        "ate_mean": mean_err,
    }


def save_trajectory_plot(pred_poses: np.ndarray, gt_poses: np.ndarray, out_path: str) -> None:
    pred_xyz = pred_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = scale_align_translation(pred_xyz, gt_xyz)

    plt.figure(figsize=(7, 5))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="GT", linewidth=2)
    plt.plot(pred_xyz[:, 0], pred_xyz[:, 2], label="Pred", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("VO Trajectory (XZ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def resolve_ckpt_path(args) -> str:
    if args.experiment_dir:
        return os.path.join(args.experiment_dir, "best_snn_sfm.pth")
    return args.ckpt_path


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = resolve_ckpt_path(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    model = MonoDepthSNN_Spike(
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
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    seq_dir = Path(args.kitti_root) / "sequences" / args.seq_id
    img_dir = seq_dir / "image_2"
    pose_path = Path(args.kitti_root) / "poses" / f"{args.seq_id}.txt"
    calib_path = seq_dir / "calib.txt"

    image_names = sorted(p.name for p in img_dir.glob("*.png"))
    if args.max_frames > 0:
        image_names = image_names[: args.max_frames]
    gt_poses = normalize_poses(load_gt_poses(str(pose_path))[: len(image_names)])

    intrinsics = load_intrinsics(str(calib_path))
    intrinsics = scale_intrinsics(intrinsics, orig_size=(1242, 375), new_size=(args.width, args.height))

    rel_poses = []
    avg_spikes = []
    avg_active = []
    avg_used_sparse = []
    with torch.no_grad():
        for idx in range(len(image_names) - 1):
            img_t = load_image(str(img_dir / image_names[idx]), (args.height, args.width)).unsqueeze(0).to(device)
            stats_prev = None
            if config.get("input_encoding") in {"delta_latency", "delta_latency_anchor"}:
                if idx > 0:
                    stats_prev = load_image(str(img_dir / image_names[idx - 1]), (args.height, args.width)).unsqueeze(0).to(device)
                else:
                    stats_prev = img_t
            kwargs = {
                "num_steps": config.get("time_steps", args.time_steps),
                "input_encoding": config.get("input_encoding", args.input_encoding),
            }
            if stats_prev is not None:
                kwargs["img_prev"] = stats_prev
            model(img_t, **kwargs)
            avg_spikes.append(model.get_spike_stats().get("avg_spike_rate", 0.0))
            if hasattr(model, "get_sparse_stats"):
                sparse_stats = model.get_sparse_stats()
                avg_active.append(float(sparse_stats.get("avg_active_ratio", 1.0)))
                avg_used_sparse.append(float(sparse_stats.get("avg_used_sparse", 0.0)))
            if hasattr(model, "predict_pose_pair"):
                img_next = load_image(str(img_dir / image_names[idx + 1]), (args.height, args.width)).unsqueeze(0).to(device)
                pose_next = model.predict_pose_pair(img_t, img_next)
            else:
                _, pose_next = model(img_t, **kwargs)
            rel_poses.append(pose_vec_to_matrix(pose_next[0]))

    pred_poses = chain_relative_poses(rel_poses)
    gt_eval = gt_poses[: pred_poses.shape[0]]
    metrics = compute_ate(pred_poses, gt_eval)
    metrics["num_frames"] = int(pred_poses.shape[0])
    metrics["avg_spike_rate"] = float(sum(avg_spikes) / max(1, len(avg_spikes)))
    metrics["avg_active_ratio"] = float(sum(avg_active) / max(1, len(avg_active))) if avg_active else 1.0
    metrics["avg_used_sparse"] = float(sum(avg_used_sparse) / max(1, len(avg_used_sparse))) if avg_used_sparse else 0.0
    metrics["checkpoint"] = ckpt_path
    metrics["seq_id"] = args.seq_id

    ensure_dir(args.output_dir)
    with open(os.path.join(args.output_dir, f"vo_ate_seq{args.seq_id}.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    np.save(os.path.join(args.output_dir, f"pred_traj_seq{args.seq_id}.npy"), pred_poses)
    np.save(os.path.join(args.output_dir, f"gt_traj_seq{args.seq_id}.npy"), gt_eval)
    save_trajectory_plot(pred_poses, gt_eval, os.path.join(args.output_dir, f"traj_seq{args.seq_id}.png"))

    print(
        f"[vo] seq={args.seq_id} frames={metrics['num_frames']} "
        f"ate_rmse={metrics['ate_rmse']:.4f} ate_mean={metrics['ate_mean']:.4f} "
        f"avg_spike={metrics['avg_spike_rate']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SNN SfM front-end with scale-aligned VO/ATE on KITTI poses.")
    parser.add_argument("--kitti-root", default="/home/larl/kitti_dataset/dataset")
    parser.add_argument("--seq-id", default="09")
    parser.add_argument("--ckpt-path", default="/home/larl/snn/monodepth_snn_sparse_exec/outputs/snn_sfm/best_snn_sfm.pth")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--output-dir", default="/home/larl/snn/monodepth_snn_sparse_exec/outputs/vo_eval")
    parser.add_argument("--max-frames", type=int, default=0)
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
