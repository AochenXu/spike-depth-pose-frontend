import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_snn_frontends import benchmark_model, load_model
from eval_snn_vo_ate import (
    chain_relative_poses,
    load_gt_poses,
    load_image,
    normalize_poses,
    scale_align_translation,
)
from sfm_common import axis_angle_to_rotation_matrix
from sfm_common import KITTIOdometryTriplet


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def pose_vec_to_matrix(pose_vec: torch.Tensor) -> np.ndarray:
    pose_vec = pose_vec.view(1, 6)
    trans = pose_vec[:, :3]
    rot = axis_angle_to_rotation_matrix(pose_vec[:, 3:])[0]
    mat = torch.eye(4, dtype=pose_vec.dtype, device=pose_vec.device)
    mat[:3, :3] = rot
    mat[:3, 3] = trans[0]
    return mat.detach().cpu().numpy()


def rotation_angle_deg(rot: np.ndarray) -> float:
    trace = float(np.trace(rot))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.degrees(math.acos(cos_theta))


def compute_ate_metrics(pred_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
    pred_xyz = pred_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz_aligned = scale_align_translation(pred_xyz, gt_xyz)
    diff = pred_xyz_aligned - gt_xyz
    return {
        "ate_rmse": float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))),
        "ate_mean": float(np.mean(np.linalg.norm(diff, axis=1))),
    }


def align_pred_poses(pred_poses: np.ndarray, gt_poses: np.ndarray) -> np.ndarray:
    aligned = pred_poses.copy()
    aligned[:, :3, 3] = scale_align_translation(pred_poses[:, :3, 3], gt_poses[:, :3, 3])
    return aligned


def compute_rpe_metrics(pred_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
    pred_aligned = align_pred_poses(pred_poses, gt_poses)
    trans_errors = []
    rot_errors = []
    for idx in range(len(pred_aligned) - 1):
        pred_rel = np.linalg.inv(pred_aligned[idx]) @ pred_aligned[idx + 1]
        gt_rel = np.linalg.inv(gt_poses[idx]) @ gt_poses[idx + 1]
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        trans_errors.append(float(np.linalg.norm(rel_err[:3, 3])))
        rot_errors.append(rotation_angle_deg(rel_err[:3, :3]))
    if not trans_errors:
        return {
            "rpe_trans_rmse": 0.0,
            "rpe_trans_mean": 0.0,
            "rpe_rot_rmse_deg": 0.0,
            "rpe_rot_mean_deg": 0.0,
        }
    trans_arr = np.asarray(trans_errors, dtype=np.float64)
    rot_arr = np.asarray(rot_errors, dtype=np.float64)
    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(trans_arr ** 2))),
        "rpe_trans_mean": float(np.mean(trans_arr)),
        "rpe_rot_rmse_deg": float(np.sqrt(np.mean(rot_arr ** 2))),
        "rpe_rot_mean_deg": float(np.mean(rot_arr)),
    }


def evaluate_model_trajectory(
    model,
    config: Dict[str, Any],
    seq_id: str,
    kitti_root: str,
    height: int,
    width: int,
    max_frames: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    seq_dir = Path(kitti_root) / "sequences" / seq_id
    img_dir = seq_dir / "image_2"
    pose_path = Path(kitti_root) / "poses" / f"{seq_id}.txt"

    image_names = sorted(p.name for p in img_dir.glob("*.png"))
    if max_frames > 0:
        image_names = image_names[:max_frames]
    gt_poses = normalize_poses(load_gt_poses(str(pose_path))[: len(image_names)])

    rel_poses: List[np.ndarray] = []
    avg_spikes = []
    avg_active = []
    avg_used_sparse = []
    valid_steps = 0
    with torch.no_grad():
        for idx in range(len(image_names) - 1):
            img_t = load_image(str(img_dir / image_names[idx]), (height, width)).unsqueeze(0).to(device)
            if config.get("input_encoding") in {"delta_latency", "delta_latency_anchor"}:
                if idx > 0:
                    img_prev = load_image(str(img_dir / image_names[idx - 1]), (height, width)).unsqueeze(0).to(device)
                else:
                    img_prev = img_t
            else:
                img_prev = None

            kwargs = {
                "num_steps": config.get("time_steps", 1),
                "input_encoding": config.get("input_encoding", "analog"),
            }
            if img_prev is not None:
                kwargs["img_prev"] = img_prev

            model(img_t, **kwargs)
            if hasattr(model, "get_spike_stats"):
                avg_spikes.append(float(model.get_spike_stats().get("avg_spike_rate", 0.0)))
            if hasattr(model, "get_sparse_stats"):
                sparse_stats = model.get_sparse_stats()
                avg_active.append(float(sparse_stats.get("avg_active_ratio", 1.0)))
                avg_used_sparse.append(float(sparse_stats.get("avg_used_sparse", 0.0)))

            img_next = load_image(str(img_dir / image_names[idx + 1]), (height, width)).unsqueeze(0).to(device)
            pose_next = model.predict_pose_pair(img_t, img_next)
            rel = pose_vec_to_matrix(pose_next[0])
            if np.isfinite(rel).all():
                valid_steps += 1
            rel_poses.append(rel)

    pred_poses = chain_relative_poses(rel_poses)
    gt_eval = gt_poses[: pred_poses.shape[0]]
    metrics = compute_ate_metrics(pred_poses, gt_eval)
    metrics.update(compute_rpe_metrics(pred_poses, gt_eval))
    metrics["num_frames"] = int(pred_poses.shape[0])
    metrics["tracking_valid_ratio"] = float(valid_steps / max(1, len(rel_poses)))
    metrics["avg_spike_rate"] = float(np.mean(avg_spikes)) if avg_spikes else 0.0
    metrics["avg_active_ratio"] = float(np.mean(avg_active)) if avg_active else 1.0
    metrics["avg_used_sparse"] = float(np.mean(avg_used_sparse)) if avg_used_sparse else 0.0
    return pred_poses, gt_eval, metrics


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectories(output_path: str, results: List[Dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.8))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        gt = result["gt"][:, :3, 3]
        pred = align_pred_poses(result["pred"], result["gt"])[:, :3, 3]
        ax.plot(gt[:, 0], gt[:, 2], label="GT", linewidth=2)
        ax.plot(pred[:, 0], pred[:, 2], label=result["label"], linewidth=2)
        ax.set_title(
            f"{result['label']}\nATE {result['metrics']['ate_rmse']:.2f} m | "
            f"RPE-t {result['metrics']['rpe_trans_mean']:.2f} m"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_row(label: str, ckpt: str, config: Dict[str, Any], vo_metrics: Dict[str, float], bench_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": label,
        "checkpoint": ckpt,
        "input_encoding": config.get("input_encoding", "analog"),
        "time_steps": config.get("time_steps", 1),
        "ate_rmse": vo_metrics["ate_rmse"],
        "ate_mean": vo_metrics["ate_mean"],
        "rpe_trans_rmse": vo_metrics["rpe_trans_rmse"],
        "rpe_trans_mean": vo_metrics["rpe_trans_mean"],
        "rpe_rot_rmse_deg": vo_metrics["rpe_rot_rmse_deg"],
        "rpe_rot_mean_deg": vo_metrics["rpe_rot_mean_deg"],
        "tracking_valid_ratio": vo_metrics["tracking_valid_ratio"],
        "num_frames": vo_metrics["num_frames"],
        "depth_ms_mean": bench_metrics["depth_ms_mean"],
        "pose_ms_mean": bench_metrics["pose_ms_mean"],
        "total_ms_mean": bench_metrics["total_ms_mean"],
        "params_million": bench_metrics["params_million"],
        "avg_spike_rate": vo_metrics["avg_spike_rate"],
        "avg_active_ratio": vo_metrics["avg_active_ratio"],
        "avg_used_sparse": vo_metrics["avg_used_sparse"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ANN and SNN front-ends under the same VO pipeline.")
    parser.add_argument("--kitti-root", default="/home/larl/kitti_dataset/dataset")
    parser.add_argument("--seq-id", default="09")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--ann-model-py", default="/home/larl/snn/monodepth_snn/models.py")
    parser.add_argument("--ann-ckpt", default="/home/larl/snn/monodepth_snn/outputs/snn_sfm/snn_sfm_train03-04-06_val09_T1_thr0.35_tau2_depthinit/best_snn_sfm.pth")
    parser.add_argument("--snn-model-py", default="/home/larl/snn/monodepth_snn_sparse_exec/models.py")
    parser.add_argument("--snn-ckpt", default="/home/larl/snn/monodepth_snn_sparse_exec/outputs/reviewer_30ep_resume_rebuilt/snn_sfm_train03-04-06_val09_T4_thr0.35_tau2_delta_latency_lifspike_sparse_slconv1-conv2_aw0.2_ds1_md0.5-80_fd0r0.9_ph256m128_pn1_cr0m0.05_es0m0.05_hs0w0.5_hp0_seq_balanced_pc0.05_fz2_pairpose_depthinit/best_snn_sfm.pth")
    parser.add_argument("--ann-label", default="ANN front-end")
    parser.add_argument("--snn-label", default="SNN front-end")
    parser.add_argument("--output-dir", default="/home/larl/snn/monodepth_snn_sparse_exec/outputs/frontend_vo_compare")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_dir(args.output_dir)

    dataset = KITTIOdometryTriplet(
        kitti_root=args.kitti_root,
        seqs=[args.seq_id],
        resize=(args.height, args.width),
        time_steps=4,
        spike_input=False,
    )

    spec_list = [
        {
            "label": args.ann_label,
            "model_py": args.ann_model_py,
            "ckpt": args.ann_ckpt,
        },
        {
            "label": args.snn_label,
            "model_py": args.snn_model_py,
            "ckpt": args.snn_ckpt,
        },
    ]

    results = []
    rows = []
    for spec in spec_list:
        model, config = load_model(spec["model_py"], spec["ckpt"], device)
        config["time_steps"] = config.get("time_steps", 1)
        pred, gt, vo_metrics = evaluate_model_trajectory(
            model=model,
            config=config,
            seq_id=args.seq_id,
            kitti_root=args.kitti_root,
            height=args.height,
            width=args.width,
            max_frames=args.max_frames,
            device=device,
        )
        bench_metrics = benchmark_model(model, config, dataset, device, args.max_pairs)
        row = build_row(spec["label"], spec["ckpt"], config, vo_metrics, bench_metrics)
        rows.append(row)
        pred_path = os.path.join(args.output_dir, f"{spec['label'].lower().replace(' ', '_')}_pred_traj_seq{args.seq_id}.npy")
        gt_path = os.path.join(args.output_dir, f"{spec['label'].lower().replace(' ', '_')}_gt_traj_seq{args.seq_id}.npy")
        np.save(pred_path, pred)
        np.save(gt_path, gt)
        save_json(
            os.path.join(args.output_dir, f"{spec['label'].lower().replace(' ', '_')}_metrics_seq{args.seq_id}.json"),
            {
                "label": spec["label"],
                "checkpoint": spec["ckpt"],
                "config": config,
                "vo_metrics": vo_metrics,
                "benchmark_metrics": bench_metrics,
                "pred_path": pred_path,
                "gt_path": gt_path,
            },
        )
        results.append({"label": spec["label"], "pred": pred, "gt": gt, "metrics": vo_metrics})

    csv_path = os.path.join(args.output_dir, f"frontend_vo_compare_seq{args.seq_id}.csv")
    json_path = os.path.join(args.output_dir, f"frontend_vo_compare_seq{args.seq_id}.json")
    plot_path = os.path.join(args.output_dir, f"frontend_vo_compare_seq{args.seq_id}.png")
    save_csv(csv_path, rows)
    save_json(json_path, {"seq_id": args.seq_id, "max_frames": args.max_frames, "rows": rows})
    plot_trajectories(plot_path, results)

    for row in rows:
        print(
            f"[compare] {row['label']} ate_rmse={row['ate_rmse']:.4f} "
            f"ate_mean={row['ate_mean']:.4f} rpe_t={row['rpe_trans_mean']:.4f} "
            f"rpe_r_deg={row['rpe_rot_mean_deg']:.4f} latency_ms={row['total_ms_mean']:.3f}"
        )
    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main(parse_args())
