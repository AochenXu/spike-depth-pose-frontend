import argparse
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must be shaped [N, 3]")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0

    rot = u @ s @ vt
    if with_scale:
        var_src = np.mean(np.sum(src_centered * src_centered, axis=1))
        scale = float(np.trace(np.diag(d) @ s) / max(var_src, 1e-12))
    else:
        scale = 1.0
    trans = dst_mean - scale * (rot @ src_mean)
    return rot, scale, trans


def sim3_align(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    rot, scale, trans = umeyama_alignment(pred_xyz.astype(np.float64), gt_xyz.astype(np.float64), with_scale=True)
    return ((scale * (rot @ pred_xyz.T)).T + trans[None, :]).astype(np.float32)


def rmse_aligned(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    pred_aligned = sim3_align(pred_xyz, gt_xyz)
    diff = pred_aligned - gt_xyz
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def sliding_window_rmse(pred_poses: np.ndarray, gt_poses: np.ndarray, window: int) -> List[Tuple[int, float]]:
    pred_xyz = pred_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]
    n = min(len(pred_xyz), len(gt_xyz))
    rows = []
    for end in range(window, n + 1):
        pred_win = pred_xyz[end - window : end]
        gt_win = gt_xyz[end - window : end]
        rows.append((end, rmse_aligned(pred_win, gt_win)))
    return rows


def save_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main(args):
    ann_pred = np.load(args.ann_pred)
    ann_gt = np.load(args.ann_gt)
    snn_pred = np.load(args.snn_pred)
    snn_gt = np.load(args.snn_gt)

    n = min(args.max_frames, len(ann_pred), len(ann_gt), len(snn_pred), len(snn_gt)) if args.max_frames > 0 else min(
        len(ann_pred), len(ann_gt), len(snn_pred), len(snn_gt)
    )
    ann_pred = ann_pred[:n]
    ann_gt = ann_gt[:n]
    snn_pred = snn_pred[:n]
    snn_gt = snn_gt[:n]

    ann_xyz = ann_pred[:, :3, 3]
    ann_gt_xyz = ann_gt[:, :3, 3]
    snn_xyz = snn_pred[:, :3, 3]
    snn_gt_xyz = snn_gt[:, :3, 3]

    ann_full = sim3_align(ann_xyz, ann_gt_xyz)
    snn_full = sim3_align(snn_xyz, snn_gt_xyz)
    ann_full_rmse = rmse_aligned(ann_xyz, ann_gt_xyz)
    snn_full_rmse = rmse_aligned(snn_xyz, snn_gt_xyz)

    ann_curve = sliding_window_rmse(ann_pred, ann_gt, args.window)
    snn_curve = sliding_window_rmse(snn_pred, snn_gt, args.window)

    csv_rows = []
    for (frame_ann, ann_rmse), (frame_snn, snn_rmse) in zip(ann_curve, snn_curve):
        csv_rows.append([frame_ann, ann_rmse, snn_rmse])

    ensure_dir(os.path.dirname(args.output_path) or ".")
    save_csv(args.csv_path, ["frame_end", "ann_short_window_rmse", "snn_short_window_rmse"], csv_rows)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(12.4, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.9])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.plot(ann_gt_xyz[:, 0], ann_gt_xyz[:, 2], color="#111827", linewidth=2.5, label="GT")
    ax1.plot(ann_full[:, 0], ann_full[:, 2], color="#0f766e", linewidth=2.5, label=args.ann_label)
    ax1.set_title(f"{args.ann_label} Full Trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.25)
    ax1.text(
        0.03,
        0.97,
        f"Full Sim(3) RMSE {ann_full_rmse:.2f} m",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    ax2.plot(snn_gt_xyz[:, 0], snn_gt_xyz[:, 2], color="#111827", linewidth=2.5, label="GT")
    ax2.plot(snn_full[:, 0], snn_full[:, 2], color="#b45309", linewidth=2.5, label=args.snn_label)
    ax2.set_title(f"{args.snn_label} Full Trajectory")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, alpha=0.25)
    ax2.text(
        0.03,
        0.97,
        f"Full Sim(3) RMSE {snn_full_rmse:.2f} m",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    ax3.plot([x for x, _ in ann_curve], [y for _, y in ann_curve], color="#0f766e", linewidth=2.3, label=args.ann_label)
    ax3.plot([x for x, _ in snn_curve], [y for _, y in snn_curve], color="#b45309", linewidth=2.3, label=args.snn_label)
    ax3.set_title(f"Local Frame-Trajectory Quality ({args.window}-frame sliding window)")
    ax3.set_xlabel("Frame index (window end)")
    ax3.set_ylabel("Short-window Sim(3) RMSE (m)")
    ax3.grid(True, alpha=0.25)
    ax3.text(
        0.99,
        0.97,
        "Each point compares one local frame trajectory window against GT",
        transform=ax3.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    handles = [
        plt.Line2D([0], [0], color="#111827", linewidth=2.5, label="GT"),
        plt.Line2D([0], [0], color="#0f766e", linewidth=2.5, label=args.ann_label),
        plt.Line2D([0], [0], color="#b45309", linewidth=2.5, label=args.snn_label),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"KITTI {args.seq_label}: frame-trajectory comparison", fontsize=14)
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.output_path}")
    print(f"[saved] {args.csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a frame-trajectory comparison figure from exported ANN/SNN trajectories.")
    parser.add_argument("--ann-pred", required=True)
    parser.add_argument("--ann-gt", required=True)
    parser.add_argument("--snn-pred", required=True)
    parser.add_argument("--snn-gt", required=True)
    parser.add_argument("--ann-label", default="ANN")
    parser.add_argument("--snn-label", default="SNN")
    parser.add_argument("--seq-label", default="seq09")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--csv-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
