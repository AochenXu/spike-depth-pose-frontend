import argparse
import os
from pathlib import Path

import numpy as np
import torch

from common import write_csv_row, write_json
from eval_snn_vo_ate import load_image, pose_vec_to_matrix, resolve_ckpt_path
from models import MonoDepthSNN_Spike
from sfm_common import load_intrinsics, scale_intrinsics
from slam_backend import compute_overlap_mae, fuse_depth_maps, reproject_depth_map, weighted_fuse_depth_maps


def predict_depth(model, image: torch.Tensor, prev_image: torch.Tensor, config, args, device) -> np.ndarray:
    kwargs = {
        "num_steps": config.get("time_steps", args.time_steps),
        "input_encoding": config.get("input_encoding", args.input_encoding),
    }
    if kwargs["input_encoding"] in {"delta_latency", "delta_latency_anchor"}:
        kwargs["img_prev"] = prev_image.unsqueeze(0).to(device)
    with torch.no_grad():
        depth, _ = model(image.unsqueeze(0).to(device), **kwargs)
    return depth[0, 0].detach().cpu().numpy().astype(np.float32)


def predict_pose_src_to_ref(model, src: torch.Tensor, ref: torch.Tensor, device) -> np.ndarray:
    with torch.no_grad():
        pose = model.predict_pose_pair(src.unsqueeze(0).to(device), ref.unsqueeze(0).to(device))
    return pose_vec_to_matrix(pose[0])


def build_depth_fusion(
    ref_depth: np.ndarray,
    warped_depths,
    hit_counts,
    offsets,
    mode: str,
):
    depth_maps = [ref_depth] + warped_depths
    if mode == "median":
        return fuse_depth_maps(depth_maps)
    if mode != "weighted":
        raise ValueError(f"Unsupported fusion mode: {mode}")

    base_weights = [1.0]
    confidence_maps = [np.ones_like(ref_depth, dtype=np.float32)]
    for hit_count, offset in zip(hit_counts, offsets):
        base_weights.append(1.0 / float(abs(offset) + 1))
        confidence_maps.append(np.sqrt(np.maximum(hit_count.astype(np.float32), 0.0)))
    return weighted_fuse_depth_maps(depth_maps, base_weights, confidence_maps=confidence_maps)


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
        sparse_exec=config.get("sparse_exec", False),
        sparse_layers=config.get("sparse_layers", ["conv1", "conv2"]),
        sparse_activity_threshold=config.get("sparse_activity_threshold", 0.4),
        sparse_fallback_ratio=config.get("sparse_fallback_ratio", 0.6),
        delta_anchor_weight=config.get("delta_anchor_weight", 0.2),
        decoder_channel_scale=config.get("decoder_channel_scale", 1.0),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    seq_dir = Path(args.kitti_root) / "sequences" / args.seq_id
    img_dir = seq_dir / "image_2"
    calib_path = seq_dir / "calib.txt"
    image_names = sorted(p.name for p in img_dir.glob("*.png"))
    if args.max_frames > 0:
        image_names = image_names[: args.max_frames]
    intrinsics = load_intrinsics(str(calib_path))
    intrinsics = scale_intrinsics(intrinsics, orig_size=(1242, 375), new_size=(args.width, args.height))

    out_dir = args.output_dir
    if args.auto_output_dir:
        out_dir = os.path.join(
            out_dir,
            f"geometry_eval_{Path(ckpt_path).stem}_{args.seq_id}_w{args.window_size}_n{len(image_names)}",
            f"_{args.fusion_mode}",
        )
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(out_dir, "frame_metrics.csv")

    frame_rows = []
    start = args.window_size
    end = len(image_names) - args.window_size
    for ref_idx in range(start, end):
        ref_tensor = load_image(str(img_dir / image_names[ref_idx]), (args.height, args.width))
        ref_prev_idx = max(0, ref_idx - 1)
        ref_prev_tensor = load_image(str(img_dir / image_names[ref_prev_idx]), (args.height, args.width))
        ref_depth = predict_depth(model, ref_tensor, ref_prev_tensor, config, args, device)

        warped_depths = []
        hit_counts = []
        offsets = []
        ref_neighbor_maes = []
        for offset in range(-args.window_size, args.window_size + 1):
            if offset == 0:
                continue
            src_idx = ref_idx + offset
            src_tensor = load_image(str(img_dir / image_names[src_idx]), (args.height, args.width))
            src_prev_idx = max(0, src_idx - 1)
            src_prev_tensor = load_image(str(img_dir / image_names[src_prev_idx]), (args.height, args.width))
            src_depth = predict_depth(model, src_tensor, src_prev_tensor, config, args, device)
            pose_src_to_ref = predict_pose_src_to_ref(model, src_tensor, ref_tensor, device)
            warped_depth, hit_count = reproject_depth_map(src_depth, pose_src_to_ref, intrinsics)
            warped_depths.append(warped_depth)
            hit_counts.append(hit_count)
            offsets.append(offset)
            ref_mae, overlap = compute_overlap_mae(ref_depth, warped_depth)
            if overlap > 0:
                ref_neighbor_maes.append(ref_mae)

        fused_depth, support_count = build_depth_fusion(
            ref_depth=ref_depth,
            warped_depths=warped_depths,
            hit_counts=hit_counts,
            offsets=offsets,
            mode=args.fusion_mode,
        )
        fused_neighbor_maes = []
        for warped_depth in warped_depths:
            fused_mae, overlap = compute_overlap_mae(fused_depth, warped_depth)
            if overlap > 0:
                fused_neighbor_maes.append(fused_mae)

        row = {
            "frame_idx": ref_idx,
            "avg_ref_neighbor_mae": float(np.mean(ref_neighbor_maes)) if ref_neighbor_maes else 0.0,
            "avg_fused_neighbor_mae": float(np.mean(fused_neighbor_maes)) if fused_neighbor_maes else 0.0,
            "fused_valid_pixels": int((fused_depth > 0).sum()),
            "avg_support_count": float(support_count[support_count > 0].mean()) if np.any(support_count > 0) else 0.0,
            "fusion_mode": args.fusion_mode,
        }
        frame_rows.append(row)
        write_csv_row(metrics_csv, row)

    summary = {
        "checkpoint": ckpt_path,
        "seq_id": args.seq_id,
        "window_size": args.window_size,
        "fusion_mode": args.fusion_mode,
        "num_frames_evaluated": len(frame_rows),
        "avg_ref_neighbor_mae": float(np.mean([row["avg_ref_neighbor_mae"] for row in frame_rows])) if frame_rows else 0.0,
        "avg_fused_neighbor_mae": float(np.mean([row["avg_fused_neighbor_mae"] for row in frame_rows])) if frame_rows else 0.0,
        "avg_fused_valid_pixels": float(np.mean([row["fused_valid_pixels"] for row in frame_rows])) if frame_rows else 0.0,
        "avg_support_count": float(np.mean([row["avg_support_count"] for row in frame_rows])) if frame_rows else 0.0,
    }
    write_json(os.path.join(out_dir, "geometry_eval_summary.json"), summary)
    print(
        f"[geometry_eval] seq={args.seq_id} frames={summary['num_frames_evaluated']} "
        f"ref_mae={summary['avg_ref_neighbor_mae']:.4f} "
        f"fused_mae={summary['avg_fused_neighbor_mae']:.4f} "
        f"avg_support={summary['avg_support_count']:.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate local geometry fusion consistency for the SNN front-end.")
    parser.add_argument("--kitti-root", default="/home/larl/kitti_dataset/dataset")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--ckpt-path", default="/home/larl/snn/monodepth_snn/outputs/snn_sfm/best_snn_sfm.pth")
    parser.add_argument("--output-dir", default="/home/larl/snn/monodepth_snn/outputs/geometry_backend")
    parser.add_argument("--auto-output-dir", action="store_true")
    parser.add_argument("--seq-id", default="09")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--fusion-mode", choices=["median", "weighted"], default="weighted")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=1)
    parser.add_argument("--v-threshold", type=float, default=0.35)
    parser.add_argument("--input-encoding", choices=["rate", "analog"], default="analog")
    parser.add_argument("--device", default="")
    parser.set_defaults(auto_output_dir=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
