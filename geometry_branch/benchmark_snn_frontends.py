import argparse
import csv
import importlib.util
import inspect
import json
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from sfm_common import KITTIOdometryTriplet


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_model(model_py: str, ckpt_path: str, device: torch.device):
    module = load_module(f"mod_{abs(hash(model_py))}", model_py)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    ctor = module.MonoDepthSNN_Spike
    sig = inspect.signature(ctor.__init__)
    kwargs = {}
    for key, default in [
        ("tau", 2.0),
        ("time_steps", 1),
        ("v_threshold", 0.35),
        ("input_encoding", "analog"),
        ("lif_output_mode", "mixed"),
        ("learnable_threshold", False),
        ("sparse_exec", False),
        ("sparse_layers", ["conv1", "conv2"]),
        ("sparse_activity_threshold", 0.4),
        ("sparse_fallback_ratio", 0.6),
        ("delta_anchor_weight", 0.2),
        ("decoder_channel_scale", 1.0),
        ("min_depth", 0.5),
        ("max_depth", 80.0),
        ("pose_hidden_channels", 256),
        ("pose_mlp_hidden", 128),
        ("pose_input_normalization", False),
        ("hybrid_static_branch", False),
        ("hybrid_static_weight", 0.5),
        ("hybrid_pose_diff", False),
    ]:
        if key in sig.parameters:
            kwargs[key] = config.get(key, default)
    model = ctor(**kwargs)
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in ckpt["model_state"].items()
        if key in current and current[key].shape == value.shape
    }
    current.update(filtered)
    model.load_state_dict(current, strict=False)
    model.to(device)
    model.eval()
    return model, config


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / float(iters)


def forward_depth(model, config: Dict[str, Any], img_prev: torch.Tensor, img_t: torch.Tensor):
    kwargs = {
        "num_steps": config.get("time_steps", 1),
        "input_encoding": config.get("input_encoding", "analog"),
    }
    if str(config.get("input_encoding")) in {"delta_latency", "delta_latency_anchor"}:
        kwargs["img_prev"] = img_prev
    return model(img_t, **kwargs)


def get_sparse_stats(model) -> Dict[str, float]:
    if hasattr(model, "get_sparse_stats"):
        return model.get_sparse_stats()
    return {"avg_active_ratio": 1.0, "avg_used_sparse": 0.0}


def benchmark_model(
    model,
    config: Dict[str, Any],
    dataset: KITTIOdometryTriplet,
    device: torch.device,
    max_pairs: int,
) -> Dict[str, Any]:
    depth_ms_list = []
    pose_ms_list = []
    total_ms_list = []
    avg_spikes = []
    avg_active = []
    avg_used_sparse = []

    num_pairs = min(max_pairs, len(dataset) - 1)
    with torch.no_grad():
        for idx in range(num_pairs):
            sample = dataset[idx]
            img_prev = sample["img_prev"].unsqueeze(0).to(device)
            img_t = sample["img_t"].unsqueeze(0).to(device)
            img_next = sample["img_next"].unsqueeze(0).to(device)

            def depth_fn():
                sync_if_cuda(device)
                forward_depth(model, config, img_prev, img_t)
                sync_if_cuda(device)

            def pose_fn():
                sync_if_cuda(device)
                model.predict_pose_pair(img_t, img_next)
                sync_if_cuda(device)

            def total_fn():
                sync_if_cuda(device)
                forward_depth(model, config, img_prev, img_t)
                model.predict_pose_pair(img_t, img_next)
                sync_if_cuda(device)

            depth_ms = measure_ms(depth_fn, warmup=1, iters=1)
            pose_ms = measure_ms(pose_fn, warmup=1, iters=1)
            total_ms = measure_ms(total_fn, warmup=1, iters=1)

            depth_ms_list.append(depth_ms)
            pose_ms_list.append(pose_ms)
            total_ms_list.append(total_ms)

            forward_depth(model, config, img_prev, img_t)
            avg_spikes.append(float(model.get_spike_stats().get("avg_spike_rate", 0.0)))
            sparse_stats = get_sparse_stats(model)
            avg_active.append(float(sparse_stats.get("avg_active_ratio", 1.0)))
            avg_used_sparse.append(float(sparse_stats.get("avg_used_sparse", 0.0)))

    return {
        "num_pairs": num_pairs,
        "depth_ms_mean": float(np.mean(depth_ms_list)),
        "depth_ms_std": float(np.std(depth_ms_list)),
        "pose_ms_mean": float(np.mean(pose_ms_list)),
        "pose_ms_std": float(np.std(pose_ms_list)),
        "total_ms_mean": float(np.mean(total_ms_list)),
        "total_ms_std": float(np.std(total_ms_list)),
        "avg_spike_rate": float(np.mean(avg_spikes)),
        "avg_active_ratio": float(np.mean(avg_active)),
        "avg_used_sparse": float(np.mean(avg_used_sparse)),
        "params_million": float(sum(p.numel() for p in model.parameters()) / 1e6),
    }


def save_table(path: str, rows: Tuple[Dict[str, Any], ...]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = KITTIOdometryTriplet(
        kitti_root=args.kitti_root,
        seqs=[args.seq_id],
        resize=(args.height, args.width),
        time_steps=max(args.old_time_steps, args.new_time_steps),
        spike_input=False,
    )

    old_model, old_config = load_model(args.old_model_py, args.old_ckpt, device)
    new_model, new_config = load_model(args.new_model_py, args.new_ckpt, device)
    old_config["time_steps"] = old_config.get("time_steps", args.old_time_steps)
    new_config["time_steps"] = new_config.get("time_steps", args.new_time_steps)

    old_result = benchmark_model(old_model, old_config, dataset, device, args.max_pairs)
    new_result = benchmark_model(new_model, new_config, dataset, device, args.max_pairs)

    rows = (
        {
            "label": "dense_pairpose",
            "checkpoint": args.old_ckpt,
            "input_encoding": old_config.get("input_encoding", "analog"),
            "time_steps": old_config.get("time_steps", 1),
            **old_result,
        },
        {
            "label": "delta_latency_sparse",
            "checkpoint": args.new_ckpt,
            "input_encoding": new_config.get("input_encoding", "delta_latency"),
            "time_steps": new_config.get("time_steps", 4),
            **new_result,
        },
    )

    ensure_dir(args.output_dir)
    json_path = os.path.join(args.output_dir, f"frontend_benchmark_seq{args.seq_id}.json")
    csv_path = os.path.join(args.output_dir, f"frontend_benchmark_seq{args.seq_id}.csv")
    with open(json_path, "w") as f:
        json.dump({"device": str(device), "rows": rows}, f, indent=2)
    save_table(csv_path, rows)

    for row in rows:
        print(
            f"[bench] {row['label']} total_ms={row['total_ms_mean']:.3f} "
            f"depth_ms={row['depth_ms_mean']:.3f} pose_ms={row['pose_ms_mean']:.3f} "
            f"spike={row['avg_spike_rate']:.4f} active={row['avg_active_ratio']:.4f} "
            f"used_sparse={row['avg_used_sparse']:.4f}"
        )
    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark dense and sparse SNN front-ends on KITTI odometry pairs.")
    parser.add_argument("--kitti-root", default="/home/larl/kitti_dataset/dataset")
    parser.add_argument("--seq-id", default="09")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--old-model-py", default="/home/larl/snn/monodepth_snn/models.py")
    parser.add_argument("--old-ckpt", default="/home/larl/snn/monodepth_snn/outputs/snn_sfm/snn_sfm_train03-04-06_val09_T1_thr0.35_tau2_depthinit/best_snn_sfm.pth")
    parser.add_argument("--old-time-steps", type=int, default=1)
    parser.add_argument("--new-model-py", default="/home/larl/snn/monodepth_snn_sparse_exec/models.py")
    parser.add_argument("--new-ckpt", default="/home/larl/snn/monodepth_snn_sparse_exec/outputs/snn_sfm/snn_sfm_train03-04-06_val09_T4_thr0.35_tau2_delta_latency_sparse_seq_balanced_pc0.05_fz0_pairpose_depthinit/best_snn_sfm.pth")
    parser.add_argument("--new-time-steps", type=int, default=4)
    parser.add_argument("--output-dir", default="/home/larl/snn/monodepth_snn_sparse_exec/outputs/benchmark")
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
