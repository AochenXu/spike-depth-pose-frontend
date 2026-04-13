import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_snn_sfm_kitti.py"
VO_SCRIPT = ROOT / "eval_snn_vo_ate.py"
BENCH_SCRIPT = ROOT / "benchmark_snn_frontends.py"


def parse_csv_list(text: str):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def format_seq_tag(text: str) -> str:
    seqs = parse_csv_list(text)
    if not seqs:
        return "none"
    return "-".join(seqs)


def build_experiment_name(args) -> str:
    init_tag = "depthinit" if args.snn_depth_ckpt else "anninit"
    sparse_tag = "sparse"
    return (
        f"snn_sfm"
        f"_train{format_seq_tag(args.train_seqs)}"
        f"_val{format_seq_tag(args.val_seqs)}"
        f"_T{args.time_steps}"
        f"_thr{args.v_threshold:g}"
        f"_tau{args.tau:g}"
        f"_{args.input_encoding}"
        f"_lif{args.lif_output_mode}"
        f"_{sparse_tag}"
        f"_sl{args.sparse_layers.replace(',', '-')}"
        f"_aw{args.delta_anchor_weight:g}"
        f"_ds{args.decoder_channel_scale:g}"
        f"_md{args.min_depth:g}-{args.max_depth:g}"
        f"_fd{args.lambda_far_depth:g}r{args.far_depth_start_ratio:g}"
        f"_ph{args.pose_hidden_channels}m{args.pose_mlp_hidden}"
        f"_pn{int(args.pose_input_normalization)}"
        f"_cr{args.lambda_change_rank:g}m{args.change_rank_margin:g}"
        f"_es{args.lambda_early_spike:g}m{args.early_spike_margin:g}"
        f"_hs0w0.5"
        f"_hp{int(args.hybrid_pose_diff)}"
        f"_{args.train_sampling}"
        f"_pc{args.lambda_pose_consistency:g}"
        f"_fz{args.freeze_depth_epochs}"
        f"_pairpose"
        f"_{init_tag}"
    )


def resolve_eval_seq(args) -> str:
    if args.eval_seq:
        return args.eval_seq
    val_seqs = parse_csv_list(args.val_seqs)
    if len(val_seqs) != 1:
        raise ValueError("--eval-seq is required when --val-seqs contains multiple sequences.")
    return val_seqs[0]


def run_command(cmd, env, dry_run: bool) -> None:
    print("[cmd]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def find_latest_experiment_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(args):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    python_bin = sys.executable
    eval_seq = resolve_eval_seq(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        python_bin,
        str(TRAIN_SCRIPT),
        "--train-seqs",
        args.train_seqs,
        "--val-seqs",
        args.val_seqs,
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--tau",
        str(args.tau),
        "--time-steps",
        str(args.time_steps),
        "--v-threshold",
        str(args.v_threshold),
        "--input-encoding",
        args.input_encoding,
        "--lif-output-mode",
        args.lif_output_mode,
        "--train-sampling",
        args.train_sampling,
        "--sparse-exec",
        "--sparse-layers",
        args.sparse_layers,
        "--sparse-activity-threshold",
        str(args.sparse_activity_threshold),
        "--sparse-fallback-ratio",
        str(args.sparse_fallback_ratio),
        "--delta-anchor-weight",
        str(args.delta_anchor_weight),
        "--decoder-channel-scale",
        str(args.decoder_channel_scale),
        "--min-depth",
        str(args.min_depth),
        "--max-depth",
        str(args.max_depth),
        "--lambda-far-depth",
        str(args.lambda_far_depth),
        "--far-depth-start-ratio",
        str(args.far_depth_start_ratio),
        "--lambda-change-rank",
        str(args.lambda_change_rank),
        "--change-rank-margin",
        str(args.change_rank_margin),
        "--change-high-quantile",
        str(args.change_high_quantile),
        "--change-low-quantile",
        str(args.change_low_quantile),
        "--lambda-early-spike",
        str(args.lambda_early_spike),
        "--early-spike-margin",
        str(args.early_spike_margin),
        "--early-diff-threshold",
        str(args.early_diff_threshold),
        "--early-time-threshold",
        str(args.early_time_threshold),
        "--pose-hidden-channels",
        str(args.pose_hidden_channels),
        "--pose-mlp-hidden",
        str(args.pose_mlp_hidden),
        *(["--pose-input-normalization"] if args.pose_input_normalization else []),
        *(["--hybrid-pose-diff"] if args.hybrid_pose_diff else []),
        "--freeze-depth-epochs",
        str(args.freeze_depth_epochs),
        "--lambda-pose-consistency",
        str(args.lambda_pose_consistency),
        "--output-dir",
        str(output_dir),
    ]
    if args.ann_encoder_ckpt:
        train_cmd.extend(["--ann-encoder-ckpt", args.ann_encoder_ckpt])
    if args.snn_depth_ckpt:
        train_cmd.extend(["--snn-depth-ckpt", args.snn_depth_ckpt])
    if args.amp:
        train_cmd.append("--amp")
    if args.device:
        train_cmd.extend(["--device", args.device])

    run_command(train_cmd, env, args.dry_run)

    exp_dir = output_dir / build_experiment_name(args) if args.dry_run else find_latest_experiment_dir(output_dir)

    vo_cmd = [
        python_bin,
        str(VO_SCRIPT),
        "--experiment-dir",
        str(exp_dir),
        "--seq-id",
        eval_seq,
        "--max-frames",
        str(args.max_frames),
        "--output-dir",
        str(exp_dir / "vo_eval"),
    ]
    if args.device:
        vo_cmd.extend(["--device", args.device])
    run_command(vo_cmd, env, args.skip_vo or args.dry_run)

    bench_cmd = [
        python_bin,
        str(BENCH_SCRIPT),
        "--seq-id",
        eval_seq,
        "--max-pairs",
        str(args.max_pairs),
        "--new-ckpt",
        str(exp_dir / "best_snn_sfm.pth"),
        "--output-dir",
        str(exp_dir / "benchmark"),
    ]
    if args.old_ckpt:
        bench_cmd.extend(["--old-ckpt", args.old_ckpt])
    if args.old_model_py:
        bench_cmd.extend(["--old-model-py", args.old_model_py])
    if args.old_time_steps is not None:
        bench_cmd.extend(["--old-time-steps", str(args.old_time_steps)])
    if args.device:
        bench_cmd.extend(["--device", args.device])
    run_command(bench_cmd, env, args.skip_benchmark or args.dry_run)

    if args.dry_run:
        return

    print(f"[done] experiment_dir={exp_dir}")
    print(f"[done] summary={exp_dir / 'experiment_summary.json'}")
    print(f"[done] vo={exp_dir / 'vo_eval' / f'vo_ate_seq{eval_seq}.json'}")
    print(f"[done] bench={exp_dir / 'benchmark' / f'frontend_benchmark_seq{eval_seq}.json'}")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the current lif-spike sparse mainline with a custom train/val split.")
    parser.add_argument("--output-dir", default=str(script_dir / "outputs" / "lif_spike_split_check"))
    parser.add_argument("--train-seqs", default="00,03,04,06")
    parser.add_argument("--val-seqs", default="08")
    parser.add_argument("--eval-seq", default="")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--v-threshold", type=float, default=0.35)
    parser.add_argument("--input-encoding", default="delta_latency")
    parser.add_argument("--lif-output-mode", default="spike")
    parser.add_argument("--train-sampling", default="seq_balanced")
    parser.add_argument("--sparse-layers", default="conv1,conv2")
    parser.add_argument("--sparse-activity-threshold", type=float, default=0.4)
    parser.add_argument("--sparse-fallback-ratio", type=float, default=0.6)
    parser.add_argument("--delta-anchor-weight", type=float, default=0.2)
    parser.add_argument("--decoder-channel-scale", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--lambda-far-depth", type=float, default=0.0)
    parser.add_argument("--far-depth-start-ratio", type=float, default=0.9)
    parser.add_argument("--lambda-change-rank", type=float, default=0.02)
    parser.add_argument("--change-rank-margin", type=float, default=0.05)
    parser.add_argument("--change-high-quantile", type=float, default=0.75)
    parser.add_argument("--change-low-quantile", type=float, default=0.25)
    parser.add_argument("--lambda-early-spike", type=float, default=0.05)
    parser.add_argument("--early-spike-margin", type=float, default=0.05)
    parser.add_argument("--early-diff-threshold", type=float, default=0.05)
    parser.add_argument("--early-time-threshold", type=float, default=0.5)
    parser.add_argument("--pose-hidden-channels", type=int, default=256)
    parser.add_argument("--pose-mlp-hidden", type=int, default=128)
    parser.add_argument("--pose-input-normalization", action="store_true")
    parser.add_argument("--hybrid-pose-diff", action="store_true")
    parser.add_argument("--freeze-depth-epochs", type=int, default=1)
    parser.add_argument("--lambda-pose-consistency", type=float, default=0.05)
    parser.add_argument("--ann-encoder-ckpt", default="")
    parser.add_argument("--snn-depth-ckpt", default="")
    parser.add_argument("--old-model-py", default="")
    parser.add_argument("--old-ckpt", default="")
    parser.add_argument("--old-time-steps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-vo", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="")
    parser.set_defaults(pose_input_normalization=True, hybrid_pose_diff=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
