import argparse
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    KITTIDepthDataset,
    aggregate_depth_metrics,
    compute_depth_metrics,
    create_summary_writer,
    depth_l1_loss,
    ensure_dir,
    format_metrics,
    measure_inference_ms,
    paired_grouped_split,
    read_list_file,
    save_depth_visualization,
    set_seed,
    write_csv_row,
    write_json,
)
from models import MonoDepthSNN_Spike, SimpleEncoder


def build_experiment_name(args) -> str:
    init_tag = "anninit" if args.init_from_ann else "scratch"
    threshold_tag = "lthr" if args.learnable_threshold else "fthr"
    return (
        f"snn_depth_{args.input_encoding}"
        f"_T{args.time_steps}"
        f"_thr{args.v_threshold:g}"
        f"_tau{args.tau:g}"
        f"_{threshold_tag}"
        f"_{init_tag}"
    )


def resolve_output_dir(args) -> Tuple[str, str]:
    base_dir = args.output_dir
    if args.auto_experiment_dir:
        base_dir = os.path.join(base_dir, build_experiment_name(args))
    ensure_dir(base_dir)
    return base_dir, build_experiment_name(args)


def append_experiment_index(index_csv: str, row: Dict[str, float]) -> None:
    write_csv_row(index_csv, row)


def spike_regularization(model: MonoDepthSNN_Spike, target_rate: float = 0.15) -> torch.Tensor:
    stats = model.get_spike_stats()
    rates = [value for key, value in stats.items() if key != "avg_spike_rate"]
    if not rates:
        return next(model.parameters()).new_tensor(0.0)
    rate_tensor = next(model.parameters()).new_tensor(rates)
    return torch.mean((rate_tensor - target_rate) ** 2)


def spike_lambda_for_epoch(args, epoch: int) -> float:
    if epoch <= args.spike_warmup_epochs:
        return float(args.lambda_spike_start)
    if args.spike_ramp_epochs <= 0:
        return float(args.lambda_spike)
    progress = min(1.0, float(epoch - args.spike_warmup_epochs) / float(args.spike_ramp_epochs))
    return float(args.lambda_spike_start + progress * (args.lambda_spike - args.lambda_spike_start))


def build_depth_splits(args) -> Dict[str, KITTIDepthDataset]:
    image_paths = read_list_file(args.image_list_file)
    depth_paths = read_list_file(args.depth_list_file)
    splits = paired_grouped_split(
        image_paths,
        depth_paths,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    return {name: KITTIDepthDataset(samples, resize=(args.height, args.width)) for name, samples in splits.items() if samples}


def evaluate_depth(
    model,
    loader,
    device: torch.device,
    num_steps: int,
    target_spike_rate: float,
    input_encoding: str,
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    model.eval()
    all_metrics = []
    total_loss = 0.0
    total_reg = 0.0
    total_batches = 0
    avg_spike_rates = []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            depth_gt = batch["depth"].to(device)
            pred_depth, _ = model(img, num_steps=num_steps, input_encoding=input_encoding)
            total_loss += float(depth_l1_loss(pred_depth, depth_gt).item())
            total_reg += float(spike_regularization(model, target_rate=target_spike_rate).item())
            total_batches += 1
            avg_spike_rates.append(model.get_spike_stats().get("avg_spike_rate", 0.0))
            for pred_item, gt_item in zip(pred_depth, depth_gt):
                all_metrics.append(compute_depth_metrics(pred_item, gt_item))
    metrics = aggregate_depth_metrics(all_metrics)
    metrics["loss"] = total_loss / max(1, total_batches)
    spike_stats = {
        "avg_spike_rate": float(sum(avg_spike_rates) / max(1, len(avg_spike_rates))),
        "spike_reg": total_reg / max(1, total_batches),
    }
    return metrics, metrics["loss"], spike_stats


def save_debug_predictions(model, dataset, device: torch.device, out_dir: str, input_encoding: str, limit: int = 3) -> None:
    ensure_dir(out_dir)
    model.eval()
    with torch.no_grad():
        for idx in range(min(limit, len(dataset))):
            sample = dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            pred_depth, _ = model(img, input_encoding=input_encoding)
            save_depth_visualization(pred_depth[0, 0], os.path.join(out_dir, f"snn_pred_{idx:03d}.png"))
            save_depth_visualization(sample["depth"][0], os.path.join(out_dir, f"snn_gt_{idx:03d}.png"))


def train_snn_depth(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    splits = build_depth_splits(args)
    train_dataset = splits["train"]
    val_dataset = splits.get("val", train_dataset)
    test_dataset = splits.get("test", val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dir, experiment_name = resolve_output_dir(args)
    print(f"[experiment] name={experiment_name}")
    print(f"[experiment] output_dir={output_dir}")
    writer = create_summary_writer(os.path.join(output_dir, "tb_snn_depth"))

    ann_encoder = SimpleEncoder(in_channels=3)
    ann_state = torch.load(args.ann_encoder_ckpt, map_location="cpu")
    ann_encoder.load_state_dict(ann_state)

    model = MonoDepthSNN_Spike(
        tau=args.tau,
        time_steps=args.time_steps,
        v_threshold=args.v_threshold,
        input_encoding=args.input_encoding,
        learnable_threshold=args.learnable_threshold,
    ).to(device)
    if args.init_from_ann:
        model.init_from_ann_encoder(ann_encoder)
    if args.resume_snn_ckpt:
        resume_ckpt = torch.load(args.resume_snn_ckpt, map_location=device)
        state = resume_ckpt.get("model_state", resume_ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[resume] loaded SNN checkpoint from {args.resume_snn_ckpt}")

    if args.freeze_decoder_warmup > 0:
        for module in [model.depth_decoder, model.pose_decoder]:
            for param in module.parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    best_model_path = os.path.join(output_dir, "best_snn_depth_model.pth")
    history_csv = os.path.join(output_dir, "snn_depth_history.csv")
    best_epoch = -1
    best_val_metrics = {}
    best_val_spike = {}

    for epoch in range(1, args.num_epochs + 1):
        if epoch == args.freeze_decoder_warmup + 1 and args.freeze_decoder_warmup > 0:
            for module in [model.depth_decoder, model.pose_decoder]:
                for param in module.parameters():
                    param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model.train()
        running_loss = 0.0
        running_reg = 0.0
        lambda_spike_epoch = spike_lambda_for_epoch(args, epoch)
        for step, batch in enumerate(tqdm(train_loader, desc=f"SNN epoch {epoch}/{args.num_epochs}"), start=1):
            img = batch["image"].to(device)
            depth_gt = batch["depth"].to(device)

            autocast_context = torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda")
            with autocast_context:
                pred_depth, _ = model(img, num_steps=args.time_steps, input_encoding=args.input_encoding)
                depth_loss = depth_l1_loss(pred_depth, depth_gt)
                reg_loss = spike_regularization(model, target_rate=args.target_spike_rate)
                loss = depth_loss + lambda_spike_epoch * reg_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(depth_loss.item())
            running_reg += float(reg_loss.item())

            if step % args.log_every == 0:
                stats = model.get_spike_stats()
                thr_stats = model.get_threshold_stats()
                print(
                    f"[train] epoch={epoch} step={step} depth_loss={depth_loss.item():.4f} "
                    f"spike_reg={reg_loss.item():.4f} lambda_spike={lambda_spike_epoch:.4f} "
                    f"avg_spike={stats.get('avg_spike_rate', 0.0):.4f} "
                    f"thr1={thr_stats.get('conv1', 0.0):.4f} thr5={thr_stats.get('conv5', 0.0):.4f}"
                )

        train_loss = running_loss / max(1, len(train_loader))
        train_reg = running_reg / max(1, len(train_loader))
        val_metrics, val_loss, val_spike = evaluate_depth(
            model, val_loader, device, args.time_steps, args.target_spike_rate, args.input_encoding
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_spike_reg": train_reg,
            "train_lambda_spike": lambda_spike_epoch,
            "val_loss": val_loss,
            "val_avg_spike_rate": val_spike["avg_spike_rate"],
            **val_metrics,
        }
        write_csv_row(history_csv, row)
        writer.add_scalar("snn/train_loss", train_loss, epoch)
        writer.add_scalar("snn/train_spike_reg", train_reg, epoch)
        writer.add_scalar("snn/train_lambda_spike", lambda_spike_epoch, epoch)
        writer.add_scalar("snn/val_loss", val_loss, epoch)
        writer.add_scalar("snn/val_abs_rel", val_metrics["abs_rel"], epoch)
        writer.add_scalar("snn/val_rmse", val_metrics["rmse"], epoch)
        writer.add_scalar("snn/val_avg_spike_rate", val_spike["avg_spike_rate"], epoch)
        for name, value in model.get_threshold_stats().items():
            writer.add_scalar(f"snn_threshold/{name}", value, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": vars(args),
            "val_metrics": val_metrics,
            "val_spike": val_spike,
            "threshold_stats": model.get_threshold_stats(),
        }
        torch.save(checkpoint, os.path.join(output_dir, "latest_snn_depth_model.pth"))

        print(f"[val] epoch={epoch} {format_metrics(val_metrics)} avg_spike={val_spike['avg_spike_rate']:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_val_spike = dict(val_spike)
            torch.save(checkpoint, best_model_path)
            write_json(
                os.path.join(output_dir, "best_snn_metrics.json"),
                {"epoch": epoch, "val_metrics": val_metrics, "val_spike": val_spike, "config": vars(args)},
            )
            print(f"[best] epoch={epoch} saved best SNN checkpoint to {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    test_metrics, test_loss, test_spike = evaluate_depth(
        model, test_loader, device, args.time_steps, args.target_spike_rate, args.input_encoding
    )
    inference_ms = measure_inference_ms(model, next(iter(test_loader))["image"][:1].to(device))
    summary = {
        "test_loss": test_loss,
        "inference_ms": inference_ms,
        **test_metrics,
        **test_spike,
    }
    write_json(os.path.join(output_dir, "snn_test_metrics.json"), summary)
    writer.close()

    save_debug_predictions(model, test_dataset, device, os.path.join(output_dir, "snn_debug"), args.input_encoding)
    experiment_summary = {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_val_spike": best_val_spike,
        "best_thresholds": model.get_threshold_stats(),
        "test_metrics": summary,
    }
    write_json(os.path.join(output_dir, "experiment_summary.json"), experiment_summary)
    append_experiment_index(
        os.path.join(args.output_dir, "experiment_index.csv"),
        {
            "experiment_name": experiment_name,
            "best_epoch": best_epoch,
            "input_encoding": args.input_encoding,
            "time_steps": args.time_steps,
            "v_threshold": args.v_threshold,
            "tau": args.tau,
            "learnable_threshold": int(args.learnable_threshold),
            "init_from_ann": int(args.init_from_ann),
            "val_abs_rel": best_val_metrics.get("abs_rel", 0.0),
            "val_rmse": best_val_metrics.get("rmse", 0.0),
            "val_delta1": best_val_metrics.get("delta1", 0.0),
            "val_avg_spike_rate": best_val_spike.get("avg_spike_rate", 0.0),
            "test_abs_rel": summary.get("abs_rel", 0.0),
            "test_rmse": summary.get("rmse", 0.0),
            "test_delta1": summary.get("delta1", 0.0),
            "test_avg_spike_rate": summary.get("avg_spike_rate", 0.0),
            "inference_ms": summary.get("inference_ms", 0.0),
        },
    )
    print(
        f"[test] {format_metrics(test_metrics)} avg_spike={test_spike['avg_spike_rate']:.4f} "
        f"inference_ms={inference_ms:.2f}"
    )
    return model, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SNN depth model from ANN encoder initialization.")
    parser.add_argument("--image-list-file", default="/home/larl/snn/dataset/kitti_train_images.txt")
    parser.add_argument("--depth-list-file", default="/home/larl/snn/dataset/kitti_train_depths.txt")
    parser.add_argument("--ann-encoder-ckpt", default="/home/larl/snn/monodepth_snn/outputs/ann_depth/best_ann_encoder.pth")
    parser.add_argument("--resume-snn-ckpt", default="")
    parser.add_argument("--output-dir", default="/home/larl/snn/monodepth_snn/outputs/snn_depth")
    parser.add_argument("--auto-experiment-dir", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--v-threshold", type=float, default=0.25)
    parser.add_argument("--input-encoding", choices=["rate", "analog"], default="rate")
    parser.add_argument("--target-spike-rate", type=float, default=0.15)
    parser.add_argument("--lambda-spike", type=float, default=0.01)
    parser.add_argument("--lambda-spike-start", type=float, default=0.0)
    parser.add_argument("--spike-warmup-epochs", type=int, default=2)
    parser.add_argument("--spike-ramp-epochs", type=int, default=4)
    parser.add_argument("--learnable-threshold", action="store_true")
    parser.add_argument("--freeze-decoder-warmup", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--init-from-ann", dest="init_from_ann", action="store_true")
    parser.add_argument("--no-init-from-ann", dest="init_from_ann", action="store_false")
    parser.set_defaults(init_from_ann=True)
    parser.set_defaults(auto_experiment_dir=True)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_snn_depth(args)
