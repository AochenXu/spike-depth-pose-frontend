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
from models import MonoDepthSNN_RGB


def build_experiment_name(args) -> str:
    decoder_tag = "md2skip"
    return f"ann_depth_{decoder_tag}_lr{args.lr:g}_ep{args.num_epochs}"


def resolve_output_dir(args) -> Tuple[str, str]:
    base_dir = args.output_dir
    if args.auto_experiment_dir:
        base_dir = os.path.join(base_dir, build_experiment_name(args))
    ensure_dir(base_dir)
    return base_dir, build_experiment_name(args)


def append_experiment_index(index_csv: str, row: Dict[str, float]) -> None:
    write_csv_row(index_csv, row)


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


def evaluate_depth(model, loader, device: torch.device) -> Tuple[Dict[str, float], float]:
    model.eval()
    all_metrics = []
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            depth_gt = batch["depth"].to(device)
            pred_depth, _ = model(img)
            total_loss += float(depth_l1_loss(pred_depth, depth_gt).item())
            total_batches += 1
            for pred_item, gt_item in zip(pred_depth, depth_gt):
                all_metrics.append(compute_depth_metrics(pred_item, gt_item))
    metrics = aggregate_depth_metrics(all_metrics)
    metrics["loss"] = total_loss / max(1, total_batches)
    return metrics, metrics["loss"]


def save_debug_predictions(model, dataset, device: torch.device, out_dir: str, limit: int = 3) -> None:
    ensure_dir(out_dir)
    model.eval()
    with torch.no_grad():
        for idx in range(min(limit, len(dataset))):
            sample = dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            pred_depth, _ = model(img)
            save_depth_visualization(pred_depth[0, 0], os.path.join(out_dir, f"ann_pred_{idx:03d}.png"))
            save_depth_visualization(sample["depth"][0], os.path.join(out_dir, f"ann_gt_{idx:03d}.png"))


def train_ann_depth(args):
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
    writer = create_summary_writer(os.path.join(output_dir, "tb_ann_depth"))

    model = MonoDepthSNN_RGB().to(device)
    if args.resume_ann_ckpt:
        state = torch.load(args.resume_ann_ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[resume] loaded ANN checkpoint from {args.resume_ann_ckpt}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    best_model_path = os.path.join(output_dir, "best_ann_depth_model.pth")
    best_encoder_path = os.path.join(output_dir, "best_ann_encoder.pth")
    history_csv = os.path.join(output_dir, "ann_depth_history.csv")
    best_epoch = -1
    best_val_metrics = {}

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"ANN epoch {epoch}/{args.num_epochs}"), start=1):
            img = batch["image"].to(device)
            depth_gt = batch["depth"].to(device)

            autocast_context = torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda")
            with autocast_context:
                pred_depth, _ = model(img)
                loss = depth_l1_loss(pred_depth, depth_gt)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            if step % args.log_every == 0:
                print(f"[train] epoch={epoch} step={step} depth_loss={loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics, val_loss = evaluate_depth(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        write_csv_row(history_csv, row)
        writer.add_scalar("ann/train_loss", train_loss, epoch)
        writer.add_scalar("ann/val_loss", val_loss, epoch)
        writer.add_scalar("ann/val_abs_rel", val_metrics["abs_rel"], epoch)
        writer.add_scalar("ann/val_rmse", val_metrics["rmse"], epoch)

        model_state = model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "config": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(model_state, os.path.join(output_dir, "latest_ann_depth_model.pth"))
        torch.save(model.encoder.state_dict(), os.path.join(output_dir, "latest_ann_encoder.pth"))

        print(f"[val] epoch={epoch} {format_metrics(val_metrics)}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            torch.save(model_state, best_model_path)
            torch.save(model.encoder.state_dict(), best_encoder_path)
            write_json(
                os.path.join(output_dir, "best_ann_metrics.json"),
                {"epoch": epoch, "val_metrics": val_metrics, "config": vars(args)},
            )
            print(f"[best] epoch={epoch} saved best ANN checkpoint to {best_model_path}")

    best_state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_state, strict=False)
    test_metrics, test_loss = evaluate_depth(model, test_loader, device)
    inference_ms = measure_inference_ms(model, next(iter(test_loader))["image"][:1].to(device))
    summary = {
        "test_loss": test_loss,
        "inference_ms": inference_ms,
        **test_metrics,
    }
    write_json(os.path.join(output_dir, "ann_test_metrics.json"), summary)
    writer.close()

    save_debug_predictions(model, test_dataset, device, os.path.join(output_dir, "ann_debug"))
    experiment_summary = {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": summary,
    }
    write_json(os.path.join(output_dir, "experiment_summary.json"), experiment_summary)
    append_experiment_index(
        os.path.join(args.output_dir, "experiment_index.csv"),
        {
            "experiment_name": experiment_name,
            "best_epoch": best_epoch,
            "val_abs_rel": best_val_metrics.get("abs_rel", 0.0),
            "val_rmse": best_val_metrics.get("rmse", 0.0),
            "val_delta1": best_val_metrics.get("delta1", 0.0),
            "test_abs_rel": summary.get("abs_rel", 0.0),
            "test_rmse": summary.get("rmse", 0.0),
            "test_delta1": summary.get("delta1", 0.0),
            "inference_ms": summary.get("inference_ms", 0.0),
        },
    )
    print(f"[test] {format_metrics(test_metrics)} inference_ms={inference_ms:.2f}")
    return model, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train ANN depth model with the current depth decoder.")
    parser.add_argument("--image-list-file", default="/home/larl/snn/dataset/kitti_train_images.txt")
    parser.add_argument("--depth-list-file", default="/home/larl/snn/dataset/kitti_train_depths.txt")
    parser.add_argument("--output-dir", default="/home/larl/snn/monodepth_snn/outputs/ann_depth_retrained")
    parser.add_argument("--auto-experiment-dir", action="store_true")
    parser.add_argument("--resume-ann-ckpt", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=8)
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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="")
    parser.set_defaults(auto_experiment_dir=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ann_depth(args)
