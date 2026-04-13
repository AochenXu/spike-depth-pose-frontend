import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import create_summary_writer, ensure_dir, set_seed, write_csv_row, write_json
from models import MonoDepthSNN_Spike, SimpleEncoder
from sfm_common import (
    KITTIOdometryTriplet,
    compute_photometric_terms,
    depth_smoothness_loss,
    far_depth_penalty,
    pose_consistency_loss,
    warp_image,
)


def parse_seqs(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def format_seq_tag(text: str) -> str:
    seqs = parse_seqs(text)
    if not seqs:
        return "none"
    return "-".join(seqs)


def build_experiment_name(args) -> str:
    init_tag = "depthinit" if args.snn_depth_ckpt else "anninit"
    pose_tag = "pairpose"
    sparse_tag = "sparse" if args.sparse_exec else "dense"
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
        f"_sl{args.sparse_layers.replace(',', '-') if args.sparse_exec else 'none'}"
        f"_aw{args.delta_anchor_weight:g}"
        f"_ds{args.decoder_channel_scale:g}"
        f"_md{args.min_depth:g}-{args.max_depth:g}"
        f"_fd{args.lambda_far_depth:g}r{args.far_depth_start_ratio:g}"
        f"_ph{args.pose_hidden_channels}m{args.pose_mlp_hidden}"
        f"_pn{int(args.pose_input_normalization)}"
        f"_cr{args.lambda_change_rank:g}m{args.change_rank_margin:g}"
        f"_es{args.lambda_early_spike:g}m{args.early_spike_margin:g}"
        f"_hs{int(args.hybrid_static_branch)}w{args.hybrid_static_weight:g}"
        f"_hp{int(args.hybrid_pose_diff)}"
        f"_{args.train_sampling}"
        f"_pc{args.lambda_pose_consistency:g}"
        f"_fz{args.freeze_depth_epochs}"
        f"_{pose_tag}"
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


def summarize_seq_counts(dataset: KITTIOdometryTriplet) -> Dict[str, int]:
    return dict(Counter(sample["seq_id"] for sample in dataset.samples))


def build_train_sampler(dataset: KITTIOdometryTriplet, strategy: str):
    if strategy == "uniform":
        return None
    if strategy != "seq_balanced":
        raise ValueError(f"Unsupported train sampling strategy: {strategy}")

    seq_counts = Counter(sample["seq_id"] for sample in dataset.samples)
    weights = [1.0 / float(seq_counts[sample["seq_id"]]) for sample in dataset.samples]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset.samples),
        replacement=True,
    )


def spike_consistency_loss(model: MonoDepthSNN_Spike) -> torch.Tensor:
    stats = model.get_spike_stats()
    rates = [value for key, value in stats.items() if key != "avg_spike_rate"]
    if not rates:
        return next(model.parameters()).new_tensor(0.0)
    rate_tensor = next(model.parameters()).new_tensor(rates)
    return rate_tensor.mean()


def change_response_ranking_loss(
    diff_map: torch.Tensor,
    response_map: torch.Tensor,
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
    margin: float = 0.05,
) -> torch.Tensor:
    diff_map = F.interpolate(diff_map, size=response_map.shape[-2:], mode="bilinear", align_corners=False)
    losses = []
    for batch_idx in range(diff_map.shape[0]):
        diff_flat = diff_map[batch_idx, 0].reshape(-1)
        high_thr = torch.quantile(diff_flat, float(high_quantile))
        low_thr = torch.quantile(diff_flat, float(low_quantile))
        high_mask = (diff_map[batch_idx : batch_idx + 1] >= high_thr).float()
        low_mask = (diff_map[batch_idx : batch_idx + 1] <= low_thr).float()
        high_resp = (response_map[batch_idx : batch_idx + 1] * high_mask).sum() / (high_mask.sum() + 1e-6)
        low_resp = (response_map[batch_idx : batch_idx + 1] * low_mask).sum() / (low_mask.sum() + 1e-6)
        losses.append(F.relu(float(margin) - (high_resp - low_resp)))
    if not losses:
        return response_map.new_tensor(0.0)
    return torch.stack(losses).mean()


def earliest_spike_time_loss(
    diff_map: torch.Tensor,
    spike_steps: torch.Tensor,
    diff_threshold: float = 0.05,
    early_threshold: float = 0.5,
    margin: float = 0.05,
) -> torch.Tensor:
    steps = spike_steps.shape[1]
    if steps <= 0:
        return spike_steps.new_tensor(0.0)
    diff_map = F.interpolate(diff_map, size=spike_steps.shape[-2:], mode="bilinear", align_corners=False)
    spike_binary = (spike_steps > float(diff_threshold)).float()
    step_ids = torch.arange(steps, device=spike_steps.device, dtype=spike_steps.dtype).view(1, steps, 1, 1, 1)
    no_spike_fill = torch.full_like(step_ids, float(steps))
    earliest = torch.where(spike_binary > 0.0, step_ids, no_spike_fill).amin(dim=1) / max(1.0, float(steps - 1))

    losses = []
    for batch_idx in range(diff_map.shape[0]):
        changed_mask = (diff_map[batch_idx : batch_idx + 1] >= float(diff_threshold)).float()
        unchanged_mask = 1.0 - changed_mask
        changed_mean = (earliest[batch_idx : batch_idx + 1] * changed_mask).sum() / (changed_mask.sum() + 1e-6)
        unchanged_mean = (earliest[batch_idx : batch_idx + 1] * unchanged_mask).sum() / (unchanged_mask.sum() + 1e-6)
        changed_early_ratio = ((earliest[batch_idx : batch_idx + 1] <= float(early_threshold)).float() * changed_mask).sum() / (changed_mask.sum() + 1e-6)
        losses.append(F.relu(changed_mean - unchanged_mean + float(margin)))
        losses.append(F.relu(float(0.5) - changed_early_ratio))
    if not losses:
        return spike_steps.new_tensor(0.0)
    return torch.stack(losses).mean()


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def parse_sparse_layers(text: str):
    names = [item.strip() for item in text.split(",") if item.strip()]
    valid = {"conv1", "conv2", "conv3", "conv4", "conv5"}
    invalid = [name for name in names if name not in valid]
    if invalid:
        raise ValueError(f"Unsupported sparse layer names: {invalid}")
    return names


def load_matching_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    current = module.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    current.update(filtered)
    module.load_state_dict(current, strict=False)


def train_snn_sfm(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir, experiment_name = resolve_output_dir(args)
    print(f"[experiment] name={experiment_name}")
    print(f"[experiment] output_dir={output_dir}")

    train_dataset = KITTIOdometryTriplet(
        kitti_root=args.kitti_root,
        seqs=parse_seqs(args.train_seqs),
        resize=(args.height, args.width),
        time_steps=args.time_steps,
        spike_input=False,
        spike_encoding=args.input_encoding,
    )
    val_dataset = KITTIOdometryTriplet(
        kitti_root=args.kitti_root,
        seqs=parse_seqs(args.val_seqs),
        resize=(args.height, args.width),
        time_steps=args.time_steps,
        spike_input=False,
        spike_encoding=args.input_encoding,
    )
    train_seq_counts = summarize_seq_counts(train_dataset)
    val_seq_counts = summarize_seq_counts(val_dataset)
    print(f"[data] train_seq_counts={train_seq_counts}")
    print(f"[data] val_seq_counts={val_seq_counts}")

    train_sampler = build_train_sampler(train_dataset, args.train_sampling)
    print(f"[data] train_sampling={args.train_sampling}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    ann_encoder = SimpleEncoder(in_channels=3)
    ann_state = torch.load(args.ann_encoder_ckpt, map_location="cpu")
    ann_encoder.load_state_dict(ann_state)

    model = MonoDepthSNN_Spike(
        tau=args.tau,
        time_steps=args.time_steps,
        v_threshold=args.v_threshold,
        input_encoding=args.input_encoding,
        lif_output_mode=args.lif_output_mode,
        sparse_exec=args.sparse_exec,
        sparse_layers=parse_sparse_layers(args.sparse_layers),
        sparse_activity_threshold=args.sparse_activity_threshold,
        sparse_fallback_ratio=args.sparse_fallback_ratio,
        delta_anchor_weight=args.delta_anchor_weight,
        decoder_channel_scale=args.decoder_channel_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        pose_hidden_channels=args.pose_hidden_channels,
        pose_mlp_hidden=args.pose_mlp_hidden,
        pose_input_normalization=args.pose_input_normalization,
        hybrid_static_branch=args.hybrid_static_branch,
        hybrid_static_weight=args.hybrid_static_weight,
        hybrid_pose_diff=args.hybrid_pose_diff,
    ).to(device)
    model.init_from_ann_encoder(ann_encoder)
    if args.resume_snn_sfm_ckpt:
        resume_ckpt = torch.load(args.resume_snn_sfm_ckpt, map_location="cpu")
        load_matching_state_dict(model, resume_ckpt["model_state"])
    elif args.snn_depth_ckpt and os.path.exists(args.snn_depth_ckpt):
        checkpoint = torch.load(args.snn_depth_ckpt, map_location="cpu")
        load_matching_state_dict(model, checkpoint["model_state"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    ensure_dir(output_dir)
    writer = create_summary_writer(os.path.join(output_dir, "tb_snn_sfm"))
    history_csv = os.path.join(output_dir, "snn_sfm_history.csv")
    start_epoch = 1
    best_val = float("inf")
    best_epoch = -1
    best_row: Dict[str, float] = {}
    if args.resume_snn_sfm_ckpt:
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if "scaler_state" in resume_ckpt and use_amp:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        best_epoch = int(resume_ckpt.get("best_epoch", -1))
        best_val = float(resume_ckpt.get("best_val", float("inf")))
        best_row = dict(resume_ckpt.get("best_row", {}))
        print(f"[resume] ckpt={args.resume_snn_sfm_ckpt} start_epoch={start_epoch} best_epoch={best_epoch} best_val={best_val:.6f}")
    elif os.path.exists(history_csv):
        os.remove(history_csv)

    for epoch in range(start_epoch, args.num_epochs + 1):
        freeze_depth = epoch <= args.freeze_depth_epochs
        set_requires_grad(model.encoder, not freeze_depth)
        if getattr(model, "static_encoder", None) is not None:
            set_requires_grad(model.static_encoder, not freeze_depth)
        set_requires_grad(model.depth_decoder, not freeze_depth)
        set_requires_grad(model.pair_pose_net, True)
        set_requires_grad(model.pose_decoder, True)
        model.train()
        running = {
            "loss": 0.0,
            "photo": 0.0,
            "smooth": 0.0,
            "spike": 0.0,
            "pose_consistency": 0.0,
            "far_depth": 0.0,
            "change_rank": 0.0,
            "early_spike": 0.0,
            "mask": 0.0,
        }
        for batch in tqdm(train_loader, desc=f"SNN SfM epoch {epoch}/{args.num_epochs}"):
            img_prev = batch["img_prev"].to(device)
            img_t = batch["img_t"].to(device)
            img_next = batch["img_next"].to(device)
            intrinsics = batch["intrinsics"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                depth_t, _ = model(
                    img_t,
                    num_steps=args.time_steps,
                    input_encoding=args.input_encoding,
                    img_prev=img_prev,
                )
                pose_prev = model.predict_pose_pair(img_t, img_prev)
                pose_next = model.predict_pose_pair(img_t, img_next)
                pose_prev_inv = model.predict_pose_pair(img_prev, img_t)
                pose_next_inv = model.predict_pose_pair(img_next, img_t)

                warp_prev = warp_image(img_prev, depth_t, pose_prev, intrinsics)
                warp_next = warp_image(img_next, depth_t, pose_next, intrinsics)
                photo_loss, aux = compute_photometric_terms(img_t, [warp_prev, warp_next], [img_prev, img_next])
                smooth_loss = depth_smoothness_loss(depth_t, img_t)
                spike_loss = spike_consistency_loss(model)
                depth_far_loss = far_depth_penalty(
                    depth_t,
                    max_depth=args.max_depth,
                    start_ratio=args.far_depth_start_ratio,
                )
                temporal_tensors = model.get_temporal_response_tensors()
                change_rank_loss = change_response_ranking_loss(
                    diff_map=torch.abs(img_t - img_prev).mean(dim=1, keepdim=True),
                    response_map=temporal_tensors["conv1_spike_steps"].mean(dim=1),
                    high_quantile=args.change_high_quantile,
                    low_quantile=args.change_low_quantile,
                    margin=args.change_rank_margin,
                )
                early_spike_loss = earliest_spike_time_loss(
                    diff_map=torch.abs(img_t - img_prev).mean(dim=1, keepdim=True),
                    spike_steps=temporal_tensors["conv1_spike_steps"],
                    diff_threshold=args.early_diff_threshold,
                    early_threshold=args.early_time_threshold,
                    margin=args.early_spike_margin,
                )
                pose_cycle_loss = 0.5 * (
                    pose_consistency_loss(pose_prev, pose_prev_inv) +
                    pose_consistency_loss(pose_next, pose_next_inv)
                )
                loss = (
                    photo_loss +
                    args.lambda_smooth * smooth_loss +
                    args.lambda_spike * spike_loss +
                    args.lambda_far_depth * depth_far_loss +
                    args.lambda_change_rank * change_rank_loss +
                    args.lambda_early_spike * early_spike_loss +
                    args.lambda_pose_consistency * pose_cycle_loss
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running["loss"] += float(loss.item())
            running["photo"] += float(photo_loss.item())
            running["smooth"] += float(smooth_loss.item())
            running["spike"] += float(spike_loss.item())
            running["pose_consistency"] += float(pose_cycle_loss.item())
            running["far_depth"] += float(depth_far_loss.item())
            running["change_rank"] += float(change_rank_loss.item())
            running["early_spike"] += float(early_spike_loss.item())
            running["mask"] += aux["auto_mask_ratio"]

        train_row = {f"train_{k}": v / max(1, len(train_loader)) for k, v in running.items()}

        model.eval()
        val_running = {
            "loss": 0.0,
            "photo": 0.0,
            "smooth": 0.0,
            "spike": 0.0,
            "pose_consistency": 0.0,
            "far_depth": 0.0,
            "change_rank": 0.0,
            "early_spike": 0.0,
            "mask": 0.0,
        }
        with torch.no_grad():
            for batch in val_loader:
                img_prev = batch["img_prev"].to(device)
                img_t = batch["img_t"].to(device)
                img_next = batch["img_next"].to(device)
                intrinsics = batch["intrinsics"].to(device)

                with autocast(enabled=use_amp):
                    depth_t, _ = model(
                        img_t,
                        num_steps=args.time_steps,
                        input_encoding=args.input_encoding,
                        img_prev=img_prev,
                    )
                    pose_prev = model.predict_pose_pair(img_t, img_prev)
                    pose_next = model.predict_pose_pair(img_t, img_next)
                    pose_prev_inv = model.predict_pose_pair(img_prev, img_t)
                    pose_next_inv = model.predict_pose_pair(img_next, img_t)
                    warp_prev = warp_image(img_prev, depth_t, pose_prev, intrinsics)
                    warp_next = warp_image(img_next, depth_t, pose_next, intrinsics)
                    photo_loss, aux = compute_photometric_terms(img_t, [warp_prev, warp_next], [img_prev, img_next])
                    smooth_loss = depth_smoothness_loss(depth_t, img_t)
                    spike_loss = spike_consistency_loss(model)
                    depth_far_loss = far_depth_penalty(
                        depth_t,
                        max_depth=args.max_depth,
                        start_ratio=args.far_depth_start_ratio,
                    )
                    temporal_tensors = model.get_temporal_response_tensors()
                    change_rank_loss = change_response_ranking_loss(
                        diff_map=torch.abs(img_t - img_prev).mean(dim=1, keepdim=True),
                        response_map=temporal_tensors["conv1_spike_steps"].mean(dim=1),
                        high_quantile=args.change_high_quantile,
                        low_quantile=args.change_low_quantile,
                        margin=args.change_rank_margin,
                    )
                    early_spike_loss = earliest_spike_time_loss(
                        diff_map=torch.abs(img_t - img_prev).mean(dim=1, keepdim=True),
                        spike_steps=temporal_tensors["conv1_spike_steps"],
                        diff_threshold=args.early_diff_threshold,
                        early_threshold=args.early_time_threshold,
                        margin=args.early_spike_margin,
                    )
                    pose_cycle_loss = 0.5 * (
                        pose_consistency_loss(pose_prev, pose_prev_inv) +
                        pose_consistency_loss(pose_next, pose_next_inv)
                    )
                    loss = (
                        photo_loss +
                        args.lambda_smooth * smooth_loss +
                        args.lambda_spike * spike_loss +
                        args.lambda_far_depth * depth_far_loss +
                        args.lambda_change_rank * change_rank_loss +
                        args.lambda_early_spike * early_spike_loss +
                        args.lambda_pose_consistency * pose_cycle_loss
                    )

                val_running["loss"] += float(loss.item())
                val_running["photo"] += float(photo_loss.item())
                val_running["smooth"] += float(smooth_loss.item())
                val_running["spike"] += float(spike_loss.item())
                val_running["pose_consistency"] += float(pose_cycle_loss.item())
                val_running["far_depth"] += float(depth_far_loss.item())
                val_running["change_rank"] += float(change_rank_loss.item())
                val_running["early_spike"] += float(early_spike_loss.item())
                val_running["mask"] += aux["auto_mask_ratio"]

        val_row = {f"val_{k}": v / max(1, len(val_loader)) for k, v in val_running.items()}
        val_row["val_avg_spike_rate"] = model.get_spike_stats().get("avg_spike_rate", 0.0)
        row = {"epoch": epoch, **train_row, **val_row}
        row["val_avg_active_ratio"] = model.get_sparse_stats().get("avg_active_ratio", 1.0)
        row["val_avg_used_sparse"] = model.get_sparse_stats().get("avg_used_sparse", 0.0)
        write_csv_row(history_csv, row)
        writer.add_scalar("snn_sfm/train_loss", train_row["train_loss"], epoch)
        writer.add_scalar("snn_sfm/val_loss", val_row["val_loss"], epoch)
        writer.add_scalar("snn_sfm/val_avg_spike_rate", val_row["val_avg_spike_rate"], epoch)
        writer.add_scalar("snn_sfm/val_avg_active_ratio", row["val_avg_active_ratio"], epoch)
        writer.add_scalar("snn_sfm/val_avg_used_sparse", row["val_avg_used_sparse"], epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else {},
            "best_val": best_val,
            "best_epoch": best_epoch,
            "best_row": best_row,
            "config": vars(args),
            **row,
        }
        torch.save(checkpoint, os.path.join(output_dir, "latest_snn_sfm.pth"))
        if args.save_every_epoch:
            torch.save(checkpoint, os.path.join(output_dir, f"epoch_{epoch:02d}_snn_sfm.pth"))
        print(
            f"[snn_sfm] epoch={epoch} train_loss={train_row['train_loss']:.4f} "
            f"val_loss={val_row['val_loss']:.4f} auto_mask={val_row['val_mask']:.4f} "
            f"avg_spike={val_row['val_avg_spike_rate']:.4f} "
            f"active={row['val_avg_active_ratio']:.4f} "
            f"used_sparse={row['val_avg_used_sparse']:.4f} "
            f"freeze_depth={int(freeze_depth)}"
        )
        if val_row["val_loss"] < best_val:
            best_val = val_row["val_loss"]
            best_epoch = epoch
            best_row = dict(row)
            checkpoint["best_val"] = best_val
            checkpoint["best_epoch"] = best_epoch
            checkpoint["best_row"] = best_row
            torch.save(checkpoint, os.path.join(output_dir, "best_snn_sfm.pth"))
            metrics_payload = {
                "experiment_name": experiment_name,
                "epoch": epoch,
                "config": vars(args),
                **row,
            }
            write_json(os.path.join(output_dir, "best_snn_sfm_metrics.json"), metrics_payload)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    experiment_summary = {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "best_epoch": best_epoch,
        "config": vars(args),
        "best_metrics": best_row,
    }
    write_json(os.path.join(output_dir, "experiment_summary.json"), experiment_summary)
    append_experiment_index(
        os.path.join(args.output_dir, "experiment_index.csv"),
        {
            "experiment_name": experiment_name,
            "best_epoch": best_epoch,
            "train_seqs": args.train_seqs,
            "val_seqs": args.val_seqs,
            "time_steps": args.time_steps,
            "v_threshold": args.v_threshold,
            "tau": args.tau,
            "input_encoding": args.input_encoding,
            "lif_output_mode": args.lif_output_mode,
            "sparse_exec": int(args.sparse_exec),
            "sparse_activity_threshold": args.sparse_activity_threshold,
            "sparse_fallback_ratio": args.sparse_fallback_ratio,
            "sparse_layers": args.sparse_layers,
            "delta_anchor_weight": args.delta_anchor_weight,
            "decoder_channel_scale": args.decoder_channel_scale,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "lambda_far_depth": args.lambda_far_depth,
            "far_depth_start_ratio": args.far_depth_start_ratio,
            "lambda_change_rank": args.lambda_change_rank,
            "change_rank_margin": args.change_rank_margin,
            "change_high_quantile": args.change_high_quantile,
            "change_low_quantile": args.change_low_quantile,
            "lambda_early_spike": args.lambda_early_spike,
            "early_spike_margin": args.early_spike_margin,
            "early_diff_threshold": args.early_diff_threshold,
            "early_time_threshold": args.early_time_threshold,
            "pose_hidden_channels": args.pose_hidden_channels,
            "pose_mlp_hidden": args.pose_mlp_hidden,
            "pose_input_normalization": int(args.pose_input_normalization),
            "hybrid_static_branch": int(args.hybrid_static_branch),
            "hybrid_static_weight": args.hybrid_static_weight,
            "hybrid_pose_diff": int(args.hybrid_pose_diff),
            "freeze_depth_epochs": args.freeze_depth_epochs,
            "train_sampling": args.train_sampling,
            "lambda_pose_consistency": args.lambda_pose_consistency,
            "init_from_depth_ckpt": int(bool(args.snn_depth_ckpt)),
            "train_loss": best_row.get("train_loss", 0.0),
            "train_photo": best_row.get("train_photo", 0.0),
            "train_smooth": best_row.get("train_smooth", 0.0),
            "train_spike": best_row.get("train_spike", 0.0),
            "train_pose_consistency": best_row.get("train_pose_consistency", 0.0),
            "train_mask": best_row.get("train_mask", 0.0),
            "val_loss": best_row.get("val_loss", 0.0),
            "val_photo": best_row.get("val_photo", 0.0),
            "val_smooth": best_row.get("val_smooth", 0.0),
            "val_spike": best_row.get("val_spike", 0.0),
            "val_pose_consistency": best_row.get("val_pose_consistency", 0.0),
            "val_mask": best_row.get("val_mask", 0.0),
            "val_avg_spike_rate": best_row.get("val_avg_spike_rate", 0.0),
            "val_avg_active_ratio": best_row.get("val_avg_active_ratio", 0.0),
            "val_avg_used_sparse": best_row.get("val_avg_used_sparse", 0.0),
        },
    )


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train SNN SfM/VO front-end on KITTI odometry sequences.")
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--ann-encoder-ckpt", default="")
    parser.add_argument("--snn-depth-ckpt", default="")
    parser.add_argument("--resume-snn-sfm-ckpt", default="")
    parser.add_argument("--output-dir", default=str(script_dir / "outputs" / "snn_sfm"))
    parser.add_argument("--auto-experiment-dir", action="store_true")
    parser.add_argument("--train-seqs", default="00,01,02,03,04,05,06,07,08")
    parser.add_argument("--val-seqs", default="09")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-spike", type=float, default=0.01)
    parser.add_argument("--lambda-pose-consistency", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--v-threshold", type=float, default=0.35)
    parser.add_argument("--input-encoding", choices=["rate", "analog", "latency", "delta_latency", "delta_latency_anchor"], default="delta_latency")
    parser.add_argument("--lif-output-mode", choices=["mixed", "spike", "membrane"], default="mixed")
    parser.add_argument("--train-sampling", choices=["uniform", "seq_balanced"], default="seq_balanced")
    parser.add_argument("--sparse-exec", action="store_true")
    parser.add_argument("--sparse-layers", default="conv1,conv2")
    parser.add_argument("--sparse-activity-threshold", type=float, default=0.4)
    parser.add_argument("--sparse-fallback-ratio", type=float, default=0.6)
    parser.add_argument("--delta-anchor-weight", type=float, default=0.2)
    parser.add_argument("--decoder-channel-scale", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--lambda-far-depth", type=float, default=0.0)
    parser.add_argument("--far-depth-start-ratio", type=float, default=0.9)
    parser.add_argument("--lambda-change-rank", type=float, default=0.0)
    parser.add_argument("--change-rank-margin", type=float, default=0.05)
    parser.add_argument("--change-high-quantile", type=float, default=0.75)
    parser.add_argument("--change-low-quantile", type=float, default=0.25)
    parser.add_argument("--lambda-early-spike", type=float, default=0.0)
    parser.add_argument("--early-spike-margin", type=float, default=0.05)
    parser.add_argument("--early-diff-threshold", type=float, default=0.05)
    parser.add_argument("--early-time-threshold", type=float, default=0.5)
    parser.add_argument("--pose-hidden-channels", type=int, default=256)
    parser.add_argument("--pose-mlp-hidden", type=int, default=128)
    parser.add_argument("--pose-input-normalization", action="store_true")
    parser.add_argument("--hybrid-static-branch", action="store_true")
    parser.add_argument("--hybrid-static-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-pose-diff", action="store_true")
    parser.add_argument("--freeze-depth-epochs", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--device", default="")
    parser.set_defaults(auto_experiment_dir=True)
    return parser.parse_args()


if __name__ == "__main__":
    train_snn_sfm(parse_args())
