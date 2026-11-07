"""Training script for TransUNet on MyoPS380 (no-val protocol).

Protocol: myops380_train304_test76_last_v1
Budget: 300 epochs, physical batch 16, 37,200 total updates.
Objective: 0.5 * CrossEntropyLoss + 0.5 * DiceLoss
Optimizer: AdamW (lr=0.0003333, wd=0.0001, betas=(0.9, 0.999))
Scheduler: Polynomial decay (power=0.9, zero-based offset)
Checkpoint: <run_dir>/checkpoints/transunet_seed<seed>.pt (atomic save per epoch)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_myops import MyopsDataset, RandomGenerator
from networks.vit_seg_modeling import VisionTransformer, CONFIGS
from utils import DiceLoss
from utils.optimization import (
    OPTIMIZATION_PROTOCOL,
    build_optimizer,
    build_scheduler,
    validate_optimization_checkpoint,
)

logger = logging.getLogger("TransUNet_Train")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def resolve_amp(mode: str, device: torch.device) -> torch.dtype | None:
    if mode == "none" or (mode == "auto" and device.type == "cpu"):
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision is only supported on CUDA.")
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 is not supported on this CUDA device.")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError(f"Unknown AMP mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransUNet on MyoPS380 without validation")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("LOCAL_DATASET", "E:/STUDY/DATASET/MyoPS380/Processed_data"),
        help="Path to preprocessed MyoPS380 dataset root containing bSSFP, LGE, T2w",
    )
    parser.add_argument(
        "--list-dir",
        type=str,
        default="lists/lists_MyoPS380",
        help="Path to split manifests folder containing train.txt and test_vol.txt",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to Google ViT pretrained checkpoint (R50-ViT-B_16.npz)",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="outputs/runs",
        help="Root directory for saving checkpoints and training runs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run directory name (default: transunet_seed<seed>)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--batch-size", type=int, default=16, help="Physical batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.0003333, help="Base learning rate (default: 0.0003333)")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="AdamW weight decay (default: 0.0001)")
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300)")
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "none", "fp16", "bf16"], help="Mixed precision mode")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker threads (default: 4)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine run directory and checkpoint path
    run_name = args.run_name or f"transunet_seed{args.seed}"
    run_dir = Path(args.run_root) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"transunet_seed{args.seed}.pt"

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
        ],
    )

    logger.info("================================================================================")
    logger.info("TransUNet MyoPS380 Training - Protocol: myops380_train304_test76_last_v1")
    logger.info("Run directory: %s", run_dir)
    logger.info("================================================================================")

    # Seed
    seed_everything(args.seed, deterministic=True)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)

    # Architecture configuration: R50-ViT-B_16 with grid (8, 8) and n_classes = 4
    config = CONFIGS["R50-ViT-B_16"]
    config.patches.grid = (8, 8)
    config.n_classes = 4
    model = VisionTransformer(config, img_size=128, num_classes=4, zero_head=False, vis=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "INFO | Device %s | AMP %s | parameters %s | train/val slices 1972/0 | effective batch <= %d",
        device.type,
        str(amp_dtype) if amp_dtype else "fp32",
        f"{total_params:,}",
        args.batch_size,
    )

    # Save run configuration
    config_dict = {
        "model_id": "transunet",
        "architecture": "R50-ViT-B_16",
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "n_classes": 4,
        "img_size": 128,
        "patches_grid": [8, 8],
        "training_protocol": {
            "id": "myops380_train304_test76_last_v1",
            "epochs": args.epochs,
            "physical_batch": args.batch_size,
            "accum_steps": 1,
            "no_val": True,
        },
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "base_lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": device.type,
        "amp": str(amp_dtype) if amp_dtype else "none",
    }
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Load pretrained weights if provided and not resuming
    if args.pretrained and not args.resume:
        logger.info("Loading pretrained ViT weights from: %s", args.pretrained)
        weights = np.load(args.pretrained)
        model.load_from(weights)
        logger.info("Pretrained weights loaded successfully.")

    model.to(device)

    # Dataset & Dataloader
    transform = RandomGenerator(output_size=(128, 128))
    train_dataset = MyopsDataset(
        data_root=args.data_root,
        list_dir=args.list_dir,
        split="train",
        transform=transform,
        label_order="legacy",
    )
    logger.info("Loaded train dataset: %d slices", len(train_dataset))

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    batches_per_epoch = len(train_loader)
    total_planned_updates = args.epochs * batches_per_epoch  # 300 * 124 = 37,200
    logger.info("Batches per epoch: %d | Total planned updates: %d", batches_per_epoch, total_planned_updates)

    # Loss, Optimizer, Scheduler
    ce_loss_fn = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss(n_classes=4)

    optimizer = build_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_updates=total_planned_updates)

    use_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 1
    global_step = 0
    total_updates = 0

    # Resume handling
    resume_file = args.resume or (ckpt_path if ckpt_path.is_file() else None)
    if resume_file and Path(resume_file).is_file():
        logger.info("Resuming training from checkpoint: %s", resume_file)
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        validate_optimization_checkpoint(ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        total_updates = ckpt.get("total_updates", 0)

        # Restore RNG states
        if "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and "cuda" in rng:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["random"])

        logger.info("Resumed from epoch %d (step %d, updates %d)", ckpt["epoch"], global_step, total_updates)

        if ckpt.get("training_complete", False):
            logger.info("Training was already marked complete at epoch %d. Nothing to run.", ckpt["epoch"])
            return

    # Training loop
    logger.info("Starting training loop from epoch %d to %d...", start_epoch, args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0

        for batch_idx, sampled_batch in enumerate(train_loader):
            images = sampled_batch["image"].to(device, non_blocking=True)
            labels = sampled_batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            if device.type == "cuda" and amp_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(images)
                    loss_ce = ce_loss_fn(logits, labels)
                    loss_dice = dice_loss_fn(logits, labels, softmax=True)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                logits = model(images)
                loss_ce = ce_loss_fn(logits, labels)
                loss_dice = dice_loss_fn(logits, labels, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                loss.backward()
                optimizer.step()

            # Polynomial LR step after every successful optimizer update
            scheduler.step()
            total_updates += 1
            global_step += 1

            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_dice += loss_dice.item()

        epoch_duration = time.time() - epoch_start_time
        avg_loss = running_loss / batches_per_epoch
        avg_ce = running_ce / batches_per_epoch
        avg_dice = running_dice / batches_per_epoch
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%3d | Loss: %.4f (CE: %.4f, Dice: %.4f) | LR: %.7f | Time: %.1fs | Updates: %d/%d",
            epoch,
            args.epochs,
            avg_loss,
            avg_ce,
            avg_dice,
            current_lr,
            epoch_duration,
            total_updates,
            total_planned_updates,
        )

        # Atomic checkpoint save
        is_final_epoch = (epoch == args.epochs)
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        checkpoint_data = {
            "epoch": epoch,
            "planned_epochs": args.epochs,
            "completed_epochs": epoch,
            "training_complete": is_final_epoch,
            "global_step": global_step,
            "total_updates": total_updates,
            "total_planned_updates": total_planned_updates,
            "model_id": "transunet",
            "parameters_count": total_params,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_scaler else None,
            "rng_state": rng_state,
            "training_protocol": {
                "id": "myops380_train304_test76_last_v1",
                "cohort": "myops380",
                "train_patients": 304,
                "train_slices": 1972,
                "test_patients": 76,
                "epochs": args.epochs,
            },
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "loss_metric": {
                "loss_total": avg_loss,
                "loss_ce": avg_ce,
                "loss_dice": avg_dice,
            },
        }

        # Safe atomic save: write to tmp file first then rename
        tmp_ckpt = ckpt_path.with_suffix(".tmp")
        torch.save(checkpoint_data, tmp_ckpt)
        tmp_ckpt.replace(ckpt_path)

    logger.info("================================================================================")
    logger.info("Training complete! 300 epochs finished. Final checkpoint saved at: %s", ckpt_path)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
