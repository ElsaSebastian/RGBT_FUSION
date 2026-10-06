"""
CRAFTv2 training script.

Usage:
    python -m craft_v2.train \
        --dataset-root /path/to/Anti-UAV-RGBT \
        --frames-root  /path/to/data/frames \
        --output-dir   craft_v2/runs/exp01

    # Resume from checkpoint:
    python -m craft_v2.train ... --resume craft_v2/runs/exp01/latest.pt
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset

from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.pipeline import CRAFTv2
from craft_v2.loss import CRAFTv2Loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paths
    dataset_root: str = ""
    frames_root:  str = ""
    output_dir:   str = "craft_v2/runs"
    train_split:  str = "train"   # e.g. "mini_train" for the 1-frame-per-seq subset

    # Data
    batch_size:          int  = 16
    num_workers:         int  = 4
    pin_memory:          bool = True
    persistent_workers:  bool = True

    # Training
    epochs:         int   = 150
    lr:             float = 1e-4
    weight_decay:   float = 1e-4
    warmup_epochs:  int   = 5
    patience:       int   = 30

    # Loss weights  (l_ag/l_al/l_f/l_m discarded)
    w_ag:  float = 0.0
    w_al:  float = 0.0
    w_m:   float = 0.0
    w_f:   float = 0.0
    w_det: float = 5.0

    # Augmentation
    hflip_prob: float = 0.5

    # Data loading
    preload:  bool = False   # load all frames into RAM at startup (~46 GB)
    modality:     str  = "both"  # "both" | "rgb" | "ir" — unimodal ablation
    use_convnext: bool = False

    # Logging
    log_every: int = 50


# ---------------------------------------------------------------------------
# Augmentation — consistent horizontal flip for both modalities + boxes
# ---------------------------------------------------------------------------

def apply_hflip(rgb, ir, box_vis, box_ir, exist, prob):
    """
    Random horizontal flip applied identically to RGB, IR, and box coords.
    Box x-coord is mirrored: x_new = (img_width - x - w) in letterbox space.
    img_width = 320 (letterbox target).
    """
    if random.random() >= prob:
        return rgb, ir, box_vis, box_ir

    rgb = torch.flip(rgb, dims=[-1])
    ir  = torch.flip(ir,  dims=[-1])

    IMG_W = 320.0

    def flip_box(box):
        # box: [4] x,y,w,h — only x changes
        flipped = box.clone()
        flipped[0] = IMG_W - box[0] - box[2]
        return flipped

    # Only flip box coords when exist=1; zeros stay zero
    if exist.item() == 1:
        box_vis = flip_box(box_vis)
        box_ir  = flip_box(box_ir)

    return rgb, ir, box_vis, box_ir


def collate_with_augment(batch, hflip_prob):
    """Custom collate that applies per-sample augmentation before batching."""
    rgb_list, ir_list, bv_list, bi_list, exist_list = [], [], [], [], []
    rgb_mask_list, ir_mask_list = [], []
    meta_keys = ["scale", "pad_x", "pad_y", "orig_w", "orig_h", "seq_name", "frame_idx"]
    meta = {k: [] for k in meta_keys}

    for sample in batch:
        rgb, ir, bv, bi = apply_hflip(
            sample["rgb"], sample["ir"],
            sample["box_vis"], sample["box_ir"],
            sample["exist"], hflip_prob,
        )
        rgb_list.append(rgb)
        ir_list.append(ir)
        bv_list.append(bv)
        bi_list.append(bi)
        exist_list.append(sample["exist"])
        rgb_mask_list.append(sample["rgb_mask_80"])
        ir_mask_list.append(sample["ir_mask_80"])
        for k in meta_keys:
            meta[k].append(sample[k])

    return {
        "rgb":         torch.stack(rgb_list),
        "ir":          torch.stack(ir_list),
        "rgb_mask_80": torch.stack(rgb_mask_list),
        "ir_mask_80":  torch.stack(ir_mask_list),
        "box_vis":     torch.stack(bv_list),
        "box_ir":      torch.stack(bi_list),
        "exist":       torch.stack(exist_list),
        **meta,
    }


# ---------------------------------------------------------------------------
# Split JSON helpers
# ---------------------------------------------------------------------------

def _make_split_json(dataset_root: str, split: str) -> str:
    """
    Return path to <split>.json if it exists alongside the split dir,
    otherwise build a minimal one from directory listing and write it
    to output_dir.  Returns the path.
    """
    candidate = os.path.join(dataset_root, f"{split}.json")
    if os.path.isfile(candidate):
        return candidate

    split_dir = os.path.join(dataset_root, split)
    seqs = {s: [] for s in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, s))}
    path = os.path.join(dataset_root, f"{split}.json")
    with open(path, "w") as f:
        json.dump(seqs, f)
    return path


# ---------------------------------------------------------------------------
# LR scheduler — linear warmup then cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg: TrainConfig):
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            # Linear warmup from lr/10 to lr
            return 0.1 + 0.9 * (epoch / max(cfg.warmup_epochs, 1))
        # Cosine decay from lr to lr/100
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return (1/100) + (1 - 1/100) * cosine   # scale relative to base lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, cfg):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss":        val_loss,
        "config":          asdict(cfg),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["val_loss"]


# ---------------------------------------------------------------------------
# One epoch — train or val
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch,
              total_epochs, n_total_steps, is_train):
    model.train() if is_train else model.eval()
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sum_losses = {k: 0.0 for k in ["loss", "l_ag", "l_al", "l_m", "l_f", "l_det"]}
    sum_offset_mag = 0.0
    n_batches = 0
    t0 = time.time()

    with ctx:
        for step, batch in enumerate(loader):
            rgb    = batch["rgb"].to(device, non_blocking=True)
            ir     = batch["ir"].to(device,  non_blocking=True)
            box_vis = batch["box_vis"].to(device, non_blocking=True)
            box_ir  = batch["box_ir"].to(device,  non_blocking=True)
            exist   = batch["exist"].to(device,   non_blocking=True)
            rgb_mask = batch["rgb_mask_80"].to(device, non_blocking=True)
            ir_mask  = batch["ir_mask_80"].to(device,  non_blocking=True)
            if cfg.modality == "rgb":
                ir      = torch.zeros_like(ir)
                ir_mask = torch.zeros_like(ir_mask)
            elif cfg.modality == "ir":
                rgb      = torch.zeros_like(rgb)
                rgb_mask = torch.zeros_like(rgb_mask)

            amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with autocast(device_type=device, dtype=amp_dtype,
                          enabled=device == "cuda"):
                model_out = model(rgb, ir, rgb_mask=rgb_mask, ir_mask=ir_mask)
                gt = {"box_vis": box_vis, "box_ir": box_ir, "exist": exist}
                losses = criterion(model_out, gt)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

            for k in sum_losses:
                sum_losses[k] += losses[k].item()
            sum_offset_mag += model_out["offset_mag"]
            n_batches += 1

            if is_train and (step + 1) % cfg.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                global_step = (epoch - 1) * n_total_steps + step + 1
                print(
                    f"Epoch {epoch:03d}/{total_epochs:03d} | "
                    f"Step {global_step:05d}/{(total_epochs * n_total_steps):05d} | "
                    f"loss={losses['loss'].item():.2f} "
                    f"l_ag={losses['l_ag'].item():.2f} "
                    f"l_al={losses['l_al'].item():.2f} "
                    f"l_m={losses['l_m'].item():.3f} "
                    f"l_f={losses['l_f'].item():.2f} "
                    f"l_det={losses['l_det'].item():.2f} | "
                    f"lr={lr_now:.2e}"
                )

    avg = {k: v / max(n_batches, 1) for k, v in sum_losses.items()}
    avg["offset_mag"] = sum_offset_mag / max(n_batches, 1)
    elapsed = time.time() - t0
    return avg, elapsed


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, resume: str = None):
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Output dir : {cfg.output_dir}")

    # --- Split JSONs ---
    val_split  = "val" if cfg.train_split == "train" else f"mini_val"
    train_json = _make_split_json(cfg.dataset_root, cfg.train_split)
    val_json   = _make_split_json(cfg.dataset_root, val_split)

    # --- Datasets ---
    train_ds = AntiUAVFrameDatasetV2(
        dataset_root=cfg.dataset_root,
        split=cfg.train_split,
        split_json=train_json,
        frames_root=cfg.frames_root if cfg.frames_root else None,
        preload=cfg.preload,
    )
    val_ds = AntiUAVFrameDatasetV2(
        dataset_root=cfg.dataset_root,
        split=val_split,
        split_json=val_json,
        frames_root=cfg.frames_root if cfg.frames_root else None,
        preload=cfg.preload,
    )
    print(f"Train frames: {len(train_ds):,}  |  Val frames: {len(val_ds):,}")

    # --- Loaders ---
    # collate_fn is a closure carrying hflip_prob
    def train_collate(batch):
        return collate_with_augment(batch, cfg.hflip_prob)

    def val_collate(batch):
        return collate_with_augment(batch, hflip_prob=0.0)   # no aug at val

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        collate_fn=train_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        collate_fn=val_collate,
        drop_last=False,
    )

    # --- Model + loss ---
    model = CRAFTv2(ch=64, use_convnext=cfg.use_convnext).to(device)
    criterion = CRAFTv2Loss(
        w_ag=cfg.w_ag, w_al=cfg.w_al, w_m=cfg.w_m,
        w_f=cfg.w_f,   w_det=cfg.w_det,
    )

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg)
    try:
        scaler = GradScaler(device="cuda", enabled=device == "cuda")
    except TypeError:
        scaler = GradScaler(enabled=device == "cuda")

    # --- Resume ---
    start_epoch  = 1
    best_val_loss = math.inf
    best_epoch    = 0
    no_improve    = 0

    if resume:
        print(f"Resuming from: {resume}")
        start_epoch, best_val_loss = load_checkpoint(
            resume, model, optimizer, scheduler, device
        )
        start_epoch += 1
        print(f"  Resuming at epoch {start_epoch}, best_val={best_val_loss:.4f}")

    # --- CSV log ---
    csv_path = os.path.join(cfg.output_dir, "metrics.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_cols  = ["epoch", "train_loss", "val_loss",
                 "l_ag", "l_al", "l_m", "l_f", "l_det", "offset_mag", "lr"]
    writer = csv.DictWriter(csv_file, fieldnames=csv_cols)
    if os.path.getsize(csv_path) == 0:
        writer.writeheader()

    n_train_steps = len(train_loader)

    # --- Epoch loop ---
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Re-seed for per-epoch diversity in shuffle
        torch.manual_seed(epoch)
        random.seed(epoch)

        # Schedule L_m off after 30% of epochs — masks are trained by then
        if epoch > int(0.3 * cfg.epochs):
            cfg.w_m = 0.0
            criterion.w_m = 0.0

        train_avg, train_t = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cfg, epoch, cfg.epochs, n_train_steps, is_train=True,
        )

        # Save immediately after train — before val so a val crash loses nothing
        save_checkpoint(
            os.path.join(cfg.output_dir, "latest.pt"),
            epoch, model, optimizer, scheduler, best_val_loss, cfg,
        )

        val_avg, val_t = run_epoch(
            model, val_loader, criterion, None, scaler, device,
            cfg, epoch, cfg.epochs, n_train_steps, is_train=False,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # --- Epoch summary ---
        improved = val_avg["loss"] < best_val_loss
        if improved:
            best_val_loss = val_avg["loss"]
            best_epoch    = epoch
            no_improve    = 0
            save_checkpoint(
                os.path.join(cfg.output_dir, "best.pt"),
                epoch, model, optimizer, scheduler, best_val_loss, cfg,
            )
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_avg['loss']:.4f} val={val_avg['loss']:.4f} | "
            f"best_val={best_val_loss:.4f} (epoch {best_epoch:03d}) | "
            f"lr={lr_now:.2e}"
            + (" ✓" if improved else "")
        )

        # CSV row uses train val-split averages for individual terms
        writer.writerow({
            "epoch":      epoch,
            "train_loss": round(train_avg["loss"], 6),
            "val_loss":   round(val_avg["loss"],   6),
            "l_ag":       round(val_avg["l_ag"],   6),
            "l_al":       round(val_avg["l_al"],   6),
            "l_m":        round(val_avg["l_m"],    6),
            "l_f":        round(val_avg["l_f"],    6),
            "l_det":      round(val_avg["l_det"],       6),
            "offset_mag": round(train_avg["offset_mag"], 6),
            "lr":         round(lr_now, 8),
        })
        csv_file.flush()

        # Early stopping
        if no_improve >= cfg.patience:
            print(f"Early stopping: no val improvement for {cfg.patience} epochs.")
            break

    csv_file.close()
    print(f"Training complete. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
    print(f"Checkpoints in: {cfg.output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train CRAFTv2")
    p.add_argument("--dataset-root",  required=True)
    p.add_argument("--frames-root",   default="")
    p.add_argument("--output-dir",    default="craft_v2/runs")
    p.add_argument("--resume",        default=None)
    # Allow any TrainConfig field to be overridden from CLI
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch-size",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--weight-decay",  type=float, default=None)
    p.add_argument("--warmup-epochs", type=int,   default=None)
    p.add_argument("--patience",      type=int,   default=None)
    p.add_argument("--num-workers",   type=int,   default=None)
    p.add_argument("--w-ag",          type=float, default=None)
    p.add_argument("--w-al",          type=float, default=None)
    p.add_argument("--w-m",           type=float, default=None)
    p.add_argument("--w-f",           type=float, default=None)
    p.add_argument("--w-det",         type=float, default=None)
    p.add_argument("--split",         type=str,   default=None,
                   help="Train split name (default: train). Use 'mini_train' for 1-frame-per-seq run.")
    p.add_argument("--log-every",     type=int,   default=None)
    p.add_argument("--preload",       action="store_true",
                   help="Load all frames into RAM at startup (~46 GB)")
    p.add_argument("--modality",      type=str, default="both",
                   choices=["both", "rgb", "ir"],
                   help="Unimodal ablation: zero out the unused modality")
    p.add_argument("--use-convnext",  action="store_true",
                   help="Use ConvNeXt-Tiny backbone (run10+)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        frames_root=args.frames_root,
        output_dir=args.output_dir,
    )
    # Apply any CLI overrides
    overrides = {
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "weight_decay":  args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience":      args.patience,
        "num_workers":   args.num_workers,
        "w_ag":          args.w_ag,
        "w_al":          args.w_al,
        "w_m":           args.w_m,
        "w_f":           args.w_f,
        "w_det":         args.w_det,
        "log_every":     args.log_every,
        "train_split":   args.split,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.preload:
        cfg.preload = True
    cfg.modality     = args.modality
    cfg.use_convnext = args.use_convnext

    train(cfg, resume=args.resume)
