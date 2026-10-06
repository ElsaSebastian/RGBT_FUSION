"""
CRAFTv2 end-to-end smoke test on real data.

Usage:
    python -m craft_v2.smoke_test_v2 \
        --dataset-root /path/to/Anti-UAV-RGBT \
        --frames-root  /path/to/data/frames

Runs 6 checks in order and reports PASSED or the exact failure.
"""

import argparse
import json
import os
import sys
import tempfile

import torch

from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.pipeline import CRAFTv2
from craft_v2.loss import CRAFTv2Loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split_json(dataset_root, split):
    candidate = os.path.join(dataset_root, f"{split}.json")
    if os.path.isfile(candidate):
        return candidate
    split_dir = os.path.join(dataset_root, split)
    seqs = {s: [] for s in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, s))}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(seqs, tmp); tmp.close()
    return tmp.name


def _collate(batch):
    keys_tensor = ["rgb", "ir", "box_vis", "box_ir", "exist"]
    keys_scalar = ["scale", "pad_x", "pad_y", "orig_w", "orig_h",
                   "seq_name", "frame_idx"]
    out = {k: torch.stack([s[k] for s in batch]) for k in keys_tensor}
    for k in keys_scalar:
        out[k] = [s[k] for s in batch]
    return out


def _step(label):
    print(f"\n{'─'*50}")
    print(f"  CHECK {label}")
    print(f"{'─'*50}")


def _ok(msg):
    print(f"  ✓  {msg}")


def _fail(msg):
    print(f"\n  ✗  FAILED: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def run(dataset_root, frames_root, device_str="auto"):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    ) if device_str == "auto" else torch.device(device_str)

    print(f"Device        : {device}")
    print(f"dataset_root  : {dataset_root}")
    print(f"frames_root   : {frames_root}")

    # -----------------------------------------------------------------------
    # CHECK 1 — Load 1 real batch from train split
    # -----------------------------------------------------------------------
    _step("1/6 — Load 1 real batch (train, batch_size=4)")

    train_json = _make_split_json(dataset_root, "train")
    train_ds = AntiUAVFrameDatasetV2(
        dataset_root=dataset_root,
        split="train",
        split_json=train_json,
        frames_root=frames_root if frames_root else None,
    )
    if len(train_ds) == 0:
        _fail("Train dataset is empty — check dataset_root and frames_root")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=_collate,
    )
    batch = next(iter(loader))

    rgb    = batch["rgb"].to(device)
    ir     = batch["ir"].to(device)
    box_vis = batch["box_vis"].to(device)
    box_ir  = batch["box_ir"].to(device)
    exist   = batch["exist"].to(device)

    n_exist = int(exist.sum().item())
    _ok(f"rgb shape   : {tuple(rgb.shape)}")
    _ok(f"ir  shape   : {tuple(ir.shape)}")
    _ok(f"exist counts: {n_exist} positive / {len(exist) - n_exist} negative in batch")
    _ok(f"box_vis[0]  : {[round(v,2) for v in box_vis[0].tolist()]}")
    _ok(f"scale       : {batch['scale'][0]:.4f}   pad_y: {batch['pad_y'][0]:.1f}")

    if rgb.shape != torch.Size([4, 3, 320, 320]):
        _fail(f"rgb shape expected [4,3,320,320], got {tuple(rgb.shape)}")
    if ir.shape != torch.Size([4, 1, 320, 320]):
        _fail(f"ir shape expected [4,1,320,320], got {tuple(ir.shape)}")

    # -----------------------------------------------------------------------
    # CHECK 2 — Forward pass, verify all 5 stage shapes
    # -----------------------------------------------------------------------
    _step("2/6 — Forward pass through CRAFTv2 (verbose shapes)")

    model = CRAFTv2(ch=64).to(device)
    model.train()

    out = model(rgb, ir, verbose=True)

    expected_shapes = {
        "boxes":         (4, 32000, 4),
        "scores":        (4, 32000, 1),
        "f_rgb_aligned": (4, 64, 80, 80),
        "f_ir_aligned":  (4, 64, 80, 80),
        "f_fused":       (4, 64, 80, 80),
        "mask_rgb":      (4, 9, 80, 80),
        "mask_ir":       (4, 9, 80, 80),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(out[key].shape)
        if actual != expected:
            _fail(f"{key}: expected {expected}, got {actual}")
        _ok(f"{key:16s}: {actual}")

    # -----------------------------------------------------------------------
    # CHECK 3 — Loss computation
    # -----------------------------------------------------------------------
    _step("3/6 — Compute loss on the same batch")

    criterion = CRAFTv2Loss(w_ag=1.0, w_al=1.0, w_m=1.0, w_f=1.0, w_det=2.0)
    gt = {"box_vis": box_vis, "box_ir": box_ir, "exist": exist}
    losses = criterion(out, gt)

    for k, v in losses.items():
        print(f"    {k:8s}: {v.item():.6f}")

    if not losses["loss"].isfinite():
        _fail(f"total loss is not finite: {losses['loss'].item()}")
    if losses["loss"].item() <= 0:
        _fail(f"total loss is not > 0: {losses['loss'].item()}")
    _ok(f"total loss finite and > 0: {losses['loss'].item():.6f}")

    # -----------------------------------------------------------------------
    # CHECK 4 — Backward pass, check gradients
    # -----------------------------------------------------------------------
    _step("4/6 — Backward pass — check for NaN gradients")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    losses["loss"].backward()

    max_norm = 0.0
    nan_params = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not p.grad.isfinite().all():
            nan_params.append(name)
        max_norm = max(max_norm, p.grad.abs().max().item())

    if nan_params:
        _fail(f"NaN/Inf gradients in: {nan_params[:5]}")

    _ok(f"No NaN gradients")
    _ok(f"Max gradient magnitude: {max_norm:.6f}")

    if max_norm == 0.0:
        _fail("All gradients are exactly zero — something is wrong with the graph")

    # -----------------------------------------------------------------------
    # CHECK 5 — Optimizer step, confirm loss changes
    # -----------------------------------------------------------------------
    _step("5/6 — Optimizer step — confirm gradients are flowing")

    loss_before = losses["loss"].item()
    optimizer.step()

    # Second forward pass with the same batch
    out2 = model(rgb, ir)
    losses2 = criterion(out2, gt)
    loss_after = losses2["loss"].item()

    _ok(f"loss before step : {loss_before:.6f}")
    _ok(f"loss after step  : {loss_after:.6f}")

    if not losses2["loss"].isfinite():
        _fail(f"loss after optimizer step is not finite: {loss_after}")
    if loss_before == loss_after:
        _fail("Loss did not change after optimizer step — gradients may not be flowing")
    _ok("Loss changed after optimizer step — gradients are flowing")

    # -----------------------------------------------------------------------
    # CHECK 6 — Eval pipeline on 100 frames from test split
    # -----------------------------------------------------------------------
    _step("6/6 — Eval pipeline on 100 test frames")

    test_json = _make_split_json(dataset_root, "test")
    test_ds = AntiUAVFrameDatasetV2(
        dataset_root=dataset_root,
        split="test",
        split_json=test_json,
        frames_root=frames_root if frames_root else None,
    )
    if len(test_ds) == 0:
        _fail("Test dataset is empty")

    # Take first 100 frames as a subset
    subset = torch.utils.data.Subset(test_ds, list(range(min(100, len(test_ds)))))
    test_loader = torch.utils.data.DataLoader(
        subset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=_collate,
    )

    model.eval()
    frames_processed = 0
    n_predictions    = 0
    errors           = []

    with torch.no_grad():
        for batch_idx, tbatch in enumerate(test_loader):
            try:
                trgb   = tbatch["rgb"].to(device)
                tir    = tbatch["ir"].to(device)
                tout   = model(trgb, tir)

                # Sigmoid scores, count above threshold
                conf = torch.sigmoid(tout["scores"])   # [B,32000,1]
                n_predictions += int((conf > 0.001).sum().item())
                frames_processed += trgb.shape[0]

                # Verify output shapes are intact
                if tuple(tout["boxes"].shape) != (trgb.shape[0], 32000, 4):
                    errors.append(f"batch {batch_idx}: unexpected boxes shape {tout['boxes'].shape}")
                if tuple(tout["scores"].shape) != (trgb.shape[0], 32000, 1):
                    errors.append(f"batch {batch_idx}: unexpected scores shape {tout['scores'].shape}")

                # Verify inverse-letterbox round-trip is sane
                for i in range(trgb.shape[0]):
                    s  = float(tbatch["scale"][i])
                    px = float(tbatch["pad_x"][i])
                    py = float(tbatch["pad_y"][i])
                    bv = tbatch["box_vis"][i].tolist()
                    if tbatch["exist"][i].item() == 1:
                        x_orig = (bv[0] - px) / s
                        y_orig = (bv[1] - py) / s
                        if x_orig < 0 or y_orig < 0:
                            errors.append(
                                f"seq={tbatch['seq_name'][i]} frame={tbatch['frame_idx'][i]}: "
                                f"inverse-lb gives negative coords ({x_orig:.1f},{y_orig:.1f})"
                            )

            except Exception as e:
                errors.append(f"batch {batch_idx}: {type(e).__name__}: {e}")

    if errors:
        _fail(f"{len(errors)} error(s) during eval:\n    " + "\n    ".join(errors[:5]))

    _ok(f"Frames processed     : {frames_processed}")
    _ok(f"Predictions (conf>0.001): {n_predictions:,}")
    _ok(f"Predictions per frame: {n_predictions / max(frames_processed,1):.1f}")
    _ok("No errors during eval pass")

    # -----------------------------------------------------------------------
    # All checks passed
    # -----------------------------------------------------------------------
    print()
    print("═" * 50)
    print("  CRAFT-v2 smoke test PASSED")
    print("  Ready for full training run.")
    print("═" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CRAFTv2 end-to-end smoke test")
    p.add_argument("--dataset-root", required=True,
                   help="Root containing sequence folders + JSONs")
    p.add_argument("--frames-root",  default="",
                   help="Root containing extracted JPEGs (defaults to dataset-root)")
    p.add_argument("--device",       default="auto",
                   help="cuda / cpu / auto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        dataset_root=args.dataset_root,
        frames_root=args.frames_root if args.frames_root else None,
        device_str=args.device,
    )
