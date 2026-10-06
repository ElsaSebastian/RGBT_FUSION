"""
Test B — Offset vector visualization.

dp_rgb / dp_ir are deformable conv offsets, shape [1, 18, 80, 80]
(2 coords x 9 sample points for a 3x3 kernel), computed inside
DeformAlignBlock.forward() but never returned. Monkey-patch the
bound forward to capture them (no source edits).
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.ops import deform_conv2d

from craft_v2.pipeline import CRAFTv2
from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.train import _make_split_json

DATASET_ROOT = "/home/elsas/Anti-UAV/mamba/anti-uav-dataset-300"
FRAMES_ROOT  = "/home/elsas/msdn/data/frames"
SPLIT        = "mini_train"
SAMPLE_IDX   = 0  # 20190925_101846_1_1, frame 0, exist=1
OUT_DIR      = "/home/elsas/msdn/craft_v2/outputs/alignment_verification"
STEP         = 8  # sample every 8th spatial location for quiver

CHECKPOINTS = {
    "mini_run": "/home/elsas/msdn/craft_v2/runs/mini_run/best.pt",
    "mini_run_infonce": "/home/elsas/msdn/craft_v2/runs/mini_run_infonce/best.pt",
}


def make_capturing_forward(block, store):
    def captured_forward(f_rgb_enh, f_ir_enh):
        x = torch.cat([f_rgb_enh, f_ir_enh], dim=1)
        feat = block.offset_net(x)

        dp_rgb   = block.offset_rgb(feat)
        mask_rgb = block.mask_rgb(feat).sigmoid()
        dp_ir    = block.offset_ir(feat)
        mask_ir  = block.mask_ir(feat).sigmoid()

        store["dp_rgb"] = dp_rgb.detach().clone()
        store["dp_ir"]  = dp_ir.detach().clone()

        offset_mag = (dp_rgb.abs().mean() + dp_ir.abs().mean()).item() / 2.0

        f_rgb_aligned = block.norm_rgb(
            deform_conv2d(f_rgb_enh, dp_rgb, block.weight_rgb, mask=mask_rgb, padding=1)
        )
        f_ir_aligned = block.norm_ir(
            deform_conv2d(f_ir_enh, dp_ir, block.weight_ir, mask=mask_ir, padding=1)
        )
        return f_rgb_aligned, f_ir_aligned, mask_rgb, mask_ir, offset_mag

    return captured_forward


def normalize01(x):
    x = x - x.min()
    denom = x.max()
    return x / denom if denom > 1e-8 else x


def run_for_checkpoint(tag, ckpt_path, batch):
    if not os.path.isfile(ckpt_path):
        print(f"[{tag}] checkpoint not found at {ckpt_path} — skipping")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CRAFTv2(ch=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rgb = batch["rgb"][SAMPLE_IDX:SAMPLE_IDX + 1]
    ir  = batch["ir"][SAMPLE_IDX:SAMPLE_IDX + 1]
    rgb_mask = batch.get("rgb_mask_80")
    ir_mask  = batch.get("ir_mask_80")
    rgb_mask = None if rgb_mask is None else rgb_mask[SAMPLE_IDX:SAMPLE_IDX + 1]
    ir_mask  = None if ir_mask  is None else ir_mask[SAMPLE_IDX:SAMPLE_IDX + 1]

    store = {}
    orig_forward = model.deform_align.forward
    model.deform_align.forward = make_capturing_forward(model.deform_align, store)

    with torch.no_grad():
        f_rgb, f_ir = model.stem(rgb, ir)
        f_rgb_enh, f_ir_enh = model.msp_align(f_rgb, f_ir, rgb_mask=rgb_mask, ir_mask=ir_mask)
        model.deform_align(f_rgb_enh, f_ir_enh)

    model.deform_align.forward = orig_forward

    dp_rgb = store["dp_rgb"]  # [1, 18, 80, 80]
    dp_ir  = store["dp_ir"]
    print(f"[{tag}] dp_rgb shape: {tuple(dp_rgb.shape)}  dp_ir shape: {tuple(dp_ir.shape)}")

    # 18 channels = 9 sample points x (dy, dx). torchvision deform_conv2d offset
    # layout is [..., 2*k, H, W] with channel order (dy_0,dx_0,dy_1,dx_1,...).
    # Use the mean offset across the 9 sample points as a representative (dy,dx) per pixel.
    def mean_dy_dx(dp):
        dp = dp[0]  # [18, H, W]
        dy = dp[0::2].mean(dim=0)  # [H, W]
        dx = dp[1::2].mean(dim=0)  # [H, W]
        return dy.numpy(), dx.numpy()

    dy_rgb, dx_rgb = mean_dy_dx(dp_rgb)
    dy_ir, dx_ir   = mean_dy_dx(dp_ir)

    H, W = dy_rgb.shape
    bg = normalize01(f_rgb_enh[0].mean(dim=0)).numpy()

    ys, xs = np.meshgrid(np.arange(0, H, STEP), np.arange(0, W, STEP), indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dy, dx, name in [(axes[0], dy_rgb, dx_rgb, "RGB offsets"), (axes[1], dy_ir, dx_ir, "IR offsets")]:
        ax.imshow(bg, cmap="gray")
        u = dx[ys, xs]
        v = dy[ys, xs]
        ax.quiver(xs, ys, u, v, color="red", angles="xy", scale_units="xy", scale=0.2)
        ax.set_title(f"{tag} — {name}")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"offset_field_quiver_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[{tag}] saved -> {out_path}")

    mag_rgb = dp_rgb.abs()
    mag_ir  = dp_ir.abs()
    mean_mag = (mag_rgb.mean() + mag_ir.mean()).item() / 2.0
    max_mag  = max(mag_rgb.max().item(), mag_ir.max().item())

    # structure check: compare std of mean direction across space vs std if offsets were
    # pure noise (a rough proxy: ratio of |mean offset vector| to mean offset magnitude)
    mean_vec_len_rgb = np.sqrt(dy_rgb.mean() ** 2 + dx_rgb.mean() ** 2)
    mean_vec_len_ir  = np.sqrt(dy_ir.mean() ** 2 + dx_ir.mean() ** 2)
    avg_mag_rgb = np.sqrt(dy_rgb ** 2 + dx_rgb ** 2).mean()
    avg_mag_ir  = np.sqrt(dy_ir ** 2 + dx_ir ** 2).mean()
    coherence_rgb = mean_vec_len_rgb / (avg_mag_rgb + 1e-8)
    coherence_ir  = mean_vec_len_ir / (avg_mag_ir + 1e-8)

    print(f"[{tag}] mean |offset|: {mean_mag:.4f}   max |offset|: {max_mag:.4f}")
    print(f"[{tag}] directional coherence (0=random, 1=all-same-direction): "
          f"RGB={coherence_rgb:.3f}  IR={coherence_ir:.3f}")

    if coherence_rgb > 0.5 or coherence_ir > 0.5:
        verdict = "STRUCTURED — offsets show a consistent dominant direction (systematic correction)."
    elif mean_mag < 1e-3:
        verdict = "NEAR-ZERO — offsets barely deviate from zero-init, DeformAlign is doing ~nothing."
    else:
        verdict = "NOISE-LIKE — offsets are non-trivial in magnitude but lack a consistent direction."
    print(f"[{tag}] VERDICT: {verdict}")
    print()

    return {"mean_mag": mean_mag, "max_mag": max_mag, "coherence_rgb": coherence_rgb, "coherence_ir": coherence_ir}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    split_json = _make_split_json(DATASET_ROOT, SPLIT)
    ds = AntiUAVFrameDatasetV2(dataset_root=DATASET_ROOT, split=SPLIT, split_json=split_json, frames_root=FRAMES_ROOT)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    seq = batch["seq_name"][SAMPLE_IDX]
    fidx = int(batch["frame_idx"][SAMPLE_IDX])
    print(f"Using sample: seq={seq}  frame_idx={fidx}")
    print()

    summary = {}
    for tag, path in CHECKPOINTS.items():
        summary[tag] = run_for_checkpoint(tag, path, batch)

    print("=" * 70)
    print("SUMMARY")
    for tag, stats in summary.items():
        if stats is None:
            print(f"  {tag}: not available")
        else:
            print(f"  {tag}: mean_mag={stats['mean_mag']:.4f} max_mag={stats['max_mag']:.4f} "
                  f"coherence_rgb={stats['coherence_rgb']:.3f} coherence_ir={stats['coherence_ir']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
