"""
Test C — Edge cross-correlation alignment check.

Sobel-edge both modalities' mean-channel feature maps, cross-correlate,
and check whether the correlation peak sits at (0,0) (well-registered)
or away from it (systematic residual misalignment). Run pre- and
post-DeformAlign to see whether DeformAlign moves the peak toward (0,0).
"""

import os
import torch
import numpy as np
import cv2
from scipy.signal import correlate2d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from craft_v2.pipeline import CRAFTv2
from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.train import _make_split_json

DATASET_ROOT = "/home/elsas/Anti-UAV/mamba/anti-uav-dataset-300"
FRAMES_ROOT  = "/home/elsas/msdn/data/frames"
SPLIT        = "mini_train"
SAMPLE_IDX   = 0  # 20190925_101846_1_1, frame 0, exist=1
OUT_DIR      = "/home/elsas/msdn/craft_v2/outputs/alignment_verification"

CHECKPOINTS = {
    "mini_run": "/home/elsas/msdn/craft_v2/runs/mini_run/best.pt",
    "mini_run_infonce": "/home/elsas/msdn/craft_v2/runs/mini_run_infonce/best.pt",
}


def normalize01(x):
    x = x - x.min()
    denom = x.max()
    return x / denom if denom > 1e-8 else x


def sobel_edges(img01):
    img8 = (img01 * 255).astype(np.uint8)
    sx = cv2.Sobel(img8, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sx ** 2 + sy ** 2)


def find_peak_offset(edge_a, edge_b):
    corr = correlate2d(edge_a, edge_b, mode="same")
    H, W = corr.shape
    center_y, center_x = H // 2, W // 2
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak_idx[0] - center_y
    dx = peak_idx[1] - center_x
    peak_val = corr[peak_idx]
    zero_offset_val = corr[center_y, center_x]
    return corr, (dy, dx), peak_val, zero_offset_val


def plot_surface(corr, peak_idx, out_path, title):
    H, W = corr.shape
    center_y, center_x = H // 2, W // 2
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(corr, cmap="inferno")
    ax.scatter([center_x], [center_y], marker="+", color="cyan", s=120, label="zero offset (0,0)")
    ax.scatter([peak_idx[1]], [peak_idx[0]], marker="x", color="lime", s=120, label="peak")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    plt.colorbar(im, ax=ax)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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

    with torch.no_grad():
        f_rgb, f_ir = model.stem(rgb, ir)
        f_rgb_enh, f_ir_enh = model.msp_align(f_rgb, f_ir, rgb_mask=rgb_mask, ir_mask=ir_mask)
        f_rgb_aligned, f_ir_aligned, mask_rgb, mask_ir, offset_mag = model.deform_align(f_rgb_enh, f_ir_enh)

    def mean_map(t):
        return normalize01(t[0].mean(dim=0)).numpy()

    stages = {
        "pre_deformalign_mspalign_only": (mean_map(f_rgb_enh), mean_map(f_ir_enh)),
        "post_deformalign": (mean_map(f_rgb_aligned), mean_map(f_ir_aligned)),
    }

    results = {}
    for stage_name, (rgb_map, ir_map) in stages.items():
        edge_rgb = sobel_edges(rgb_map)
        edge_ir  = sobel_edges(ir_map)
        corr, (dy, dx), peak_val, zero_val = find_peak_offset(edge_rgb, edge_ir)
        out_path = os.path.join(OUT_DIR, f"correlation_surface_{tag}_{stage_name}.png")
        plot_surface(corr, (corr.shape[0] // 2 + dy, corr.shape[1] // 2 + dx), out_path,
                     f"{tag} | {stage_name}")
        results[stage_name] = {"dy": dy, "dx": dx, "peak_val": float(peak_val), "zero_val": float(zero_val)}
        print(f"[{tag}] {stage_name}: peak offset (dy,dx)=({dy},{dx})  "
              f"peak_corr={peak_val:.2f}  corr_at_zero={zero_val:.2f}")
        print(f"  saved -> {out_path}")

    pre_dist = (results["pre_deformalign_mspalign_only"]["dy"] ** 2 + results["pre_deformalign_mspalign_only"]["dx"] ** 2) ** 0.5
    post_dist = (results["post_deformalign"]["dy"] ** 2 + results["post_deformalign"]["dx"] ** 2) ** 0.5
    print(f"[{tag}] distance from (0,0): pre-DeformAlign={pre_dist:.2f}px  post-DeformAlign={post_dist:.2f}px")
    if post_dist < pre_dist:
        print(f"[{tag}] DeformAlign MOVED the peak closer to (0,0) — correcting misalignment as intended.")
    elif post_dist > pre_dist:
        print(f"[{tag}] DeformAlign moved the peak FARTHER from (0,0) — unexpected, worth investigating.")
    else:
        print(f"[{tag}] No change in peak distance from (0,0).")
    print()

    return results


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

    all_results = {}
    for tag, path in CHECKPOINTS.items():
        all_results[tag] = run_for_checkpoint(tag, path, batch)

    print("=" * 70)
    print("INTERPRETATION:")
    print("  Peak at/near (0,0)  -> genuinely well-aligned")
    print("  Peak far from (0,0) -> systematic misalignment remains")
    print("  post-DeformAlign peak closer to (0,0) than pre -> DeformAlign is")
    print("  correcting misalignment as intended, even if the cosine loss collapsed")
    print("=" * 70)


if __name__ == "__main__":
    main()
