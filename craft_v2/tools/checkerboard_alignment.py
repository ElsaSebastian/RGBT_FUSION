"""
Test A — Checkerboard overlay alignment check.

Composites f_rgb_aligned/f_ir_aligned (post-DeformAlign) and
f_rgb_enh/f_ir_enh (post-MSPAlign, pre-DeformAlign) into checkerboard
images so visible discontinuities at block boundaries reveal residual
spatial misalignment between modalities.
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from craft_v2.pipeline import CRAFTv2
from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.train import _make_split_json

DATASET_ROOT = "/home/elsas/Anti-UAV/mamba/anti-uav-dataset-300"
FRAMES_ROOT  = "/home/elsas/msdn/data/frames"
SPLIT        = "mini_train"
SAMPLE_IDX   = 0  # 20190925_101846_1_1, frame 0, exist=1 (fixed via shuffle=False)
OUT_DIR      = "/home/elsas/msdn/craft_v2/outputs/alignment_verification"
BLOCK        = 8
UPSCALE_TO   = 480

CHECKPOINTS = {
    "mini_run": "/home/elsas/msdn/craft_v2/runs/mini_run/best.pt",
    "mini_run_infonce": "/home/elsas/msdn/craft_v2/runs/mini_run_infonce/best.pt",
}


def normalize01(x):
    x = x - x.min()
    denom = x.max()
    return x / denom if denom > 1e-8 else x


def checkerboard_composite(map_a, map_b, block=BLOCK):
    H, W = map_a.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask = ((yy // block) + (xx // block)) % 2  # 0/1 alternating blocks
    comp = np.where(mask == 0, map_a, map_b)
    return comp, mask


def upscale(img, size=UPSCALE_TO):
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(img8, (size, size), interpolation=cv2.INTER_NEAREST)


def annotate(img_bgr, text):
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def get_sample(ds):
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    seq = batch["seq_name"][SAMPLE_IDX]
    fidx = int(batch["frame_idx"][SAMPLE_IDX])
    exist = float(batch["exist"][SAMPLE_IDX])
    return batch, seq, fidx, exist


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
        return normalize01(t[0].mean(dim=0)).numpy()  # [H,W]

    pairs = {
        "post_deformalign": (mean_map(f_rgb_aligned), mean_map(f_ir_aligned)),
        "pre_deformalign_mspalign_only": (mean_map(f_rgb_enh), mean_map(f_ir_enh)),
    }

    saved = {}
    for stage_name, (rgb_map, ir_map) in pairs.items():
        comp, _ = checkerboard_composite(rgb_map, ir_map)
        img = upscale(comp)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_bgr = annotate(img_bgr, f"{tag} | {stage_name}")
        out_path = os.path.join(OUT_DIR, f"checkerboard_composite_{tag}_{stage_name}.png")
        cv2.imwrite(out_path, img_bgr)
        saved[stage_name] = out_path
        print(f"[{tag}] saved {stage_name} -> {out_path}")

    return saved


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    split_json = _make_split_json(DATASET_ROOT, SPLIT)
    ds = AntiUAVFrameDatasetV2(dataset_root=DATASET_ROOT, split=SPLIT, split_json=split_json, frames_root=FRAMES_ROOT)
    batch, seq, fidx, exist = get_sample(ds)
    print(f"Using sample: seq={seq}  frame_idx={fidx}  exist={exist}")
    print()

    results = {}
    for tag, path in CHECKPOINTS.items():
        results[tag] = run_for_checkpoint(tag, path, batch)

    print()
    print("=" * 70)
    print("INTERPRETATION GUIDE:")
    print("  Aligned    -> edges/structures continuous across checkerboard block boundaries")
    print("  Misaligned -> visible 'jumps'/discontinuities at block edges")
    print("Compare 'post_deformalign' vs 'pre_deformalign_mspalign_only' to see if")
    print("DeformAlign improves registration over MSPAlign alone.")
    print("=" * 70)
    print()
    print(f"Sample used (kept consistent for Test B / Test C): seq={seq}, frame_idx={fidx}")


if __name__ == "__main__":
    main()
