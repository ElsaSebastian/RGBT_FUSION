"""
Test 1 — Shuffled-pair collapse diagnostic for MSPAlign.

Loads mini_run/best.pt, runs SharedStem + MSPAlign on 8 frames from
mini_train, and compares cosine similarity of TRUE (matched) RGB/IR
pairs against SHUFFLED (mismatched) pairs.

If both are near 1.0 with a near-zero gap, MSPAlign has collapsed to
an image-independent representation rather than learning genuine
cross-modal correspondence.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from craft_v2.pipeline import CRAFTv2
from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.train import _make_split_json

DATASET_ROOT = "/home/elsas/Anti-UAV/mamba/anti-uav-dataset-300"
FRAMES_ROOT  = "/home/elsas/msdn/data/frames"
CKPT_PATH    = "/home/elsas/msdn/craft_v2/runs/mini_run/best.pt"
SPLIT        = "mini_train"
N_FRAMES     = 8


def main():
    torch.manual_seed(0)

    split_json = _make_split_json(DATASET_ROOT, SPLIT)
    ds = AntiUAVFrameDatasetV2(
        dataset_root=DATASET_ROOT,
        split=SPLIT,
        split_json=split_json,
        frames_root=FRAMES_ROOT,
    )
    loader = DataLoader(ds, batch_size=N_FRAMES, shuffle=True)
    batch = next(iter(loader))

    print(f"Loaded {batch['rgb'].shape[0]} frames from split='{SPLIT}'")
    print(f"exist values: {batch['exist'].tolist()}")
    print()

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = CRAFTv2(ch=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print()

    with torch.no_grad():
        f_rgb, f_ir = model.stem(batch["rgb"], batch["ir"])
        f_rgb_enh, f_ir_enh = model.msp_align(
            f_rgb, f_ir,
            rgb_mask=batch.get("rgb_mask_80"),
            ir_mask=batch.get("ir_mask_80"),
        )

    B, C, H, W = f_rgb_enh.shape
    print(f"f_rgb_enh / f_ir_enh shape: {tuple(f_rgb_enh.shape)}")
    print()

    # --- Mean-pool spatial dims to get one vector per sample ---
    rgb_vec = f_rgb_enh.mean(dim=(2, 3))  # [B, C]
    ir_vec  = f_ir_enh.mean(dim=(2, 3))   # [B, C]

    # --- True pairs ---
    true_sim = F.cosine_similarity(rgb_vec, ir_vec, dim=-1)  # [B]

    # --- Shuffled pairs: fixed permutation with no fixed points ---
    perm = torch.randperm(B)
    # ensure no i == perm[i]; if any fixed points, rotate by 1
    if (perm == torch.arange(B)).any():
        perm = torch.roll(torch.arange(B), shifts=1)
    shuffled_sim = F.cosine_similarity(rgb_vec, ir_vec[perm], dim=-1)  # [B]

    true_vals = [round(v, 4) for v in true_sim.tolist()]
    shuf_vals = [round(v, 4) for v in shuffled_sim.tolist()]

    print(f"True pairs similarity:     mean={true_sim.mean():.4f}  std={true_sim.std():.4f}  values={true_vals}")
    print(f"Shuffled pairs similarity: mean={shuffled_sim.mean():.4f}  std={shuffled_sim.std():.4f}  values={shuf_vals}")
    print(f"Shuffle permutation used: {perm.tolist()}")
    print()

    gap = (true_sim.mean() - shuffled_sim.mean()).item()
    print(f"Alignment gap (true - shuffled): {gap:.4f}")
    print()

    # --- Spatial variance check (one sample) ---
    sample_idx = 0
    rgb_spatial_std = f_rgb_enh[sample_idx].flatten(1).std(dim=-1).mean().item()
    ir_spatial_std  = f_ir_enh[sample_idx].flatten(1).std(dim=-1).mean().item()
    print(f"Spatial std f_rgb_enh[sample {sample_idx}] (mean over channels): {rgb_spatial_std:.4f}")
    print(f"Spatial std f_ir_enh[sample {sample_idx}]  (mean over channels): {ir_spatial_std:.4f}")
    print("(low spatial std => feature map is nearly constant across space, additional collapse evidence)")
    print()

    # --- Interpretation ---
    print("=" * 70)
    print("INTERPRETATION LOGIC:")
    print("  gap > 0.2                          -> genuine alignment")
    print("  gap < 0.05 AND both means near 1.0 -> CONFIRMED COLLAPSE")
    print("  otherwise                          -> inconclusive")
    print("=" * 70)

    if gap > 0.2:
        verdict = "GENUINE ALIGNMENT — true pairs meaningfully more similar than random pairs."
    elif gap < 0.05 and true_sim.mean() > 0.9 and shuffled_sim.mean() > 0.9:
        verdict = "CONFIRMED COLLAPSE — model treats all pairs as similar regardless of content."
    else:
        verdict = "INCONCLUSIVE — gap and/or similarity levels do not clearly match either pattern."

    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
