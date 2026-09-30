"""
Test 1b — Shuffled-pair collapse diagnostic for L_f (fused feature alignment).

Same template/logic as test_collapse.py, applied to f_fused (HGAFuse output)
instead of f_rgb_enh/f_ir_enh (MSPAlign output).

L_f rewards f_fused being close (cosine sim) to target = 0.5*(f_rgb_aligned + f_ir_aligned).
To test whether this is genuinely discriminative or collapsed:
  - f_fused is computed ONCE per sample from its TRUE rgb/ir pairing (the actual
    forward pass — f_fused does not change, exactly as the trained model produces it).
  - true_sim[i]     = cos_sim(f_fused[i], target_true[i])   where target_true uses the
                       correct f_ir_aligned[i]
  - shuffled_sim[i]  = cos_sim(f_fused[i], target_shuf[i])  where target_shuf substitutes
                       a mismatched f_ir_aligned[perm[i]] in place of the true partner

If f_fused genuinely encodes which IR sample it was fused with, it should match
target_true much better than target_shuf (large gap). If f_fused is generic/collapsed
(e.g. always close to some fixed direction or trivially close to any rgb-ir average),
the gap will be near zero.
"""

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
        f_rgb_aligned, f_ir_aligned, mask_rgb, mask_ir, offset_mag = model.deform_align(f_rgb_enh, f_ir_enh)
        f_fused = model.hga_fuse(f_rgb_aligned, f_ir_aligned)  # TRUE pairing, as the trained model actually produces it

    B, C, H, W = f_fused.shape
    print(f"f_fused shape: {tuple(f_fused.shape)}")
    print()

    # --- Shuffled permutation, no fixed points ---
    perm = torch.randperm(B)
    if (perm == torch.arange(B)).any():
        perm = torch.roll(torch.arange(B), shifts=1)

    # --- Targets: mean-pool over spatial dims to match _l_f's formula ---
    fused_mean = f_fused.mean(dim=(-2, -1))                 # [B, C]
    target_true_mean = 0.5 * (f_rgb_aligned + f_ir_aligned).mean(dim=(-2, -1))            # [B, C], true partner
    target_shuf_mean  = 0.5 * (f_rgb_aligned + f_ir_aligned[perm]).mean(dim=(-2, -1))      # [B, C], mismatched partner

    true_sim = F.cosine_similarity(fused_mean, target_true_mean, dim=-1)  # [B]
    shuffled_sim = F.cosine_similarity(fused_mean, target_shuf_mean, dim=-1)  # [B]

    true_vals = [round(v, 4) for v in true_sim.tolist()]
    shuf_vals = [round(v, 4) for v in shuffled_sim.tolist()]

    print(f"True pairs similarity (f_fused vs true target):     mean={true_sim.mean():.4f}  std={true_sim.std():.4f}  values={true_vals}")
    print(f"Shuffled pairs similarity (f_fused vs shuf target): mean={shuffled_sim.mean():.4f}  std={shuffled_sim.std():.4f}  values={shuf_vals}")
    print(f"Shuffle permutation used: {perm.tolist()}")
    print()

    gap = (true_sim.mean() - shuffled_sim.mean()).item()
    print(f"Alignment gap (true - shuffled): {gap:.4f}")
    print()

    # --- Spatial variance check (one sample), same logic as test_collapse.py ---
    sample_idx = 0
    fused_spatial_std = f_fused[sample_idx].flatten(1).std(dim=-1).mean().item()
    print(f"Spatial std f_fused[sample {sample_idx}] (mean over channels): {fused_spatial_std:.4f}")
    print("(low spatial std => feature map is nearly constant across space, additional collapse evidence)")
    print()

    # --- Interpretation, same thresholds as test_collapse.py ---
    print("=" * 70)
    print("INTERPRETATION LOGIC:")
    print("  gap > 0.2                          -> genuine discrimination (L_f is meaningful)")
    print("  gap < 0.05 AND both means near 1.0 -> CONFIRMED COLLAPSE")
    print("  otherwise                          -> inconclusive")
    print("=" * 70)

    if gap > 0.2:
        verdict = "GENUINE — f_fused matches its true partner meaningfully better than a mismatched one."
    elif gap < 0.05 and true_sim.mean() > 0.9 and shuffled_sim.mean() > 0.9:
        verdict = "CONFIRMED COLLAPSE — f_fused matches any target equally well regardless of true pairing."
    else:
        verdict = "INCONCLUSIVE — gap and/or similarity levels do not clearly match either pattern."

    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
