"""
Test 2 — Attention map inspection for MSPAlign.

MSPAlign._mha() computes the attention matrix as a local variable that
is never returned or stored on the module, so no forward-hook can see
it directly. This script monkey-patches the bound `_mha` method on a
live model instance (no source file edits) to capture `attn` for one
TRUE pair and one MISMATCHED pair, then compares them.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from craft_v2.pipeline import CRAFTv2
from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.train import _make_split_json

DATASET_ROOT = "/home/elsas/Anti-UAV/mamba/anti-uav-dataset-300"
FRAMES_ROOT  = "/home/elsas/msdn/data/frames"
CKPT_PATH    = "/home/elsas/msdn/craft_v2/runs/mini_run/best.pt"
SPLIT        = "mini_train"
N_FRAMES     = 8
OUT_DIR      = "/home/elsas/msdn/craft_v2/outputs/attention_diag"


def make_capturing_mha(model, store, tag):
    """Returns a replacement for model.msp_align._mha that records attn under `tag`."""
    orig_mha = model.msp_align._mha

    def captured_mha(q_feat, kv_feat, q_proj, k_proj, v_proj, out_proj):
        B, Nq, C = q_feat.shape
        Nkv = kv_feat.shape[1]
        H, D = model.msp_align.num_heads, model.msp_align.head_dim

        Q = q_proj(q_feat).view(B, Nq, H, D).transpose(1, 2)
        K = k_proj(kv_feat).view(B, Nkv, H, D).transpose(1, 2)
        V = v_proj(kv_feat).view(B, Nkv, H, D).transpose(1, 2)

        scale = D ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, C)

        store[tag] = attn.detach().clone()  # [B, H, Nq, Nkv]
        return out_proj(out)

    return captured_mha


def run_pair(model, rgb, ir, rgb_mask, ir_mask, store, tag):
    model.msp_align._mha = make_capturing_mha(model, store, tag)
    with torch.no_grad():
        f_rgb, f_ir = model.stem(rgb, ir)
        model.msp_align(f_rgb, f_ir, rgb_mask=rgb_mask, ir_mask=ir_mask)
    # _mha is called twice per forward (rgb->ir direction, ir->rgb direction);
    # store[tag] ends up holding the LAST call (ir-attends-to-rgb). Capture both explicitly.


def run_pair_both_directions(model, rgb, ir, rgb_mask, ir_mask):
    """Capture both attention directions by patching with a call counter."""
    calls = []
    orig_mha = model.msp_align._mha

    def captured_mha(q_feat, kv_feat, q_proj, k_proj, v_proj, out_proj):
        B, Nq, C = q_feat.shape
        Nkv = kv_feat.shape[1]
        H, D = model.msp_align.num_heads, model.msp_align.head_dim

        Q = q_proj(q_feat).view(B, Nq, H, D).transpose(1, 2)
        K = k_proj(kv_feat).view(B, Nkv, H, D).transpose(1, 2)
        V = v_proj(kv_feat).view(B, Nkv, H, D).transpose(1, 2)

        scale = D ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, C)

        calls.append(attn.detach().clone())  # [B, H, Nq, Nkv]
        return out_proj(out)

    model.msp_align._mha = captured_mha
    with torch.no_grad():
        f_rgb, f_ir = model.stem(rgb, ir)
        model.msp_align(f_rgb, f_ir, rgb_mask=rgb_mask, ir_mask=ir_mask)
    model.msp_align._mha = orig_mha

    # forward() calls: rgb_upd = _mha(q_rgb, kv_ir, ...) then ir_upd = _mha(q_ir, kv_rgb, ...)
    attn_rgb_to_ir, attn_ir_to_rgb = calls[0], calls[1]
    return attn_rgb_to_ir, attn_ir_to_rgb


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    split_json = _make_split_json(DATASET_ROOT, SPLIT)
    ds = AntiUAVFrameDatasetV2(
        dataset_root=DATASET_ROOT, split=SPLIT, split_json=split_json, frames_root=FRAMES_ROOT,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=N_FRAMES, shuffle=True)
    batch = next(iter(loader))

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = CRAFTv2(ch=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rgb, ir = batch["rgb"], batch["ir"]
    rgb_mask = batch.get("rgb_mask_80")
    ir_mask = batch.get("ir_mask_80")

    i, j = 0, 1  # true pair = (i,i); mismatched pair = (i,j)
    print(f"True pair: frame {i} RGB with frame {i} IR")
    print(f"Mismatched pair: frame {i} RGB with frame {j} IR")
    print()

    def slice_b(t, idx):
        return None if t is None else t[idx:idx+1]

    # TRUE pair forward
    attn_true_r2i, attn_true_i2r = run_pair_both_directions(
        model, rgb[i:i+1], ir[i:i+1], slice_b(rgb_mask, i), slice_b(ir_mask, i)
    )

    # MISMATCHED pair forward: frame i's RGB with frame j's IR
    attn_mis_r2i, attn_mis_i2r = run_pair_both_directions(
        model, rgb[i:i+1], ir[j:j+1], slice_b(rgb_mask, i), slice_b(ir_mask, j)
    )

    print(f"attn shape (rgb-attends-to-ir): {tuple(attn_true_r2i.shape)}  [B,H,Nq=6400,Nkv=256]")
    print()

    # Use rgb->ir attention, head 0, average over query tokens to get a per-key-token profile,
    # AND reshape per-query attention to query-grid heatmap (sum over kv) for spatial visualization.
    def to_heatmap(attn, head=0):
        # attn: [1, H, Nq, Nkv] -> use head `head`, sum over kv dim to show "total attention mass per query loc"
        a = attn[0, head]               # [Nq, Nkv]
        mass = a.sum(dim=-1)            # [Nq] (should be ~1 since softmax over kv, so this is near-constant;
                                          # use entropy instead for a more informative map)
        # Use negative entropy of the attention distribution per query token as the informative signal:
        # low entropy = sharply focused attention (real correspondence), high entropy = diffuse/uniform attention
        entropy = -(a * (a.clamp(min=1e-9)).log()).sum(dim=-1)  # [Nq]
        side = int(entropy.shape[0] ** 0.5)
        return entropy.view(side, side).numpy()

    heat_true = to_heatmap(attn_true_r2i)
    heat_mis = to_heatmap(attn_mis_r2i)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axes[0].imshow(heat_true, cmap="viridis")
    axes[0].set_title("TRUE pair — attention entropy per query")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(heat_mis, cmap="viridis")
    axes[1].set_title("MISMATCHED pair — attention entropy per query")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()

    true_path = os.path.join(OUT_DIR, "attention_true_pair.png")
    mis_path = os.path.join(OUT_DIR, "attention_mismatched_pair.png")
    fig.savefig(true_path, dpi=120)
    plt.close(fig)

    # also save them separately as requested
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    im = ax2.imshow(heat_true, cmap="viridis")
    ax2.set_title("TRUE pair — attention entropy per query")
    plt.colorbar(im, ax=ax2)
    fig2.savefig(true_path, dpi=120)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(5, 4.5))
    im = ax3.imshow(heat_mis, cmap="viridis")
    ax3.set_title("MISMATCHED pair — attention entropy per query")
    plt.colorbar(im, ax=ax3)
    fig3.savefig(mis_path, dpi=120)
    plt.close(fig3)

    print(f"Saved: {true_path}")
    print(f"Saved: {mis_path}")
    print()

    # L2 difference between the full raw attention tensors (rgb->ir direction)
    l2_diff_r2i = (attn_true_r2i - attn_mis_r2i).pow(2).sum().sqrt().item()
    l2_diff_i2r = (attn_true_i2r - attn_mis_i2r).pow(2).sum().sqrt().item()
    l2_norm_true_r2i = attn_true_r2i.pow(2).sum().sqrt().item()

    print(f"L2 diff (true vs mismatched), rgb-attends-to-ir direction: {l2_diff_r2i:.4f}")
    print(f"L2 diff (true vs mismatched), ir-attends-to-rgb direction: {l2_diff_i2r:.4f}")
    print(f"L2 norm of true-pair attn tensor (rgb->ir) for scale reference: {l2_norm_true_r2i:.4f}")
    relative_diff = l2_diff_r2i / l2_norm_true_r2i
    print(f"Relative L2 diff (rgb->ir): {relative_diff:.4f}")
    print()

    print("=" * 70)
    if relative_diff < 0.05:
        verdict = "Attention maps nearly IDENTICAL regardless of IR content -> collapse at attention stage itself."
    elif relative_diff > 0.3:
        verdict = "Attention maps differ substantially with content -> attention is content-sensitive (not collapsed at this stage)."
    else:
        verdict = "Attention maps differ moderately -> inconclusive at the attention stage alone."
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
