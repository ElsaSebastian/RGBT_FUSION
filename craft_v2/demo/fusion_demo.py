"""
CRAFTv2 fusion pipeline visualiser.

Runs on 4 real test frames (2 exist=1, 2 exist=0) and saves one 5-row PNG per
frame, capturing intermediate tensors via forward hooks.

Usage
-----
python -m craft_v2.demo.fusion_demo \
    --checkpoint   craft_v2/runs/run05/latest.pt \
    --dataset-root /home/elsas/Anti-UAV/mamba/anti-uav-dataset-300 \
    --frames-root  /home/elsas/msdn/data/frames \
    --output-dir   craft_v2/outputs/fusion_demo
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from craft_v2.pipeline import CRAFTv2
from craft_v2.dataset import AntiUAVFrameDatasetV2


# ---------------------------------------------------------------------------
# Hook storage
# ---------------------------------------------------------------------------

class _HookStore:
    """Accumulates outputs from named forward hooks (single-sample, no grad)."""

    def __init__(self):
        self._data: Dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self, module: torch.nn.Module, name: str) -> None:
        def _hook(mod, inp, out):
            self._data[name] = out.detach().cpu() if isinstance(out, torch.Tensor) else out
        self._handles.append(module.register_forward_hook(_hook))

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self._data.get(name)

    def clear(self) -> None:
        self._data.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _to_np_img(tensor: torch.Tensor) -> np.ndarray:
    """[C,H,W] or [H,W] → H×W or H×W×C float32 in [0,1]."""
    t = tensor.float()
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
        if t.shape[2] == 1:
            t = t.squeeze(2)
    arr = t.numpy()
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.clip(0, 1)


def _mean_channels(tensor: torch.Tensor) -> np.ndarray:
    """[B,C,H,W] → H×W float32 normalised."""
    feat = tensor[0].mean(0)  # [H,W]
    arr = feat.numpy().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def _draw_box(ax, box_xywh, color="lime", lw=1.5):
    x, y, w, h = box_xywh
    rect = patches.Rectangle((x, y), w, h,
                               linewidth=lw, edgecolor=color, facecolor="none")
    ax.add_patch(rect)


def _imshow(ax, img, cmap=None, title=""):
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=7, color="white", pad=2)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Single-frame figure
# ---------------------------------------------------------------------------

def _visualise_frame(
    sample: Dict,
    model: CRAFTv2,
    device: torch.device,
    frame_label: str,
    output_path: str,
) -> None:
    rgb_raw = sample["rgb"]          # [3,320,320]
    ir_raw  = sample["ir"]           # [1,320,320]
    exist   = int(sample["exist"].item())
    box_vis = sample["box_vis"].tolist()   # letterboxed
    box_ir  = sample["box_ir"].tolist()

    # ---- Hook setup --------------------------------------------------------
    store = _HookStore()
    store.register(model.msp_align,    "msp_out")
    store.register(model.deform_align, "deform_out")
    store.register(model.hga_fuse,     "fused_out")

    # We also intercept offset magnitudes from deform_align.offset_net
    _offset_feat: Dict[str, torch.Tensor] = {}
    def _offset_hook(mod, inp, out):
        _offset_feat["feat"] = out.detach().cpu()
    _h_offset = model.deform_align.offset_net.register_forward_hook(_offset_hook)

    # Intercept the raw gate logits from HGAFuse.gate_conv (pre-sigmoid)
    _gate_feat: Dict[str, torch.Tensor] = {}
    def _gate_hook(mod, inp, out):
        _gate_feat["logits"] = out.detach().cpu()
    _h_gate = model.hga_fuse.gate_conv.register_forward_hook(_gate_hook)

    # ---- Forward pass ------------------------------------------------------
    rgb_in = rgb_raw.unsqueeze(0).to(device)
    ir_in  = ir_raw.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(rgb_in, ir_in)

    store.remove()
    _h_offset.remove()
    _h_gate.remove()

    # ---- Unpack intermediates ----------------------------------------------
    msp_out    = store.get("msp_out")       # tuple (f_rgb_enh, f_ir_enh)
    deform_out = store.get("deform_out")    # tuple (f_rgb_aligned, f_ir_aligned, mask_rgb, mask_ir, offset_mag)
    fused_out  = store.get("fused_out")     # [1,C,H,W]

    # msp_out is the tuple returned by MSPAlign.forward
    f_rgb_enh, f_ir_enh = msp_out if msp_out is not None else (None, None)
    f_rgb_aligned, f_ir_aligned, mask_rgb, mask_ir, _deform_offset_mag = (
        deform_out if deform_out is not None else (None, None, None, None, None)
    )
    f_fused = fused_out  # [1,C,H,W]

    # ---- Scalar metrics ----------------------------------------------------
    # Offset magnitude: from the 18-channel offset prediction
    offset_rgb = model.deform_align.offset_rgb
    offset_ir  = model.deform_align.offset_ir
    feat_cpu   = _offset_feat.get("feat")
    if feat_cpu is not None:
        dp_rgb = offset_rgb(feat_cpu.to("cpu"))  # [1,18,H,W]
        dp_ir  = offset_ir(feat_cpu.to("cpu"))
        offsets = torch.cat([dp_rgb, dp_ir], dim=1)  # [1,36,H,W]
        offset_mag = offsets.norm(dim=1).mean().item()
    else:
        offset_mag = float("nan")

    # Gate alpha from HGAFuse — captured directly from gate_conv's forward hook
    # (computed from the hub node output inside HGAFuse, not directly from
    # concatenated rgb/ir features as in the previous architecture version).
    gate_logits = _gate_feat.get("logits")
    if gate_logits is not None:
        gate_alpha_map = gate_logits.sigmoid()[0, 0]  # [H,W]
        gate_alpha = gate_alpha_map.mean().item()
    else:
        gate_alpha_map = None
        gate_alpha = float("nan")

    # ---- Row-6 diagnostics: score map, alpha gate map, cosine-sim gain -------
    # Score map: DetectionHead cls output reshaped to [80,80]
    score_map_np = None
    if "scores" in out:
        H_feat = W_feat = 80
        sm = out["scores"][0].reshape(H_feat, W_feat).cpu().numpy().astype(np.float32)
        lo, hi = sm.min(), sm.max()
        score_map_np = (sm - lo) / (hi - lo + 1e-8)

    # Alpha gate map [80,80]: where does the model trust RGB vs IR?
    # (reuses the gate_alpha_map captured from HGAFuse.gate_conv's forward hook above)
    alpha_map_np = None
    if gate_alpha_map is not None:
        alpha_map_np = gate_alpha_map.detach().numpy().astype(np.float32)

    # Cosine-similarity gain: per-pixel sim(f_fused, f_rgb_enh) vs sim(f_fused, f_ir_enh)
    # Positive = fused is closer to RGB; negative = closer to IR.
    cossim_np = None
    if f_fused is not None and f_rgb_enh is not None and f_ir_enh is not None:
        import torch.nn.functional as F_nn
        # Upsample enh features to match fused spatial size if needed (both 80×80 here)
        fv = f_fused[0]          # [C,H,W]
        rv = f_rgb_enh[0]        # [C,H,W]
        iv = f_ir_enh[0]         # [C,H,W]
        sim_rgb = F_nn.cosine_similarity(fv, rv, dim=0)   # [H,W]
        sim_ir  = F_nn.cosine_similarity(fv, iv, dim=0)   # [H,W]
        gain = (sim_rgb - sim_ir).numpy().astype(np.float32)  # >0 → RGB-like, <0 → IR-like
        # Normalise to [0,1] for display, centre at 0.5
        abs_max = max(abs(gain.min()), abs(gain.max())) + 1e-8
        cossim_np = (gain / abs_max) * 0.5 + 0.5

    print(f"[{frame_label}] exist={exist} | gt_box_vis={[round(v,1) for v in box_vis]}"
          f" | offset_mag={offset_mag:.4f} | gate_alpha_mean={gate_alpha:.4f}")

    # ---- Build figure -------------------------------------------------------
    plt.style.use("dark_background")
    fig, axes = plt.subplots(6, 2, figsize=(12, 14),
                              gridspec_kw={"hspace": 0.4, "wspace": 0.05})

    # Row 1: RGB raw | IR raw  (with GT box if exist=1)
    rgb_np = _to_np_img(rgb_raw)
    ir_np  = _to_np_img(ir_raw)

    ax_rgb = axes[0, 0]
    ax_ir  = axes[0, 1]
    _imshow(ax_rgb, rgb_np, title="RGB raw")
    _imshow(ax_ir,  ir_np,  cmap="gray", title="IR raw")
    if exist == 1:
        _draw_box(ax_rgb, box_vis, color="lime")
        _draw_box(ax_ir,  box_ir,  color="cyan")

    # Row 2: RGB letterboxed | IR letterboxed
    _imshow(axes[1, 0], rgb_np, title="RGB letterboxed 320×320")
    _imshow(axes[1, 1], ir_np,  cmap="gray", title="IR letterboxed 320×320")

    # Row 3: f_rgb_enh | f_ir_enh  (MSPAlign)
    if f_rgb_enh is not None:
        _imshow(axes[2, 0], _mean_channels(f_rgb_enh), cmap="viridis", title="f_rgb_enh (MSPAlign mean-ch)")
        _imshow(axes[2, 1], _mean_channels(f_ir_enh),  cmap="viridis", title="f_ir_enh  (MSPAlign mean-ch)")
    else:
        axes[2, 0].set_visible(False)
        axes[2, 1].set_visible(False)

    # Row 4: f_rgb_aligned | f_ir_aligned  (DeformAlign)
    if f_rgb_aligned is not None:
        _imshow(axes[3, 0], _mean_channels(f_rgb_aligned), cmap="viridis", title="f_rgb_aligned (DeformAlign mean-ch)")
        _imshow(axes[3, 1], _mean_channels(f_ir_aligned),  cmap="viridis", title="f_ir_aligned  (DeformAlign mean-ch)")
    else:
        axes[3, 0].set_visible(False)
        axes[3, 1].set_visible(False)

    # Row 5: f_fused mean — full width, square aspect
    for ax in axes[4, :]:
        ax.set_visible(False)

    ax_fused = fig.add_subplot(6, 1, 5)
    ax_fused.set_facecolor("#111111")
    if f_fused is not None:
        ax_fused.imshow(_mean_channels(f_fused), cmap="viridis",
                        interpolation="nearest", aspect="equal")
    ax_fused.set_title("f_fused (HGAFuse mean-ch)", fontsize=7, color="white", pad=2)
    ax_fused.set_anchor("C")
    ax_fused.axis("off")

    # Row 6: score map | alpha gate map | cosine-sim gain  (3 panels, full width)
    for ax in axes[5, :]:
        ax.set_visible(False)

    gs6 = fig.add_gridspec(6, 3, top=0.155, bottom=0.01, hspace=0.4, wspace=0.15,
                            left=0.05, right=0.95)
    ax_score  = fig.add_subplot(gs6[5, 0])
    ax_alpha  = fig.add_subplot(gs6[5, 1])
    ax_cossim = fig.add_subplot(gs6[5, 2])

    # Score map — where does the model predict UAV?
    if score_map_np is not None:
        im_s = ax_score.imshow(score_map_np, cmap="hot", interpolation="nearest",
                               aspect="equal", vmin=0, vmax=1)
        if exist == 1:
            # Mark GT box centre on score map (feature-map coords, stride=4)
            cx = (box_vis[0] + box_vis[2] / 2) / 4
            cy = (box_vis[1] + box_vis[3] / 2) / 4
            ax_score.plot(cx, cy, "g+", markersize=8, markeredgewidth=1.5)
        fig.colorbar(im_s, ax=ax_score, fraction=0.046, pad=0.04)
    ax_score.set_title("Score map (DetHead cls)\nGT centre = green +",
                        fontsize=6, color="white", pad=2)
    ax_score.axis("off")

    # Alpha gate map — which sensor does the fusion gate trust?
    if alpha_map_np is not None:
        im_a = ax_alpha.imshow(alpha_map_np, cmap="RdBu_r", interpolation="nearest",
                               aspect="equal", vmin=0, vmax=1)
        fig.colorbar(im_a, ax=ax_alpha, fraction=0.046, pad=0.04)
    ax_alpha.set_title("Alpha gate α\n1=RGB  0=IR",
                        fontsize=6, color="white", pad=2)
    ax_alpha.axis("off")

    # Cosine-sim gain — is f_fused more RGB-like or IR-like per pixel?
    if cossim_np is not None:
        im_c = ax_cossim.imshow(cossim_np, cmap="RdBu", interpolation="nearest",
                                aspect="equal", vmin=0, vmax=1)
        fig.colorbar(im_c, ax=ax_cossim, fraction=0.046, pad=0.04)
    ax_cossim.set_title("CosSim gain\nred=RGB-like  blue=IR-like",
                         fontsize=6, color="white", pad=2)
    ax_cossim.axis("off")

    # Figure title
    fig.suptitle(
        f"{frame_label}  |  exist={exist}  "
        f"offset_mag={offset_mag:.3f}  gate_α={gate_alpha:.3f}",
        fontsize=9, color="white", y=0.998,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor="#1a1a1a", edgecolor="none")
    plt.close(fig)
    print(f"  → saved {output_path}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_split_json(dataset_root: str, split: str) -> str:
    """Auto-build a split JSON from the dataset_root directory listing."""
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    seqs = {s: [] for s in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, s))}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(seqs, tmp)
    tmp.close()
    return tmp.name


def _pick_frames(
    dataset: AntiUAVFrameDatasetV2,
    n_exist1: int = 2,
    n_exist0: int = 2,
    seed: int = 42,
) -> List[int]:
    import random
    rng = random.Random(seed)

    exist1 = [i for i, e in enumerate(dataset.entries)
               if e.vis_exist == 1 and e.ir_exist == 1]
    exist0 = [i for i, e in enumerate(dataset.entries)
               if not (e.vis_exist == 1 and e.ir_exist == 1)]

    chosen1 = rng.sample(exist1, min(n_exist1, len(exist1)))
    chosen0 = rng.sample(exist0, min(n_exist0, len(exist0)))

    # Pad if not enough
    while len(chosen1) < n_exist1:
        chosen1.append(chosen1[-1] if chosen1 else 0)
    while len(chosen0) < n_exist0:
        chosen0.append(chosen0[-1] if chosen0 else 0)

    return chosen1[:n_exist1] + chosen0[:n_exist0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CRAFTv2 fusion pipeline visualiser")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--dataset-root", required=True,
                        help="Root of the Anti-UAV annotation tree")
    parser.add_argument("--frames-root",  required=True,
                        help="Root of the extracted JPEG frames tree")
    parser.add_argument("--output-dir",   default="craft_v2/outputs/fusion_demo",
                        help="Directory to write PNGs into")
    parser.add_argument("--split",        default="test",
                        help="Dataset split to draw frames from (default: test)")
    parser.add_argument("--split-json",   default=None,
                        help="Path to split JSON; auto-built from dataset-root if omitted")
    parser.add_argument("--device",       default="cpu",
                        help="Torch device (default: cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Model -------------------------------------------------------------
    model = CRAFTv2(ch=64).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- Dataset -----------------------------------------------------------
    split_json = args.split_json or _build_split_json(args.dataset_root, args.split)
    dataset = AntiUAVFrameDatasetV2(
        dataset_root=args.dataset_root,
        split=args.split,
        split_json=split_json,
        frames_root=args.frames_root,
    )
    print(f"Dataset: {len(dataset)} frames  (split={args.split})")

    # ---- Pick 4 frames (2 exist=1, 2 exist=0) ------------------------------
    indices = _pick_frames(dataset, n_exist1=2, n_exist0=2)

    filenames = [
        "frame_00_exist1.png",
        "frame_01_exist1.png",
        "frame_02_exist0.png",
        "frame_03_exist0.png",
    ]
    labels = [
        "frame_00 (exist=1)",
        "frame_01 (exist=1)",
        "frame_02 (exist=0)",
        "frame_03 (exist=0)",
    ]

    # ---- Visualise ---------------------------------------------------------
    for idx, fname, label in zip(indices, filenames, labels):
        sample = dataset[idx]
        out_path = os.path.join(args.output_dir, fname)
        _visualise_frame(sample, model, device, label, out_path)

    print("\nAll frames saved to:", args.output_dir)


if __name__ == "__main__":
    main()
