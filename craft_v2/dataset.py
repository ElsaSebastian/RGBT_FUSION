"""
CRAFTv2 dataset — single-frame, reads directly from mp4 videos.

Directory layout expected:
    <dataset_root>/
        <split>/          # train | val | test
            <seq_name>/
                infrared.mp4
                visible.mp4
                infrared.json   # {"exist": [...], "gt_rect": [[x,y,w,h], ...]}
                visible.json

Split membership files (JSON, keys = seq_name, values = [attr, ...]):
    <split_dir>/train.json
    <split_dir>/val.json
    <split_dir>/test.json

__getitem__ returns a dict with letterbox parameters so the eval script
can apply the inverse transform:
    x_orig = (x_lb - pad_x) / scale
    y_orig = (y_lb - pad_y) / scale
    w_orig =  w_lb / scale
    h_orig =  h_lb / scale
"""

import json
import os
from typing import List, Tuple, Dict, Any

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Letterbox helpers (self-contained, no dependency on msdn/)
# ---------------------------------------------------------------------------

def _letterbox_image(img: np.ndarray, target: int = 320) -> Tuple[np.ndarray, float, int, int]:
    """
    Letterbox-resize img to target×target.
    Returns (padded_img, scale, pad_x, pad_y).
    pad_x / pad_y are integer pixel offsets of the content region.
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2

    if img.ndim == 3:
        out = np.zeros((target, target, img.shape[2]), dtype=np.uint8)
    else:
        out = np.zeros((target, target), dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out, scale, pad_x, pad_y


def _letterbox_box(box_xywh, scale: float, pad_x: int, pad_y: int):
    """Transform [x, y, w, h] from original space to letterboxed space."""
    x, y, w, h = box_xywh
    return (
        x * scale + pad_x,
        y * scale + pad_y,
        w * scale,
        h * scale,
    )


# ---------------------------------------------------------------------------
# Per-frame index entry
# ---------------------------------------------------------------------------

class _FrameEntry:
    __slots__ = ("vis_path", "ir_path", "vis_exist", "ir_exist",
                 "vis_box", "ir_box", "seq_name", "frame_idx",
                 "orig_vis_w", "orig_vis_h", "orig_ir_w", "orig_ir_h")

    def __init__(self, vis_path, ir_path, vis_exist, ir_exist,
                 vis_box, ir_box, seq_name, frame_idx,
                 orig_vis_w, orig_vis_h, orig_ir_w, orig_ir_h):
        self.vis_path   = vis_path
        self.ir_path    = ir_path
        self.vis_exist  = vis_exist
        self.ir_exist   = ir_exist
        self.vis_box    = vis_box    # [x,y,w,h] original vis pixels or None
        self.ir_box     = ir_box     # [x,y,w,h] original ir  pixels or None
        self.seq_name   = seq_name
        self.frame_idx  = frame_idx
        self.orig_vis_w = orig_vis_w
        self.orig_vis_h = orig_vis_h
        self.orig_ir_w  = orig_ir_w
        self.orig_ir_h  = orig_ir_h


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class AntiUAVFrameDatasetV2(Dataset):
    """
    Single-frame dataset for CRAFTv2.

    Parameters
    ----------
    dataset_root : str
        Root that contains the annotation JSONs.
        Layout: <dataset_root>/<split>/<seq_name>/infrared.json
                <dataset_root>/<split>/<seq_name>/visible.json
    split : str
        One of 'train', 'val', 'test'.
    split_json : str
        Path to the split membership JSON.
        Keys = seq_name, values = [attr, ...].
    frames_root : str or None
        Ignored — kept for CLI backwards-compatibility. Frames are read
        directly from mp4 files in the dataset_root tree.
    target_size : int
        Letterbox target size (default 320).
    """

    # Anti-UAV dataset native resolutions
    VIS_W, VIS_H = 1920, 1080
    IR_W,  IR_H  = 640,  512

    def __init__(
        self,
        dataset_root: str,
        split: str,
        split_json: str,
        frames_root: str = None,
        target_size: int = 320,
        preload: bool = False,
    ):
        self.target_size = target_size
        self.split = split
        self._frames_root = frames_root  # None → use mp4; set → use JPEGs
        self._frame_cache: Dict[str, np.ndarray] = {}

        with open(split_json) as f:
            split_map: Dict[str, List[str]] = json.load(f)

        self.seq_attrs: Dict[str, List[str]] = split_map
        self.entries: List[_FrameEntry] = []

        annot_split_dir = os.path.join(dataset_root, split)

        for seq_name in sorted(split_map.keys()):
            seq_dir  = os.path.join(annot_split_dir, seq_name)
            vis_mp4  = os.path.join(seq_dir, "visible.mp4")
            ir_mp4   = os.path.join(seq_dir, "infrared.mp4")
            vis_json = os.path.join(seq_dir, "visible.json")
            ir_json  = os.path.join(seq_dir, "infrared.json")

            if not os.path.isfile(vis_mp4) or not os.path.isfile(ir_mp4):
                continue

            vis_gt = self._load_gt(vis_json)
            ir_gt  = self._load_gt(ir_json)

            # Determine frame count from the annotation JSON (avoids opening mp4 twice)
            n_vis = len(vis_gt["exist"]) if vis_gt else 0
            n_ir  = len(ir_gt["exist"])  if ir_gt  else 0
            n_frames = min(n_vis, n_ir)
            if n_frames == 0:
                continue

            for i in range(n_frames):
                vis_exist = int(vis_gt["exist"][i]) if vis_gt and i < len(vis_gt["exist"]) else 0
                ir_exist  = int(ir_gt["exist"][i])  if ir_gt  and i < len(ir_gt["exist"])  else 0

                vis_box = None
                ir_box  = None
                if vis_gt and i < len(vis_gt["gt_rect"]):
                    raw = vis_gt["gt_rect"][i]
                    if len(raw) == 4:
                        vis_box = raw
                if ir_gt and i < len(ir_gt["gt_rect"]):
                    raw = ir_gt["gt_rect"][i]
                    if len(raw) == 4:
                        ir_box = raw

                self.entries.append(_FrameEntry(
                    vis_path   = vis_mp4,
                    ir_path    = ir_mp4,
                    vis_exist  = vis_exist,
                    ir_exist   = ir_exist,
                    vis_box    = vis_box,
                    ir_box     = ir_box,
                    seq_name   = seq_name,
                    frame_idx  = i,
                    orig_vis_w = self.VIS_W,
                    orig_vis_h = self.VIS_H,
                    orig_ir_w  = self.IR_W,
                    orig_ir_h  = self.IR_H,
                ))

        _ = preload  # no-op: frames read on-demand from mp4

    # ------------------------------------------------------------------
    # RAM preload — letterbox once at startup, cache uint8 320×320 arrays
    # Full-res storage would require ~930 GB for visible; letterboxed is ~61 GB.
    # Cache value: (lb_uint8_ndarray, scale, pad_x, pad_y)
    # ------------------------------------------------------------------

    def _preload_frames(self):
        import time as _time
        # Collect unique (path, is_color) pairs in order
        pairs: List[tuple] = []
        seen: set = set()
        for e in self.entries:
            if e.vis_path not in seen:
                pairs.append((e.vis_path, True))
                seen.add(e.vis_path)
            if e.ir_path not in seen:
                pairs.append((e.ir_path, False))
                seen.add(e.ir_path)

        n = len(pairs)
        t = self.target_size
        vis_kb = t * t * 3 // 1024
        ir_kb  = t * t * 1 // 1024
        est_gb = (sum(vis_kb if c else ir_kb for _, c in pairs)) / (1024 * 1024)
        print(f"[preload] Letterboxing {n:,} frames → {t}×{t} uint8  "
              f"(~{est_gb:.0f} GB)…", flush=True)
        t0 = _time.time()

        for i, (path, is_color) in enumerate(pairs):
            if is_color:
                raw = cv2.imread(path, cv2.IMREAD_COLOR)
                if raw is None:
                    raise RuntimeError(f"[preload] Cannot read: {path}")
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            else:
                raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise RuntimeError(f"[preload] Cannot read: {path}")
            lb, scale, pad_x, pad_y = _letterbox_image(raw, self.target_size)
            self._frame_cache[path] = (lb, scale, pad_x, pad_y)

            if (i + 1) % 10_000 == 0 or (i + 1) == n:
                print(f"[preload]  {i+1:,}/{n:,}  "
                      f"({_time.time() - t0:.0f}s)", flush=True)

        print(f"[preload] Done — {n:,} frames in RAM.", flush=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_jpeg_frame(self, mp4_path: str, frame_idx: int, color: bool):
        """Read a pre-extracted JPEG frame from frames_root tree."""
        # mp4_path = <dataset_root>/<split>/<seq>/visible.mp4
        # jpeg path = <frames_root>/<split>/<seq>/visible/<frame_idx:04d>.jpg
        parts    = mp4_path.replace('\\', '/').split('/')
        mod      = parts[-1].replace('.mp4', '')   # 'visible' or 'infrared'
        seq      = parts[-2]
        split    = parts[-3]
        jpg_path = os.path.join(self._frames_root, split, seq, mod,
                                f'{frame_idx:04d}.jpg')
        if color:
            img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Cannot read: {jpg_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Cannot read: {jpg_path}")
        return _letterbox_image(img, self.target_size)

    def _read_mp4_frame(self, mp4_path: str, frame_idx: int, color: bool):
        """Seek to frame_idx in mp4_path and return (letterboxed, scale, pad_x, pad_y)."""
        cap = cv2.VideoCapture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Cannot read frame {frame_idx} from {mp4_path}")
        if color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return _letterbox_image(frame, self.target_size)

    @staticmethod
    def _load_gt(json_path: str):
        if not json_path or not os.path.isfile(json_path):
            return None
        with open(json_path) as f:
            return json.load(f)

    @staticmethod
    def _probe_size(img_dir: str, default_w: int, default_h: int) -> Tuple[int, int]:
        """Read one image to get native resolution; fall back to defaults."""
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                img = cv2.imread(os.path.join(img_dir, fname))
                if img is not None:
                    return img.shape[1], img.shape[0]
        return default_w, default_h

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]

        # --- Load + letterbox ---
        # Use JPEG frames if available (fast), else seek mp4 (slow but universal)
        if self._frames_root:
            vis_lb, vis_scale, vis_pad_x, vis_pad_y = self._read_jpeg_frame(
                e.vis_path, e.frame_idx, color=True)
            ir_lb,  ir_scale,  ir_pad_x,  ir_pad_y  = self._read_jpeg_frame(
                e.ir_path,  e.frame_idx, color=False)
        else:
            vis_lb, vis_scale, vis_pad_x, vis_pad_y = self._read_mp4_frame(
                e.vis_path, e.frame_idx, color=True)
            ir_lb,  ir_scale,  ir_pad_x,  ir_pad_y  = self._read_mp4_frame(
                e.ir_path,  e.frame_idx, color=False)

        # --- To tensors ---
        rgb_tensor = torch.from_numpy(vis_lb.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,320,320]
        ir_tensor  = torch.from_numpy(ir_lb.astype(np.float32)  / 255.0).unsqueeze(0)       # [1,320,320]

        # IR augmentation: random brightness + contrast jitter (train only)
        # Helps TC-HARD sequences where IR contrast is very low
        if self.split == 'train':
            import random
            brightness = random.uniform(-0.2, 0.2)
            contrast   = random.uniform(0.7, 1.3)
            ir_mean    = ir_tensor.mean()
            ir_tensor  = (ir_tensor - ir_mean) * contrast + ir_mean + brightness
            ir_tensor  = ir_tensor.clamp(0.0, 1.0)

        # --- GT boxes in letterboxed 320×320 space ---
        # exist=1 only when BOTH sensors report exist=1
        both_exist = int(e.vis_exist == 1 and e.ir_exist == 1)

        if e.vis_box is not None and both_exist:
            bvx, bvy, bvw, bvh = _letterbox_box(e.vis_box, vis_scale, vis_pad_x, vis_pad_y)
        else:
            bvx = bvy = bvw = bvh = 0.0

        if e.ir_box is not None and both_exist:
            bix, biy, biw, bih = _letterbox_box(e.ir_box, ir_scale, ir_pad_x, ir_pad_y)
        else:
            bix = biy = biw = bih = 0.0

        box_vis = torch.tensor([bvx, bvy, bvw, bvh], dtype=torch.float32)
        box_ir  = torch.tensor([bix, biy, biw, bih], dtype=torch.float32)

        # --- Padding masks: 1.0=real pixel, 0.0=letterbox padding ---
        # RGB (visible): 1920×1080 → 320×(180) with side padding
        new_w_vis = int(e.orig_vis_w * vis_scale)
        new_h_vis = int(e.orig_vis_h * vis_scale)
        mask_rgb_320 = torch.zeros(1, self.target_size, self.target_size)
        mask_rgb_320[:, vis_pad_y:vis_pad_y + new_h_vis, vis_pad_x:vis_pad_x + new_w_vis] = 1.0
        rgb_mask_80 = F.max_pool2d(mask_rgb_320.unsqueeze(0), kernel_size=4, stride=4).squeeze(0)  # [1,80,80]

        # IR: 640×512 → (320×256) with top/bottom padding
        new_w_ir = int(e.orig_ir_w * ir_scale)
        new_h_ir = int(e.orig_ir_h * ir_scale)
        mask_ir_320 = torch.zeros(1, self.target_size, self.target_size)
        mask_ir_320[:, ir_pad_y:ir_pad_y + new_h_ir, ir_pad_x:ir_pad_x + new_w_ir] = 1.0
        ir_mask_80 = F.max_pool2d(mask_ir_320.unsqueeze(0), kernel_size=4, stride=4).squeeze(0)  # [1,80,80]

        return {
            "rgb":         rgb_tensor,
            "ir":          ir_tensor,
            "rgb_mask_80": rgb_mask_80,
            "ir_mask_80":  ir_mask_80,
            "box_vis": box_vis,
            "box_ir":  box_ir,
            "exist":   torch.tensor(both_exist, dtype=torch.float32),
            # Letterbox parameters for the VISIBLE sensor (used in eval inverse transform)
            "scale":   float(vis_scale),
            "pad_x":   float(vis_pad_x),
            "pad_y":   float(vis_pad_y),
            "orig_w":  e.orig_vis_w,
            "orig_h":  e.orig_vis_h,
            # Sequence metadata (not used by model, useful for per-attr eval)
            "seq_name":  e.seq_name,
            "frame_idx": e.frame_idx,
        }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, random, tempfile, json as _json

    DATASET_ROOT = "/home/elsas/Anti-UAV/mamba/anti-uav-dataset-300"
    FRAMES_ROOT  = os.path.join(os.path.dirname(__file__), "..", "data", "frames")

    # Build split json from the real dataset_root train sequences
    train_annot_dir = os.path.join(DATASET_ROOT, "train")
    if not os.path.isdir(train_annot_dir):
        print(f"[smoke] dataset_root not found: {train_annot_dir}")
        sys.exit(1)

    seqs = {s: [] for s in sorted(os.listdir(train_annot_dir))
            if os.path.isdir(os.path.join(train_annot_dir, s))}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _json.dump(seqs, tmp); tmp.close()
    split_json = tmp.name
    print(f"[smoke] {len(seqs)} train sequences from {train_annot_dir}")

    ds = AntiUAVFrameDatasetV2(
        dataset_root=DATASET_ROOT,
        split="train",
        split_json=split_json,
        frames_root=FRAMES_ROOT,
    )
    print(f"Dataset length : {len(ds)}")

    if len(ds) == 0:
        print("[smoke] Dataset is empty.")
        sys.exit(1)

    # --- 3 random samples, preferring exist=1 ---
    random.seed(0)
    exist1_idxs = [i for i, e in enumerate(ds.entries)
                   if e.vis_exist == 1 and e.ir_exist == 1]
    print(f"exist=1 frames : {len(exist1_idxs)}")

    sample_idxs = random.sample(exist1_idxs, min(3, len(exist1_idxs)))
    if len(sample_idxs) < 3:
        sample_idxs += random.sample(range(len(ds)), 3 - len(sample_idxs))

    print("\n=== 3 samples ===")
    for idx in sample_idxs:
        s = ds[idx]
        print(f"  idx={idx:6d}  exist={s['exist'].item():.0f}"
              f"  box_vis={[round(v,2) for v in s['box_vis'].tolist()]}"
              f"  scale={s['scale']:.4f}  pad_y={s['pad_y']:.1f}"
              f"  seq={s['seq_name']}  frame={s['frame_idx']}")

        # Inverse-letterbox round-trip for exist=1 samples
        if s["exist"].item() == 1:
            bx, by, bw, bh = s["box_vis"].tolist()
            sc, px, py = s["scale"], s["pad_x"], s["pad_y"]
            x0 = (bx - px) / sc
            y0 = (by - py) / sc
            w0 = bw / sc
            h0 = bh / sc
            print(f"           inverse-lb → [{x0:.1f}, {y0:.1f}, {w0:.1f}, {h0:.1f}]"
                  f"  (should match gt_rect in visible.json)")

    print("\nSmoke-test OK.")
