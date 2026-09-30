"""
CRAFTv2 detection evaluation — Anti-UAV300 test split.

Usage:
    python -m craft_v2.eval_detection \
        --checkpoint    craft_v2/runs/exp01/best.pt \
        --dataset-root  /path/to/Anti-UAV-RGBT \
        --frames-root   /path/to/data/frames \
        --output-dir    craft_v2/runs/exp01/eval \
        [--batch-size 16] [--num-workers 4] [--conf-thresh 0.001]
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from craft_v2.dataset import AntiUAVFrameDatasetV2
from craft_v2.pipeline import CRAFTv2


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _xywh_to_xyxy_np(box):
    """[x, y, w, h] → [x1, y1, x2, y2]  (works on plain lists/tuples)."""
    x, y, w, h = box
    return x, y, x + w, y + h


def iou_xywh(a, b):
    """Scalar IoU between two [x,y,w,h] boxes."""
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy_np(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy_np(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(boxes, scores, iou_thresh=0.7, top_k=300):
    """
    Non-maximum suppression.
    boxes  : list of [x,y,w,h]
    scores : list of float
    Returns indices of kept boxes (sorted by score desc, max top_k).
    """
    if not boxes:
        return []

    # Sort by score descending
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    order = order[:top_k]

    kept = []
    suppressed = [False] * len(order)

    for i, idx in enumerate(order):
        if suppressed[i]:
            continue
        kept.append(idx)
        for j in range(i + 1, len(order)):
            if suppressed[j]:
                continue
            if iou_xywh(boxes[idx], boxes[order[j]]) >= iou_thresh:
                suppressed[j] = True

    return kept


# ---------------------------------------------------------------------------
# AP computation — 101-point interpolation (no external libraries)
# ---------------------------------------------------------------------------

def compute_ap(recalls, precisions):
    """
    101-point interpolated AP.
    recalls, precisions: lists/arrays of floats, same length, paired.
    """
    thresholds = [t / 100.0 for t in range(101)]   # 0.0, 0.01, ..., 1.0
    ap = 0.0
    for thr in thresholds:
        # Max precision at recalls >= thr
        p_at_thr = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(p_at_thr) if p_at_thr else 0.0
    return ap / 101.0


def compute_map(predictions, ground_truths, iou_threshold):
    """
    Standard mAP for a single IoU threshold.

    predictions : list of dicts
        {seq_name, frame_idx, box [x,y,w,h], score, orig_w, orig_h}
    ground_truths : list of dicts
        {seq_name, frame_idx, box [x,y,w,h], exist}
        exist=0 entries included to count FP — they carry no GT box.

    Returns AP (float, 0–1).
    """
    # Index GTs: (seq_name, frame_idx) → list of boxes (only where exist=1)
    gt_index = defaultdict(list)       # key → list of box dicts
    gt_matched = defaultdict(list)     # key → list of bool (matched flags)

    for gt in ground_truths:
        if gt["exist"] == 1:
            key = (gt["seq_name"], gt["frame_idx"])
            gt_index[key].append(gt["box"])
            gt_matched[key].append(False)

    total_gt = sum(len(v) for v in gt_index.values())
    if total_gt == 0:
        return 0.0

    # Sort predictions by score descending
    preds_sorted = sorted(predictions, key=lambda p: p["score"], reverse=True)

    tp = []
    fp = []

    for pred in preds_sorted:
        key = (pred["seq_name"], pred["frame_idx"])
        gt_boxes   = gt_index.get(key, [])
        matched    = gt_matched.get(key, [])

        best_iou  = 0.0
        best_j    = -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = iou_xywh(pred["box"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j   = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched[best_j] = True
            gt_matched[key] = matched
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # Cumulative sums → precision/recall curve
    cum_tp = 0
    cum_fp = 0
    recalls    = []
    precisions = []
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        recalls.append(cum_tp / total_gt)
        precisions.append(cum_tp / (cum_tp + cum_fp))

    return compute_ap(recalls, precisions)


def compute_map_range(predictions, ground_truths,
                      iou_low=0.5, iou_high=0.95, iou_step=0.05):
    """Average AP over a range of IoU thresholds (COCO-style mAP@0.5:0.95)."""
    thresholds = []
    t = iou_low
    while t <= iou_high + 1e-9:
        thresholds.append(round(t, 2))
        t += iou_step

    aps = [compute_map(predictions, ground_truths, thr) for thr in thresholds]
    return sum(aps) / len(aps) if aps else 0.0


# ---------------------------------------------------------------------------
# Size-filtered AP (APS / APM / APL)
# ---------------------------------------------------------------------------

def _filter_by_size(predictions, ground_truths, area_min, area_max):
    """
    Keep only GTs (and their matching pred frames) within the area band.
    area = gt_w * gt_h in ORIGINAL pixel space.
    Predictions on frames that have no GT in this size band are still FP.
    """
    valid_keys = set()
    gt_filtered = []
    for gt in ground_truths:
        if gt["exist"] == 1:
            area = gt["box"][2] * gt["box"][3]
            if area_min <= area < area_max:
                valid_keys.add((gt["seq_name"], gt["frame_idx"]))
                gt_filtered.append(gt)
        else:
            gt_filtered.append(gt)   # exist=0 frames always kept

    # Predictions: keep those on frames that appear in valid_keys
    # (predictions on other frames would never be TP for this size band)
    pred_filtered = [p for p in predictions
                     if (p["seq_name"], p["frame_idx"]) in valid_keys]

    return pred_filtered, gt_filtered


# ---------------------------------------------------------------------------
# Inverse letterbox
# ---------------------------------------------------------------------------

def inv_letterbox(box_lb, scale, pad_x, pad_y):
    """Convert [x,y,w,h] from letterbox space to original pixel space."""
    x, y, w, h = box_lb
    return (
        (x - pad_x) / scale,
        (y - pad_y) / scale,
        w / scale,
        h / scale,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, loader, device, conf_thresh, nms_iou, top_k):
    """
    Run model over the loader and collect raw predictions and ground truths.

    Returns:
        predictions : list of dicts
        ground_truths: list of dicts
    """
    model.eval()
    predictions  = []
    ground_truths = []

    n_batches = len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0:
                print(f"  batch {batch_idx}/{n_batches} ({100*batch_idx/n_batches:.1f}%)", flush=True)
            rgb    = batch["rgb"].to(device, non_blocking=True)
            ir     = batch["ir"].to(device,  non_blocking=True)
            if args.modality == "rgb":
                ir = torch.zeros_like(ir)
            elif args.modality == "ir":
                rgb = torch.zeros_like(rgb)

            out = model(rgb, ir)

            pred_boxes  = out["boxes"]   # [B, 6400, 4]
            pred_scores = out["scores"]  # [B, 6400, 1]

            conf = torch.sigmoid(pred_scores).squeeze(-1)  # [B, 6400]

            B = rgb.shape[0]
            for i in range(B):
                seq_name  = batch["seq_name"][i]
                frame_idx = int(batch["frame_idx"][i])
                scale     = float(batch["scale"][i])
                pad_x     = float(batch["pad_x"][i])
                pad_y     = float(batch["pad_y"][i])
                orig_w    = int(batch["orig_w"][i])
                orig_h    = int(batch["orig_h"][i])
                exist     = int(batch["exist"][i].item())
                box_vis   = batch["box_vis"][i].tolist()   # lb space

                # --- GT record ---
                if exist == 1:
                    gt_box_orig = inv_letterbox(box_vis, scale, pad_x, pad_y)
                    ground_truths.append({
                        "seq_name":  seq_name,
                        "frame_idx": frame_idx,
                        "box":       list(gt_box_orig),
                        "exist":     1,
                    })
                else:
                    ground_truths.append({
                        "seq_name":  seq_name,
                        "frame_idx": frame_idx,
                        "box":       None,
                        "exist":     0,
                    })

                # --- Filter predictions by conf threshold ---
                confs_i = conf[i].cpu().tolist()          # 6400
                boxes_i = pred_boxes[i].cpu().tolist()    # 6400 × 4

                cand_boxes  = []
                cand_scores = []
                for n in range(len(confs_i)):
                    if confs_i[n] > conf_thresh:
                        box_lb = boxes_i[n]
                        box_orig = inv_letterbox(box_lb, scale, pad_x, pad_y)
                        # Clamp to image bounds
                        x = max(0.0, box_orig[0])
                        y = max(0.0, box_orig[1])
                        w = min(box_orig[2], orig_w - x)
                        h = min(box_orig[3], orig_h - y)
                        if w > 0 and h > 0:
                            cand_boxes.append([x, y, w, h])
                            cand_scores.append(confs_i[n])

                # NMS
                kept = nms(cand_boxes, cand_scores, iou_thresh=nms_iou, top_k=top_k)
                for k in kept:
                    predictions.append({
                        "seq_name":  seq_name,
                        "frame_idx": frame_idx,
                        "box":       cand_boxes[k],
                        "score":     cand_scores[k],
                        "orig_w":    orig_w,
                        "orig_h":    orig_h,
                    })

    return predictions, ground_truths


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

ATTRIBUTES = ["OV", "OC", "FM", "SV", "LI", "TC", "LR"]


def evaluate(
    checkpoint:  str,
    dataset_root: str,
    frames_root: str,
    output_dir:  str,
    test_json:   str = None,
    split:       str = "test",
    batch_size:  int = 16,
    num_workers: int = 4,
    conf_thresh: float = 0.001,
    nms_iou:     float = 0.7,
    top_k:       int   = 300,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Split JSON ---
    if test_json is None:
        test_json = os.path.join(dataset_root, "label_new", "test.json")
    if not os.path.isfile(test_json):
        # Auto-generate from directory listing
        test_dir = os.path.join(dataset_root, "test")
        seqs = {s: [] for s in sorted(os.listdir(test_dir))
                if os.path.isdir(os.path.join(test_dir, s))}
        test_json = os.path.join(output_dir, "test_auto.json")
        with open(test_json, "w") as f:
            json.dump(seqs, f)

    with open(test_json) as f:
        test_split: dict = json.load(f)   # {seq_name: [attr, ...]}

    # --- Dataset + loader ---
    test_ds = AntiUAVFrameDatasetV2(
        dataset_root=dataset_root,
        split=split,
        split_json=test_json,
        frames_root=frames_root if frames_root else None,
    )

    def collate(batch):
        keys_tensor = ["rgb", "ir", "box_vis", "box_ir", "exist"]
        keys_scalar = ["scale", "pad_x", "pad_y", "orig_w", "orig_h",
                       "seq_name", "frame_idx"]
        out = {k: torch.stack([s[k] for s in batch]) for k in keys_tensor}
        for k in keys_scalar:
            out[k] = [s[k] for s in batch]
        return out

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
        drop_last=False,
    )

    total_frames = len(test_ds)
    n_seqs = len(test_split)
    print(f"Test sequences : {n_seqs}")
    print(f"Total frames   : {total_frames:,}")

    # --- Model ---
    model = CRAFTv2(ch=64, use_convnext=args.use_convnext).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # --- Inference ---
    print("Running inference …")
    predictions, ground_truths = run_inference(
        model, loader, device, conf_thresh, nms_iou, top_k
    )
    print(f"  Predictions: {len(predictions):,}  |  GT frames: {len(ground_truths):,}")

    # --- Overall mAP ---
    print("Computing metrics …")
    map50     = compute_map(predictions, ground_truths, iou_threshold=0.5)
    map50_95  = compute_map_range(predictions, ground_truths, 0.5, 0.95, 0.05)

    # --- Size-split AP (original pixel space, AP@0.5) ---
    small_preds,  small_gts  = _filter_by_size(predictions, ground_truths,
                                                0,    1024)
    medium_preds, medium_gts = _filter_by_size(predictions, ground_truths,
                                                1024, 9216)
    large_preds,  large_gts  = _filter_by_size(predictions, ground_truths,
                                                9216, float("inf"))

    ap_small  = compute_map(small_preds,  small_gts,  0.5) if small_gts  else 0.0
    ap_medium = compute_map(medium_preds, medium_gts, 0.5) if medium_gts else 0.0
    ap_large  = compute_map(large_preds,  large_gts,  0.5) if large_gts  else 0.0

    # --- Per-attribute AP@0.5 ---
    # Build seq → attr set
    seq_attrs = {seq: set(attrs) for seq, attrs in test_split.items()}

    attr_ap = {}
    for attr in ATTRIBUTES:
        attr_seqs = {seq for seq, attrs in seq_attrs.items() if attr in attrs}
        if not attr_seqs:
            attr_ap[attr] = float("nan")
            continue
        attr_preds = [p for p in predictions if p["seq_name"] in attr_seqs]
        attr_gts   = [g for g in ground_truths if g["seq_name"] in attr_seqs]
        attr_ap[attr] = compute_map(attr_preds, attr_gts, 0.5)

    # --- Per-sequence results ---
    seq_names = list(test_split.keys())
    seq_results = {}
    for seq in seq_names:
        s_preds = [p for p in predictions if p["seq_name"] == seq]
        s_gts   = [g for g in ground_truths if g["seq_name"] == seq]
        seq_results[seq] = {
            "ap50":   compute_map(s_preds, s_gts, 0.5),
            "n_preds": len(s_preds),
            "n_gt":    sum(1 for g in s_gts if g["exist"] == 1),
            "attrs":   list(test_split[seq]),
        }

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    # detection_results.json
    results_payload = {
        "checkpoint": checkpoint,
        "n_seqs":     n_seqs,
        "n_frames":   total_frames,
        "map50":      map50,
        "map50_95":   map50_95,
        "ap_small":   ap_small,
        "ap_medium":  ap_medium,
        "ap_large":   ap_large,
        "attr_ap":    attr_ap,
        "per_seq":    seq_results,
    }
    with open(os.path.join(output_dir, "detection_results.json"), "w") as f:
        json.dump(results_payload, f, indent=2)

    # detection_metrics.csv
    csv_path = os.path.join(output_dir, "detection_metrics.csv")
    csv_cols  = (["checkpoint", "n_seqs", "n_frames",
                  "mAP50", "mAP50_95", "AP_small", "AP_medium", "AP_large"]
                 + [f"AP_{a}" for a in ATTRIBUTES])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        row = {
            "checkpoint": checkpoint,
            "n_seqs":     n_seqs,
            "n_frames":   total_frames,
            "mAP50":      round(map50    * 100, 2),
            "mAP50_95":   round(map50_95 * 100, 2),
            "AP_small":   round(ap_small  * 100, 2),
            "AP_medium":  round(ap_medium * 100, 2),
            "AP_large":   round(ap_large  * 100, 2),
        }
        for a in ATTRIBUTES:
            v = attr_ap[a]
            row[f"AP_{a}"] = round(v * 100, 2) if v == v else "N/A"
        w.writerow(row)

    # detection_metrics.txt  — human-readable paper table
    def pct(v):
        return f"{v * 100:5.1f}%" if v == v else "  N/A "

    txt_lines = [
        "═" * 43,
        "CRAFT-v2 Detection Results — Anti-UAV300",
        f"Checkpoint: {checkpoint}",
        f"Test sequences: {n_seqs}",
        f"Total frames: {total_frames:,}",
        "═" * 43,
        f"mAP@0.5         : {pct(map50)}",
        f"mAP@0.5:0.95    : {pct(map50_95)}",
        f"AP-Small        : {pct(ap_small)}",
        f"AP-Medium       : {pct(ap_medium)}",
        f"AP-Large        : {pct(ap_large)}",
        "",
        "Per-attribute AP@0.5:",
    ]
    for a in ATTRIBUTES:
        txt_lines.append(f"  {a:<4}: {pct(attr_ap[a])}")
    txt_lines.append("═" * 43)

    txt_body = "\n".join(txt_lines)
    print("\n" + txt_body + "\n")
    with open(os.path.join(output_dir, "detection_metrics.txt"), "w") as f:
        f.write(txt_body + "\n")

    print(f"Results saved to: {output_dir}")
    return results_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CRAFTv2 detection evaluation")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset-root",  required=True)
    p.add_argument("--frames-root",   default="")
    p.add_argument("--output-dir",    default="craft_v2/runs/eval")
    p.add_argument("--test-json",     default=None,
                   help="Path to test.json; defaults to <dataset-root>/label_new/test.json")
    p.add_argument("--split",         default="test",
                   help="Dataset split name to load frames/annotations from (default: test)")
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--conf-thresh",   type=float, default=0.001)
    p.add_argument("--nms-iou",       type=float, default=0.7)
    p.add_argument("--top-k",         type=int,   default=300)
    p.add_argument("--modality",      type=str,   default="both",
                   choices=["both", "rgb", "ir"])
    p.add_argument("--use-convnext",  action="store_true", default=False,
                   help="Use ConvNeXt-Tiny backbone (run10+); omit for run08/run09")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint   = args.checkpoint,
        dataset_root = args.dataset_root,
        frames_root  = args.frames_root,
        output_dir   = args.output_dir,
        test_json    = args.test_json,
        split        = args.split,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        conf_thresh  = args.conf_thresh,
        nms_iou      = args.nms_iou,
        top_k        = args.top_k,
    )
