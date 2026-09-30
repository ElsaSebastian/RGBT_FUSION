"""
Create mini_train.json and mini_val.json — one frame per sequence.

For each sequence, picks the first exist=1 frame (both sensors agree).
Falls back to frame 0 if no exist=1 frame is found.

Creates under dataset_root:
  mini_train/<seq>/visible.json      1-frame annotation
  mini_train/<seq>/infrared.json
  mini_train.json                    sequence list

Creates under frames_root:
  mini_train/<seq>/visible/0000.jpg  symlink to chosen frame
  mini_train/<seq>/infrared/0000.jpg

Same for mini_val from val split.

Usage:
  python -m craft_v2.tools.make_mini_split \
      --dataset-root /home/elsas/Anti-UAV/mamba/anti-uav-dataset-300 \
      --frames-root  /home/elsas/msdn/data/frames \
      --label-dir    label_new
"""

import argparse
import json
import os


def find_best_frame(vis_json_path: str, ir_json_path: str) -> int:
    """Return index of first frame where both sensors have exist=1, else 0."""
    with open(vis_json_path) as f:
        vis = json.load(f)
    with open(ir_json_path) as f:
        ir = json.load(f)

    n = min(len(vis["exist"]), len(ir["exist"]))
    for i in range(n):
        if vis["exist"][i] == 1 and ir["exist"][i] == 1:
            return i
    return 0


def make_mini(src_split: str, dst_split: str, seq_attrs: dict,
              dataset_root: str, frames_root: str):
    dst_annot_root  = os.path.join(dataset_root, dst_split)
    dst_frames_root = os.path.join(frames_root,  dst_split)

    exist1_count = 0
    exist0_count = 0
    missing      = []

    for seq_name, attrs in seq_attrs.items():
        # --- Source paths ---
        src_vis_json = os.path.join(dataset_root, src_split, seq_name, "visible.json")
        src_ir_json  = os.path.join(dataset_root, src_split, seq_name, "infrared.json")
        src_vis_dir  = os.path.join(frames_root,  src_split, seq_name, "visible")
        src_ir_dir   = os.path.join(frames_root,  src_split, seq_name, "infrared")

        if not os.path.isfile(src_vis_json) or not os.path.isfile(src_ir_json):
            missing.append(seq_name)
            continue
        if not os.path.isdir(src_vis_dir) or not os.path.isdir(src_ir_dir):
            missing.append(seq_name)
            continue

        # Load full annotations
        with open(src_vis_json) as f:
            vis_gt = json.load(f)
        with open(src_ir_json) as f:
            ir_gt = json.load(f)

        # Pick frame index
        frame_idx = find_best_frame(src_vis_json, src_ir_json)
        both_exist = int(vis_gt["exist"][frame_idx] == 1 and
                         ir_gt["exist"][frame_idx] == 1)
        if both_exist:
            exist1_count += 1
        else:
            exist0_count += 1

        # --- Create annotation dirs ---
        dst_seq_annot = os.path.join(dst_annot_root, seq_name)
        os.makedirs(dst_seq_annot, exist_ok=True)

        # Write 1-frame annotation JSONs
        vis_box = vis_gt["gt_rect"][frame_idx] if frame_idx < len(vis_gt["gt_rect"]) else [0, 0, 0, 0]
        ir_box  = ir_gt["gt_rect"][frame_idx]  if frame_idx < len(ir_gt["gt_rect"])  else [0, 0, 0, 0]

        with open(os.path.join(dst_seq_annot, "visible.json"), "w") as f:
            json.dump({"exist": [vis_gt["exist"][frame_idx]],
                       "gt_rect": [vis_box]}, f)
        with open(os.path.join(dst_seq_annot, "infrared.json"), "w") as f:
            json.dump({"exist": [ir_gt["exist"][frame_idx]],
                       "gt_rect": [ir_box]}, f)

        # --- Create frame symlink dirs ---
        dst_vis_dir = os.path.join(dst_frames_root, seq_name, "visible")
        dst_ir_dir  = os.path.join(dst_frames_root, seq_name, "infrared")
        os.makedirs(dst_vis_dir, exist_ok=True)
        os.makedirs(dst_ir_dir,  exist_ok=True)

        src_vis_frame = os.path.join(src_vis_dir, f"{frame_idx:04d}.jpg")
        src_ir_frame  = os.path.join(src_ir_dir,  f"{frame_idx:04d}.jpg")
        dst_vis_frame = os.path.join(dst_vis_dir, "0000.jpg")
        dst_ir_frame  = os.path.join(dst_ir_dir,  "0000.jpg")

        for dst, src in [(dst_vis_frame, src_vis_frame),
                         (dst_ir_frame,  src_ir_frame)]:
            if os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)

    if missing:
        print(f"  [warn] {len(missing)} sequences skipped (missing files): {missing[:5]}")

    return exist1_count, exist0_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/home/elsas/Anti-UAV/mamba/anti-uav-dataset-300")
    p.add_argument("--frames-root",  default="/home/elsas/msdn/data/frames")
    p.add_argument("--label-dir",    default="label_new")
    args = p.parse_args()

    label_dir = os.path.join(args.dataset_root, args.label_dir)

    for src_split, dst_split in [("train", "mini_train"), ("val", "mini_val")]:
        label_json = os.path.join(label_dir, f"{src_split}.json")
        with open(label_json) as f:
            seq_attrs = json.load(f)

        print(f"\n=== {src_split} → {dst_split}  ({len(seq_attrs)} sequences) ===")
        e1, e0 = make_mini(src_split, dst_split, seq_attrs,
                           args.dataset_root, args.frames_root)

        # Write split JSON at dataset_root level (where _make_split_json looks)
        out_json = os.path.join(args.dataset_root, f"{dst_split}.json")
        with open(out_json, "w") as f:
            json.dump(seq_attrs, f)

        print(f"  sequences : {len(seq_attrs)}")
        print(f"  exist=1   : {e1}")
        print(f"  exist=0   : {e0}")
        print(f"  split JSON: {out_json}")
        print(f"  annots    : {args.dataset_root}/{dst_split}/<seq>/")
        print(f"  frames    : {args.frames_root}/{dst_split}/<seq>/")


if __name__ == "__main__":
    main()
