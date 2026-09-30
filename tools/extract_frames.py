#!/usr/bin/env python3
"""Pre-extract Anti-UAV300 video frames to JPEG for fast training I/O.

Output structure:
  <out>/<split>/<sequence>/visible/0000.jpg
  <out>/<split>/<sequence>/infrared/0000.jpg

Usage:
  python craft_v2/tools/extract_frames.py \
      --src /path/to/Anti-UAV300 \
      --out /path/to/frames \
      --splits train val test
"""

import argparse
import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed


def discover_sequences(dataset_root, splits):
    seqs = []
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            print(f"  Skipping {split} (not found)")
            continue
        for seq_name in sorted(os.listdir(split_dir)):
            seq_dir = os.path.join(split_dir, seq_name)
            vis_mp4 = os.path.join(seq_dir, "visible.mp4")
            ir_mp4  = os.path.join(seq_dir, "infrared.mp4")
            if os.path.isfile(vis_mp4) and os.path.isfile(ir_mp4):
                seqs.append({"split": split, "name": seq_name,
                              "vis": vis_mp4, "ir": ir_mp4})
        print(f"  {split}: {sum(1 for s in seqs if s['split']==split)} sequences")
    return seqs


def extract_sequence(seq, out_root, quality):
    for modality, vid_path in [("visible", seq["vis"]), ("infrared", seq["ir"])]:
        out_dir = os.path.join(out_root, seq["split"], seq["name"], modality)
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(vid_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            p = os.path.join(out_dir, f"{i:04d}.jpg")
            if not os.path.exists(p):
                cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        cap.release()
    return seq["split"], seq["name"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src",     required=True, help="Anti-UAV300 dataset root")
    p.add_argument("--out",     required=True, help="Output frames root")
    p.add_argument("--splits",  nargs="+", default=["train", "val", "test"])
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    seqs = discover_sequences(args.src, args.splits)
    print(f"\nTotal: {len(seqs)} sequences → {args.out}\n")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_sequence, s, args.out, args.quality): s
                   for s in seqs}
        for fut in as_completed(futures):
            split, name = fut.result()
            done += 1
            print(f"  [{done}/{len(seqs)}] {split}/{name}", flush=True)

    print(f"\nDone. Frames saved to: {args.out}")


if __name__ == "__main__":
    main()
