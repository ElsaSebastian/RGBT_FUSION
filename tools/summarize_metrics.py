"""
Print Training Results + Loss Term Analysis directly from a metrics.csv.

Usage:
    python -m craft_v2.tools.summarize_metrics --csv craft_v2/runs/mini_run/metrics.csv
    python -m craft_v2.tools.summarize_metrics --csv craft_v2/runs/run06/metrics.csv
"""

import argparse
import csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--checkpoints", type=int, nargs="+", default=None,
                   help="Epochs to show in loss term table (default: 1, 25%%, 50%%, last)")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        print("No rows in metrics.csv yet.")
        return

    epochs = [int(r["epoch"]) for r in rows]
    val_losses = [float(r["val_loss"]) for r in rows]
    best_idx = val_losses.index(min(val_losses))

    print("=== TRAINING RESULTS ===")
    print(f"Epochs completed   : {len(rows)}")
    print(f"First epoch        : {epochs[0]}")
    print(f"Last epoch         : {epochs[-1]}")
    print(f"Best val loss      : {rows[best_idx]['val_loss']}  (epoch {rows[best_idx]['epoch']})")
    print(f"Final train loss   : {rows[-1]['train_loss']}  (epoch {rows[-1]['epoch']})")
    print(f"Final val loss     : {rows[-1]['val_loss']}  (epoch {rows[-1]['epoch']})")
    reduction = (float(rows[0]["train_loss"]) - float(rows[-1]["train_loss"])) / float(rows[0]["train_loss"]) * 100
    print(f"Train loss reduction: {reduction:.2f}%")

    print()
    print("=== LOSS TERM ANALYSIS ===")
    cols = [c for c in ["l_ag", "l_al", "l_m", "l_f", "l_det", "offset_mag"] if c in rows[0]]
    rows_by_epoch = {int(r["epoch"]): r for r in rows}

    checkpoints = args.checkpoints
    if checkpoints is None:
        n = len(epochs)
        candidates = sorted(set([epochs[0], epochs[n // 4], epochs[n // 2], epochs[-1]]))
        checkpoints = [e for e in candidates if e in rows_by_epoch]

    header = f"{'Term':<12}" + "".join(f"{'Epoch ' + str(e):>14}" for e in checkpoints)
    print(header)
    print("-" * len(header))
    for c in cols:
        row = f"{c:<12}" + "".join(
            f"{float(rows_by_epoch[e][c]):>14.6f}" for e in checkpoints if e in rows_by_epoch
        )
        print(row)


if __name__ == "__main__":
    main()
