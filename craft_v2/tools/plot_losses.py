"""
Plot loss curves from metrics.csv

Usage:
    python -m craft_v2.tools.plot_losses --csv craft_v2/runs/mini_run/metrics.csv
    python -m craft_v2.tools.plot_losses --csv craft_v2/runs/run06/metrics.csv
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(csv_path: str, out_path: str = None):
    df = pd.read_csv(csv_path)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), "loss_curves.png")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"CRAFT-v2 Loss Curves\n{csv_path}", fontsize=12)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- 1. Total loss ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(df["epoch"], df["train_loss"], label="train", color="steelblue")
    ax.plot(df["epoch"], df["val_loss"],   label="val",   color="tomato")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # --- 2. Alignment losses ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(df["epoch"], df["l_ag"], label="L_ag (cross-modal)", color="mediumseagreen")
    ax.plot(df["epoch"], df["l_al"], label="L_al (deform)",      color="darkorange")
    ax.set_title("Alignment Losses")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # --- 3. Detection loss ---
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(df["epoch"], df["l_det"], label="L_det", color="purple")
    ax.set_title("Detection Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # --- 4. Fusion loss ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(df["epoch"], df["l_f"], label="L_f (fusion)", color="teal")
    ax.set_title("Fusion Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # --- 5. offset_mag ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df["epoch"], df["offset_mag"], label="offset_mag", color="saddlebrown")
    ax.set_title("OffsetNet Magnitude\n(↑ = deformations growing)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |offset|")
    ax.legend(); ax.grid(alpha=0.3)

    # --- 6. LR ---
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(df["epoch"], df["lr"], color="gray")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.grid(alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to metrics.csv")
    p.add_argument("--out", default=None,  help="Output PNG path (default: alongside csv)")
    args = p.parse_args()
    plot(args.csv, args.out)
