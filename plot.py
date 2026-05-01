"""
plot.py — Generate all figures from saved result JSONs.

Usage:
    python plot.py                  # reads results/*.json, saves figures/
    python plot.py --results_dir results --out_dir figures
"""

import argparse
import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

METHODS       = ["naive", "ewc", "lwf", "hybrid"]
METHOD_LABELS = {"naive": "Naive", "ewc": "EWC", "lwf": "LwF", "hybrid": "Hybrid"}
COLORS        = {"naive": "#e74c3c", "ewc": "#3498db", "lwf": "#2ecc71", "hybrid": "#9b59b6"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> dict:
    data = {}
    for method in METHODS:
        path = os.path.join(results_dir, f"{method}.json")
        if os.path.exists(path):
            with open(path) as f:
                data[method] = json.load(f)
    if not data:
        raise FileNotFoundError(f"No result JSONs found in '{results_dir}'. Run main.py first.")
    return data


def task_label(i: int) -> str:
    start = i * 20
    return f"T{i+1}\n({start}–{start+19})"


# ── plot 1: accuracy matrix heatmaps ─────────────────────────────────────────

def plot_accuracy_matrices(results: dict, out_dir: str):
    available = [m for m in METHODS if m in results]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, available):
        mat = np.array(results[method]["acc_matrix"]) * 100
        T   = mat.shape[0]

        # Mask upper triangle (not yet evaluated)
        mask = np.zeros_like(mat)
        for i in range(T):
            for j in range(i + 1, T):
                mask[i, j] = np.nan
        mat_masked = np.where(mask == 0, mat, np.nan)

        im = ax.imshow(mat_masked, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(T):
            for j in range(i + 1):
                val = mat[i, j]
                color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels([f"T{j+1}" for j in range(T)], fontsize=8)
        ax.set_yticklabels([f"After T{i+1}" for i in range(T)], fontsize=8)
        ax.set_xlabel("Task evaluated on", fontsize=9)
        ax.set_title(f"{METHOD_LABELS[method]}\nAA={results[method]['aa']:.1f}%  "
                     f"BWT={results[method]['bwt']:.1f}%", fontsize=10)

    fig.suptitle("Accuracy Matrix — each cell = accuracy on column task after training on row task",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── plot 2: AA / BWT bar chart ────────────────────────────────────────────────

def plot_aa_bwt(results: dict, out_dir: str):
    available = [m for m in METHODS if m in results]
    x     = np.arange(len(available))
    width = 0.35

    aa_vals  = [results[m]["aa"]  for m in available]
    bwt_vals = [results[m]["bwt"] for m in available]
    colors   = [COLORS[m] for m in available]
    labels   = [METHOD_LABELS[m] for m in available]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # AA bar chart
    bars = ax1.bar(x, aa_vals, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, aa_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax1.set_title("Average Accuracy (AA)\nhigher is better", fontsize=11)
    ax1.set_ylim(0, max(aa_vals) * 1.2)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.grid(axis="y", alpha=0.3)

    # BWT bar chart
    bars2 = ax2.bar(x, bwt_vals, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars2, bwt_vals):
        offset = -1.5 if val < 0 else 0.3
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                 f"{val:.1f}%", ha="center", va="top" if val < 0 else "bottom",
                 fontsize=10, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Backward Transfer (%)", fontsize=11)
    ax2.set_title("Backward Transfer (BWT)\ncloser to 0 is better", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Comparison of Continual Learning Methods on Split-CIFAR-100", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "aa_bwt_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── plot 3: per-task accuracy curves ─────────────────────────────────────────

def plot_task_curves(results: dict, out_dir: str):
    available = [m for m in METHODS if m in results]
    T         = 5
    fig, axes = plt.subplots(1, T, figsize=(4 * T, 3.5), sharey=True)

    for task_id in range(T):
        ax = axes[task_id]
        for method in available:
            mat = np.array(results[method]["acc_matrix"]) * 100
            # accuracy on task_id after each subsequent training step
            curve = [mat[i, task_id] for i in range(task_id, T)]
            x_vals = list(range(task_id + 1, T + 1))
            ax.plot(x_vals, curve, marker="o", label=METHOD_LABELS[method],
                    color=COLORS[method], linewidth=2, markersize=5)

        ax.set_title(f"Task {task_id+1}\n(classes {task_id*20}–{task_id*20+19})", fontsize=9)
        ax.set_xlabel("Tasks trained so far", fontsize=8)
        if task_id == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.set_xticks(range(task_id + 1, T + 1))
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=10, ncol=len(available))
    fig.suptitle("Per-Task Accuracy Over Time — shows forgetting as new tasks are learned",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "task_accuracy_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── plot 4: training time per task ────────────────────────────────────────────

def plot_times(results: dict, out_dir: str):
    available = [m for m in METHODS if m in results if "task_times" in results[m]]
    if not available:
        return

    T      = 5
    x      = np.arange(T)
    width  = 0.8 / len(available)
    fig, ax = plt.subplots(figsize=(9, 4))

    for i, method in enumerate(available):
        times  = results[method]["task_times"]
        offset = (i - len(available) / 2 + 0.5) * width
        ax.bar(x + offset, times, width, label=METHOD_LABELS[method],
               color=COLORS[method], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {j+1}" for j in range(T)], fontsize=10)
    ax.set_ylabel("Training time (s)", fontsize=11)
    ax.set_title("Training Time per Task", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_times.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir",     default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_results(args.results_dir)

    print(f"Loaded results for: {list(results.keys())}")
    print(f"Generating figures → {args.out_dir}/\n")

    plot_accuracy_matrices(results, args.out_dir)
    plot_aa_bwt(results, args.out_dir)
    plot_task_curves(results, args.out_dir)
    plot_times(results, args.out_dir)

    print("\nDone. All figures saved.")


if __name__ == "__main__":
    main()
