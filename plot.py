"""
plot.py — Generate all figures for the intermediate report.

Reads results/*.json produced by train.py and saves figures to ./figures/.

Figures produced:
    1. accuracy_matrix_<method>.png   — heatmap of the acc matrix per method
    2. aa_bwt_comparison.png          — grouped bar chart AA vs BWT
    3. resource_comparison.png        — time and RAM per task per method
    4. task_accuracy_curves.png       — per-task accuracy degradation over time
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_COLORS = {
    "naive":  "#e74c3c",
    "ewc":    "#3498db",
    "lwf":    "#2ecc71",
    "hybrid": "#9b59b6",
}
METHOD_LABELS = {
    "naive":  "Naive",
    "ewc":    "EWC",
    "lwf":    "LwF",
    "hybrid": "Hybrid",
}


def load_results() -> dict:
    results = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            results[name] = json.load(f)
    if not results:
        raise FileNotFoundError(f"No JSON files found in {RESULTS_DIR}/. Run train.py first.")
    return results


# ── Figure 1: accuracy heatmap per method ─────────────────────────────────────

def plot_acc_matrix(method_name: str, data: dict):
    mat = np.array(data["acc_matrix"]) * 100
    T   = mat.shape[0]
    # Only lower-triangle is filled; mask upper
    mask = np.tril(np.ones((T, T), dtype=bool))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.where(mask, mat, np.nan), vmin=0, vmax=100,
                   cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"T{i+1}" for i in range(T)])
    ax.set_yticklabels([f"After T{i+1}" for i in range(T)])
    ax.set_xlabel("Evaluated on task")
    ax.set_ylabel("After training on task")
    ax.set_title(f"Accuracy Matrix — {METHOD_LABELS.get(method_name, method_name)}")

    for i in range(T):
        for j in range(T):
            if mask[i, j]:
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                        fontsize=9, color="black")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"accuracy_matrix_{method_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: AA / BWT bar chart ──────────────────────────────────────────────

def plot_aa_bwt(results: dict):
    methods = list(results.keys())
    aas  = [results[m]["aa"]  for m in methods]
    bwts = [results[m]["bwt"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, aas,  width, label="AA (%)",
                   color=[METHOD_COLORS.get(m, "gray") for m in methods], alpha=0.85)
    bars2 = ax.bar(x + width/2, bwts, width, label="BWT (%)",
                   color=[METHOD_COLORS.get(m, "gray") for m in methods], alpha=0.4,
                   hatch="//", edgecolor="black")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylabel("Score (%)")
    ax.set_title("Average Accuracy (AA) and Backward Transfer (BWT)")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.3 if h >= 0 else -1.2),
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "aa_bwt_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: resource usage ───────────────────────────────────────────────────

def plot_resources(results: dict):
    methods = list(results.keys())
    times   = [np.mean(results[m]["task_times"])   for m in methods]
    rams    = [np.mean(results[m]["task_rams_mb"])  for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = [METHOD_COLORS.get(m, "gray") for m in methods]
    labels = [METHOD_LABELS.get(m, m)      for m in methods]

    ax1.bar(labels, times, color=colors, alpha=0.85)
    ax1.set_ylabel("Seconds")
    ax1.set_title("Mean Wall-Clock Time per Task")
    for i, v in enumerate(times):
        ax1.text(i, v + 0.1, f"{v:.1f}s", ha="center", fontsize=9)

    ax2.bar(labels, rams, color=colors, alpha=0.85)
    ax2.set_ylabel("MB")
    ax2.set_title("Mean RAM Δ per Task")
    for i, v in enumerate(rams):
        ax2.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "resource_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: per-task forgetting curves ─────────────────────────────────────

def plot_forgetting_curves(results: dict):
    """
    For each task t, show how accuracy on that task evolves as subsequent tasks are trained.
    """
    NUM_TASKS = 5
    fig, axes = plt.subplots(1, NUM_TASKS, figsize=(14, 4), sharey=True)

    for task_id in range(NUM_TASKS):
        ax = axes[task_id]
        for m, data in results.items():
            mat = np.array(data["acc_matrix"]) * 100
            # Accuracy on task_id after training tasks task_id..T-1
            curve = [mat[i, task_id] for i in range(task_id, NUM_TASKS)]
            x     = list(range(task_id + 1, NUM_TASKS + 1))
            ax.plot(x, curve, marker="o", color=METHOD_COLORS.get(m, "gray"),
                    label=METHOD_LABELS.get(m, m))

        ax.set_title(f"Task {task_id+1}")
        ax.set_xlabel("Tasks trained so far")
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        if task_id == 0:
            ax.set_ylabel("Accuracy (%)")
            ax.legend(fontsize=7)

    fig.suptitle("Per-Task Accuracy Over Training (forgetting curves)", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "task_accuracy_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    results = load_results()
    print(f"Found methods: {list(results.keys())}\n")

    print("Generating figures...")
    for m, data in results.items():
        plot_acc_matrix(m, data)

    if len(results) > 1:
        plot_aa_bwt(results)
        plot_resources(results)
        plot_forgetting_curves(results)
    else:
        m, data = next(iter(results.items()))
        plot_aa_bwt(results)
        plot_resources(results)

    print(f"\nAll figures saved to ./{FIGURES_DIR}/")


if __name__ == "__main__":
    main()
