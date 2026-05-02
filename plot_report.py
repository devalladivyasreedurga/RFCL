"""
Report Plots — generates all figures needed for the report.
Run after all experiments are complete (including core50).

Plots generated:
  1. AA + BWT comparison bar chart (all ViT methods)
  2. Accuracy matrix heatmaps (Naive vs ADAM)
  3. Per-task forgetting line chart (Task 1 accuracy as more tasks trained)
  4. Version progression chart (v1 -> v2 -> v3 for LwF and EWC)
  5. ResNet-18 vs ViT backbone comparison
  6. CORe50 results (Naive vs ADAM)

All plots saved to results/report_plots/
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join("results", "report_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# LOAD RESULTS
# ──────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    return json.load(open(path))

results = {
    # ResNet-18
    "Naive (ResNet)":       load("results/naive.json"),
    "EWC (ResNet)":         load("results/ewc.json"),
    "LwF (ResNet)":         load("results/lwf.json"),
    "Hybrid (ResNet)":      load("results/hybrid.json"),
    "PASS (ResNet)":        load("results/pass_resnet.json"),
    "ADAM (ResNet)":        load("results/adam_resnet.json"),
    # ViT v1
    "Naive (ViT v1)":   load("results/naive_vit.json"),
    "EWC (ViT v1)":     load("results/ewc_vit.json"),
    "LwF (ViT v1)":     load("results/lwf_vit.json"),
    "ADAM":             load("results/adam_vit.json"),
    # ViT v2
    "Naive (ViT v2)":   load("version2VIT/results/naive_v2.json"),
    "EWC (ViT v2)":     load("version2VIT/results/ewc_v2.json"),
    "LwF (ViT v2)":     load("version2VIT/results/lwf_v2.json"),
    "Hybrid (ViT v2)":  load("version2VIT/results/hybrid_v2.json"),
    # ViT v3
    "Naive (ViT v3)":   load("version3VIT/results/naive_v3.json"),
    "EWC (ViT v3)":     load("version3VIT/results/ewc_v3.json"),
    "LwF (ViT v3)":     load("version3VIT/results/lwf_v3.json"),
    "Hybrid (ViT v3)":  load("version3VIT/results/hybrid_v3.json"),
    # CORe50
    "Naive (CORe50)":   load("core50_eval/results/naive_core50.json"),
    "ADAM (CORe50)":    load("core50_eval/results/adam_core50.json"),
}

def get(name):
    return results.get(name)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

# ──────────────────────────────────────────────
# PLOT 1: AA + BWT — ViT methods comparison
# ──────────────────────────────────────────────
def plot_aa_bwt():
    methods = {
        "Naive\n(ViT v1)":   get("Naive (ViT v1)"),
        "EWC\n(ViT v1)":     get("EWC (ViT v1)"),
        "LwF\n(ViT v1)":     get("LwF (ViT v1)"),
        "LwF\n(ViT v2)":     get("LwF (ViT v2)"),
        "LwF\n(ViT v3)":     get("LwF (ViT v3)"),
        "ADAM":              get("ADAM"),
    }
    methods = {k: v for k, v in methods.items() if v is not None}

    names  = list(methods.keys())
    aa     = [methods[n]["aa"]  for n in names]
    bwt    = [methods[n]["bwt"] for n in names]
    x      = np.arange(len(names))
    w      = 0.35

    colors_aa  = ["#d62728" if "Naive" in n else
                  "#ff7f0e" if "EWC"   in n else
                  "#9467bd" if "LwF"   in n else
                  "#2ca02c" for n in names]
    colors_bwt = [c + "99" for c in colors_aa]   # semi-transparent

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w/2, aa,  width=w, color=colors_aa,  label="AA (%)",  alpha=0.9, edgecolor="white")
    bars2 = ax2.bar(x + w/2, bwt, width=w, color=colors_bwt, label="BWT (%)", alpha=0.75, edgecolor="white", hatch="//")

    ax1.set_ylabel("Average Accuracy — AA (%)", fontsize=11)
    ax2.set_ylabel("Backward Transfer — BWT (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(-100, 10)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
                 f"{bar.get_height():.1f}", ha="center", va="top", fontsize=8)

    ax1.set_title("Average Accuracy & Backward Transfer — ViT Methods on Split-CIFAR-100",
                  fontsize=12, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    save(fig, "1_aa_bwt_comparison.png")

# ──────────────────────────────────────────────
# PLOT 2: Accuracy Matrix Heatmaps — Naive vs ADAM
# ──────────────────────────────────────────────
def plot_heatmaps():
    pairs = [
        ("Naive (ViT v1)", "Naive"),
        ("ADAM",           "ADAM"),
    ]
    pairs = [(k, label) for k, label in pairs if get(k) is not None]
    if not pairs:
        print("Skipping heatmaps — no data")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, pairs):
        mat = np.array(get(key)["acc_matrix"]) * 100
        T   = mat.shape[0]

        # Mask upper triangle (not yet trained)
        masked = np.ma.masked_where(mat == 0, mat)
        im = ax.imshow(masked, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels([f"Task {i+1}" for i in range(T)], fontsize=9)
        ax.set_yticklabels([f"After\nTask {i+1}" for i in range(T)], fontsize=9)
        ax.set_xlabel("Evaluated on", fontsize=10)
        ax.set_ylabel("Trained up to", fontsize=10)
        ax.set_title(f"{label}\nAA: {get(key)['aa']:.1f}%  BWT: {get(key)['bwt']:.1f}%",
                     fontsize=11, fontweight="bold")

        for i in range(T):
            for j in range(T):
                if mat[i, j] > 0:
                    ax.text(j, i, f"{mat[i,j]:.0f}%", ha="center", va="center",
                            fontsize=9, color="black" if mat[i,j] > 40 else "white",
                            fontweight="bold")

        plt.colorbar(im, ax=ax, label="Accuracy (%)")

    fig.suptitle("Accuracy Matrix: How accuracy on each task changes over training",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "2_accuracy_heatmaps.png")

# ──────────────────────────────────────────────
# PLOT 3: Forgetting curves — all tasks, all ViT methods
# Each line = one task's accuracy tracked over training
# Side-by-side: Naive vs ADAM
# ──────────────────────────────────────────────
def plot_forgetting_curve():
    naive = get("Naive (ViT v1)")
    adam  = get("ADAM")
    if naive is None or adam is None:
        print("Skipping forgetting curve — missing data")
        return

    task_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    T = 5

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (title, data) in zip(axes, [
        ("Naive Fine-Tuning (ViT)", naive),
        ("ADAM — ViT + Adapters + NCM", adam),
    ]):
        mat = np.array(data["acc_matrix"]) * 100
        aa  = data["aa"]
        bwt = data["bwt"]

        for task_idx in range(T):
            x_vals = list(range(task_idx + 1, T + 1))
            y_vals = [mat[step, task_idx] for step in range(task_idx, T)]
            ax.plot(x_vals, y_vals, marker="o", linewidth=2.2, markersize=7,
                    label=f"Task {task_idx + 1}", color=task_colors[task_idx])

        ax.set_title(f"{title}\nAA: {aa:.1f}%   BWT: {bwt:.1f}%",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Training progressed to task", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_xticks(range(1, T + 1))
        ax.set_xticklabels([f"Task {i}" for i in range(1, T + 1)], fontsize=9)
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("Forgetting Curves: How Each Task's Accuracy Changes as Training Progresses",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "3_forgetting_curve.png")

# ──────────────────────────────────────────────
# PLOT 4: Version progression (LwF v1 -> v2 -> v3, EWC v1 -> v3)
# ──────────────────────────────────────────────
def plot_version_progression():
    groups = {
        "LwF": [
            ("v1\n(basic KD)",         get("LwF (ViT v1)")),
            ("v2\n(+WA +BiC)",         get("LwF (ViT v2)")),
            ("v3\n(adaptive+multi-T)", get("LwF (ViT v3)")),
        ],
        "EWC": [
            ("v1\n(standard)",          get("EWC (ViT v1)")),
            ("v3\n(class-selective)",   get("EWC (ViT v3)")),
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    colors = {"LwF": "#9467bd", "EWC": "#ff7f0e"}

    for ax, (method, versions) in zip(axes, groups.items()):
        versions = [(label, d) for label, d in versions if d is not None]
        if not versions:
            continue
        labels = [v[0] for v in versions]
        aa     = [v[1]["aa"]  for v in versions]
        bwt    = [v[1]["bwt"] for v in versions]
        x      = np.arange(len(labels))
        c      = colors[method]

        ax2 = ax.twinx()
        ax.bar(x - 0.18, aa,  width=0.35, color=c,       alpha=0.85, label="AA (%)",  edgecolor="white")
        ax2.bar(x + 0.18, bwt, width=0.35, color=c+"66", alpha=0.85, label="BWT (%)", edgecolor="white", hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("AA (%)", fontsize=10)
        ax2.set_ylabel("BWT (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax2.set_ylim(-100, 10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(f"{method} — Improvement Across Versions", fontsize=11, fontweight="bold")

        for i, v in enumerate(aa):
            ax.text(x[i] - 0.18, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5, color=c)
        for i, v in enumerate(bwt):
            ax2.text(x[i] + 0.18, v - 2, f"{v:.1f}%", ha="center", fontsize=8.5, color=c)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("Algorithm Improvements Across Versions — Split-CIFAR-100",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "4_version_progression.png")

# ──────────────────────────────────────────────
# PLOT 5: ResNet-18 vs ViT backbone
# ──────────────────────────────────────────────
def plot_backbone_comparison():
    pairs = [
        ("Naive",   get("Naive (ResNet)"),  get("Naive (ViT v1)")),
        ("EWC",     get("EWC (ResNet)"),    get("EWC (ViT v1)")),
        ("LwF",     get("LwF (ResNet)"),    get("LwF (ViT v1)")),
        ("Hybrid",  get("Hybrid (ResNet)"), get("ADAM")),
    ]
    pairs = [(n, r, v) for n, r, v in pairs if r is not None and v is not None]
    if not pairs:
        print("Skipping backbone comparison — missing data")
        return

    names   = [p[0] for p in pairs]
    resnet  = [p[1]["aa"] for p in pairs]
    vit     = [p[2]["aa"] for p in pairs]
    x       = np.arange(len(names))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, resnet, width=w, label="ResNet-18", color="#aec7e8", edgecolor="white")
    b2 = ax.bar(x + w/2, vit,    width=w, label="ViT-B/16",  color="#1f77b4", edgecolor="white")

    for bar in b1 + b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Average Accuracy — AA (%)", fontsize=11)
    ax.set_title("Backbone Comparison: ResNet-18 vs ViT-B/16\n(Last column: Hybrid ResNet vs ADAM ViT)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    fig.tight_layout()
    save(fig, "5_backbone_comparison.png")

# ──────────────────────────────────────────────
# PLOT 6: CORe50 Results
# ──────────────────────────────────────────────
def plot_core50():
    naive = get("Naive (CORe50)")
    adam  = get("ADAM (CORe50)")

    if naive is None or adam is None:
        print("Skipping CORe50 plot — results not yet available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: AA + BWT bar chart
    ax = axes[0]
    methods = {"Naive": naive, "ADAM": adam}
    colors  = {"Naive": "#d62728", "ADAM": "#2ca02c"}
    x = np.arange(2)
    aa_vals  = [naive["aa"],  adam["aa"]]
    bwt_vals = [naive["bwt"], adam["bwt"]]

    ax2 = ax.twinx()
    ax.bar(x - 0.18,  aa_vals,  width=0.35, color=[colors["Naive"], colors["ADAM"]],
           alpha=0.9, edgecolor="white", label="AA (%)")
    ax2.bar(x + 0.18, bwt_vals, width=0.35,
            color=[colors["Naive"]+"88", colors["ADAM"]+"88"],
            alpha=0.85, edgecolor="white", hatch="//", label="BWT (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(["Naive", "ADAM"], fontsize=11)
    ax.set_ylabel("AA (%)", fontsize=10)
    ax2.set_ylabel("BWT (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax2.set_ylim(-100, 10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("CORe50 NC: AA and BWT", fontsize=11, fontweight="bold")
    for i, v in enumerate(aa_vals):
        ax.text(x[i] - 0.18, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    for i, v in enumerate(bwt_vals):
        ax2.text(x[i] + 0.18, v - 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    # Right: Per-task final accuracy
    ax = axes[1]
    task_labels = [f"Task {i+1}\n(cls {i*10}-{i*10+9})" for i in range(5)]
    naive_last = [x * 100 for x in naive["acc_matrix"][-1]]
    adam_last  = [x * 100 for x in adam["acc_matrix"][-1]]
    xp = np.arange(5)

    ax.bar(xp - 0.2, naive_last, width=0.38, label="Naive", color="#d62728", alpha=0.85, edgecolor="white")
    ax.bar(xp + 0.2, adam_last,  width=0.38, label="ADAM",  color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.set_xticks(xp)
    ax.set_xticklabels(task_labels, fontsize=8.5)
    ax.set_ylabel("Accuracy after all 5 tasks (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("CORe50 NC: Per-Task Final Accuracy", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)

    fig.suptitle("CORe50 Evaluation — 5 Tasks x 10 Classes (NC Scenario)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "6_core50_results.png")

# ──────────────────────────────────────────────
# PLOT 7: Full ResNet comparison (all methods)
# ──────────────────────────────────────────────
def plot_resnet_full():
    methods = {
        "Naive":          get("Naive (ResNet)"),
        "EWC":            get("EWC (ResNet)"),
        "LwF":            get("LwF (ResNet)"),
        "Hybrid":         get("Hybrid (ResNet)"),
        "ADAM\n(ResNet)": get("ADAM (ResNet)"),
        "PASS\n(ResNet)": get("PASS (ResNet)"),
    }
    methods = {k: v for k, v in methods.items() if v is not None}
    if not methods:
        print("Skipping ResNet full comparison — missing data")
        return

    names = list(methods.keys())
    aa    = [methods[n]["aa"]  for n in names]
    bwt   = [methods[n]["bwt"] for n in names]
    x     = np.arange(len(names))
    w     = 0.35

    colors = {
        "Naive":          "#d62728",
        "EWC":            "#ff7f0e",
        "LwF":            "#9467bd",
        "Hybrid":         "#8c564b",
        "ADAM\n(ResNet)": "#1f77b4",
        "PASS\n(ResNet)": "#2ca02c",
    }

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    col_list = [colors.get(n, "#888") for n in names]
    bars1 = ax1.bar(x - w/2, aa,  width=w, color=col_list, alpha=0.9,  edgecolor="white", label="AA (%)")
    bars2 = ax2.bar(x + w/2, bwt, width=w, color=col_list, alpha=0.55, edgecolor="white", hatch="//", label="BWT (%)")

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=8)

    ax1.set_ylabel("Average Accuracy — AA (%)", fontsize=11)
    ax2.set_ylabel("Backward Transfer — BWT (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(-100, 10)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.set_title("ResNet-18: All Methods on Split-CIFAR-100\n"
                  "Softmax-head methods vs Prototype-based methods (ADAM, PASS)",
                  fontsize=11, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.tight_layout()
    save(fig, "7_resnet_full_comparison.png")

# ──────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating report plots...\n")
    plot_aa_bwt()
    plot_heatmaps()
    plot_forgetting_curve()
    plot_version_progression()
    plot_backbone_comparison()
    plot_core50()
    plot_resnet_full()
    print(f"\nAll plots saved to: {OUT_DIR}")
