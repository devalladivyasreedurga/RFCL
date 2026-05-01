"""
Plot: Per-task accuracy after training all 5 tasks
Compares baseline ViT methods vs ADAM (ViT with adapters)
Shows clearly how shared mutable head causes forgetting vs adapter approach
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load data ──────────────────────────────────────────────────────────────────
files = {
    "Naive":  "results/naive_vit.json",
    "EWC":    "results/ewc_vit.json",
    "LwF":    "results/lwf_vit.json",
    "ADAM\n(ViT + Adapters)": "results/adam_vit.json",
}

data = {}
for name, path in files.items():
    d = json.load(open(path))
    data[name] = {
        "last_row": [x * 100 for x in d["acc_matrix"][-1]],
        "aa":  d["aa"],
        "bwt": d["bwt"],
    }

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

task_labels  = [f"Task {i+1}\n(classes {i*20}–{i*20+19})" for i in range(5)]
x            = np.arange(5)
n_methods    = len(data)
bar_width    = 0.18
offsets      = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * bar_width

colors = {
    "Naive":               "#d62728",   # red
    "EWC":                 "#ff7f0e",   # orange
    "LwF":                 "#9467bd",   # purple
    "ADAM\n(ViT + Adapters)": "#2ca02c",  # green
}

for i, (name, info) in enumerate(data.items()):
    bars = ax.bar(
        x + offsets[i],
        info["last_row"],
        width=bar_width,
        label=f"{name.replace(chr(10), ' ')}  (AA {info['aa']}%)",
        color=colors[name],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )

# ── Annotations ────────────────────────────────────────────────────────────────
ax.set_xlabel("Task (class range)", fontsize=12)
ax.set_ylabel("Accuracy after all 5 tasks (%)", fontsize=12)
ax.set_title("Per-Task Accuracy After Full Training\nViT Baseline Methods vs. ADAM (ViT + Adapters)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=9)
ax.set_ylim(0, 105)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Shade early tasks to emphasise forgetting zone
ax.axvspan(-0.5, 3.5, alpha=0.04, color="red", label="_nolegend_")
ax.text(1.5, 99, "Forgetting zone\n(early tasks)", ha="center", fontsize=8,
        color="red", alpha=0.7)

ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

# Add horizontal line at ADAM average
adam_avg = data["ADAM\n(ViT + Adapters)"]["aa"]
ax.axhline(adam_avg, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.6)
ax.text(4.6, adam_avg + 1.5, f"ADAM avg\n{adam_avg}%", color="#2ca02c",
        fontsize=8, ha="right")

plt.tight_layout()
out = "results/vit_adapter_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
