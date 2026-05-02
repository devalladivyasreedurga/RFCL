"""
Flowchart: How ADAM works
ViT-B/16 + Per-task Adapters + NCM Classifier
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9)
ax.axis("off")

# ── helper functions ──────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel=None, color="#4C72B0", textcolor="white", style="round,pad=0.1", fontsize=10):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle=style, linewidth=1.5,
                            edgecolor="white", facecolor=color, zorder=3)
    ax.add_patch(fancy)
    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=7.5, color=textcolor, alpha=0.9, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)

def arrow(ax, x1, y1, x2, y2, label=None, color="#555555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.12, my, label, fontsize=8, color=color, va="center")

def dashed_box(ax, x, y, w, h, label, color="#AAAAAA"):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.1", linewidth=1.2,
                            edgecolor=color, facecolor="#F5F5F5",
                            linestyle="--", zorder=3)
    ax.add_patch(fancy)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, color="#555555", fontweight="bold", zorder=4)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(6.5, 8.65, "ADAM: Adapter-based Continual Learning with NCM Classifier",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

# ── Column positions ──────────────────────────────────────────────────────────
# Main pipeline runs left to right

# 1. Input
box(ax, 1.0, 7.2, 1.6, 0.7, "Input Image", "224 x 224 x 3", color="#555577")
arrow(ax, 1.8, 7.2, 2.5, 7.2)

# 2. Frozen ViT
box(ax, 3.3, 7.2, 1.5, 0.7, "ViT-B/16", "Frozen backbone", color="#1f77b4")
ax.text(3.3, 6.6, "No gradient updates", ha="center", fontsize=7.5, color="#1f77b4", style="italic")
arrow(ax, 4.1, 7.2, 4.8, 7.2, label="768-dim")

# 3. Feature split
ax.plot([5.1, 5.1], [6.0, 7.2], color="#888", lw=1.5, zorder=2)  # vertical line
ax.plot([5.1, 5.5], [7.2, 7.2], color="#888", lw=1.5, zorder=2)  # to base
ax.plot([5.1, 5.5], [6.6, 6.6], color="#888", lw=1.5, zorder=2)  # to adapter1
ax.plot([5.1, 5.5], [6.0, 6.0], color="#888", lw=1.5, zorder=2)  # to adapter2

# Base features
box(ax, 6.2, 7.2, 1.3, 0.55, "Base Features", "768-dim", color="#2ca02c")

# Adapters
box(ax, 6.2, 6.6, 1.3, 0.55, "Adapter 1", "MLP  128-dim", color="#d62728")
box(ax, 6.2, 6.0, 1.3, 0.55, "Adapter 2", "MLP  128-dim", color="#d62728")

# Dots indicating more adapters
ax.text(6.2, 5.55, ". . .", ha="center", va="center", fontsize=14, color="#d62728")

box(ax, 6.2, 5.1, 1.3, 0.55, "Adapter N", "MLP  128-dim", color="#d62728", style="round,pad=0.1")

# Adapter label
ax.text(6.2, 4.65, "One per task (frozen after training)", ha="center",
        fontsize=7.5, color="#d62728", style="italic")

# Arrows to concat
for y_src in [7.2, 6.6, 6.0, 5.1]:
    arrow(ax, 6.85, y_src, 7.55, 6.15)

# 4. Concatenate
box(ax, 8.1, 6.15, 1.1, 2.0, "Concat\n&\nExpand", color="#8c564b", fontsize=9)
ax.text(8.1, 5.05, "768 + N x 128\ndim features", ha="center", fontsize=7.5,
        color="#8c564b", style="italic")

arrow(ax, 8.65, 6.15, 9.2, 6.15)

# 5. Tukey Transform
box(ax, 9.9, 6.15, 1.2, 0.65, "Tukey\nTransform", color="#7f7f7f", fontsize=9)
ax.text(9.9, 5.72, "relu + pow(0.5)\nL2 normalize", ha="center", fontsize=7.5,
        color="#7f7f7f", style="italic")

arrow(ax, 10.5, 6.15, 11.1, 6.15)

# 6. NCM Classifier
box(ax, 11.9, 6.15, 1.5, 0.7, "NCM Classifier", "Nearest prototype", color="#17becf", fontsize=9)

arrow(ax, 11.9, 5.8, 11.9, 5.2)

# 7. Output
box(ax, 11.9, 4.85, 1.4, 0.6, "Predicted Class", color="#2ca02c", fontsize=9)

# ── Prototype store (bottom panel) ───────────────────────────────────────────
ax.text(6.5, 4.15, "Prototype Memory (stored per class — never overwritten)",
        ha="center", va="center", fontsize=9, fontweight="bold", color="#17becf")

proto_colors = ["#ffd700", "#ffa07a", "#98fb98", "#87cefa", "#dda0dd"]
proto_labels = ["Task 1\nclasses", "Task 2\nclasses", "Task 3\nclasses",
                "Task 4\nclasses", "Task 5\nclasses"]
for i, (c, l) in enumerate(zip(proto_colors, proto_labels)):
    bx = 4.0 + i * 1.75
    fancy = FancyBboxPatch((bx - 0.65, 3.25), 1.3, 0.65,
                            boxstyle="round,pad=0.08", facecolor=c,
                            edgecolor="#aaa", linewidth=1.0, zorder=3)
    ax.add_patch(fancy)
    ax.text(bx, 3.58, l, ha="center", va="center", fontsize=7.5,
            color="#333", fontweight="bold", zorder=4)

# Arrow from NCM to prototype store
ax.annotate("", xy=(11.9, 3.9), xytext=(11.9, 4.5),
            arrowprops=dict(arrowstyle="<-", color="#17becf", lw=1.5,
                            linestyle="dashed"), zorder=2)
ax.text(12.05, 4.2, "compare\ndistances", fontsize=7.5, color="#17becf", va="center")

# ── Key insight box ───────────────────────────────────────────────────────────
# insight = FancyBboxPatch((0.3, 0.3), 12.4, 2.55,
#                           boxstyle="round,pad=0.15", facecolor="#f0f8ff",
#                           edgecolor="#1f77b4", linewidth=1.5, zorder=1)
# ax.add_patch(insight)
# ax.text(6.5, 2.6, "Why ADAM does not forget", ha="center", fontsize=10,
#         fontweight="bold", color="#1f77b4")
# ax.text(6.5, 2.1,
#         "1.  Backbone is frozen — shared features never change",
#         ha="center", fontsize=9, color="#333")
# ax.text(6.5, 1.7,
#         "2.  Each task gets its own adapter — old adapters are frozen after training",
#         ha="center", fontsize=9, color="#333")
# ax.text(6.5, 1.3,
#         "3.  NCM uses stored prototype averages — no weight updates erase old classes",
#         ha="center", fontsize=9, color="#333")
# ax.text(6.5, 0.85,
#         "Result:  new tasks add capacity without touching anything learned before",
#         ha="center", fontsize=9.5, color="#1f77b4", fontweight="bold")

plt.tight_layout()
out = "results/adam_flowchart.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
plt.show()
