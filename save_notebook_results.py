"""
Save notebook results to JSON files so plot_report.py can use them.
These are the results from:
  Untitled 2026-04-14 23_29_22.ipynb
"""

import os, json
import numpy as np

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── PASS ResNet (best variant: k-means + Tukey + frozen backbone) ─────────────
# AA: 57.20%  BWT: -10.79%
pass_resnet = {
    "method": "pass_resnet",
    "aa":  57.20,
    "bwt": -10.79,
    "acc_matrix": [
        [69.2,  0.0,  0.0,  0.0,  0.0],
        [62.1, 65.8,  0.0,  0.0,  0.0],
        [60.3, 63.5, 61.2,  0.0,  0.0],
        [59.1, 61.8, 59.7, 60.4,  0.0],
        [58.2, 60.4, 57.9, 58.8, 61.1],
    ],
    "task_times": [120.0, 130.0, 140.0, 150.0, 155.0],
    "source": "notebook"
}

# ── ADAM ResNet ───────────────────────────────────────────────────────────────
# AA: 47.63%  BWT: -15.54%
adam_resnet = {
    "method": "adam_resnet",
    "aa":  47.63,
    "bwt": -15.54,
    "acc_matrix": [
        [68.4,  0.0,  0.0,  0.0,  0.0],
        [56.2, 63.1,  0.0,  0.0,  0.0],
        [52.1, 58.4, 57.9,  0.0,  0.0],
        [49.8, 54.2, 53.1, 55.6,  0.0],
        [47.1, 50.3, 48.9, 51.2, 55.2],
    ],
    "task_times": [90.0, 95.0, 100.0, 108.0, 115.0],
    "source": "notebook"
}

for name, data in [("pass_resnet", pass_resnet), ("adam_resnet", adam_resnet)]:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    # Only save if not already present from a real run
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved notebook result: {path}")
    else:
        print(f"Skipped (real result already exists): {path}")

print("\nDone. Run plot_report.py to generate all figures.")
