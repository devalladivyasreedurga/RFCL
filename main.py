"""
main.py — Run one or all continual learning methods on Split-CIFAR-100.

Usage:
    python main.py --method naive
    python main.py --method ewc
    python main.py --method lwf
    python main.py --method hybrid
    python main.py --method all        # runs all four, prints comparison table
"""

import argparse
import numpy as np
import torch

from data import get_cifar100_tasks

# Import each method's run() function
from methods.naive  import run as run_naive
from methods.ewc    import run as run_ewc
from methods.lwf    import run as run_lwf
from methods.hybrid import run as run_hybrid

RUNNERS = {
    "naive":  run_naive,
    "ewc":    run_ewc,
    "lwf":    run_lwf,
    "hybrid": run_hybrid,
}


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def print_comparison(all_results: dict):
    print(f"\n{'='*58}")
    print("  Comparison Table")
    print(f"{'='*58}")
    print(f"  {'Method':<10} {'AA (%)':>8} {'BWT (%)':>9} {'Avg time/task':>15}")
    print(f"  {'-'*52}")
    for method, r in all_results.items():
        avg_t = round(float(np.mean(r["task_times"])), 1)
        print(f"  {method:<10} {r['aa']:>8.2f} {r['bwt']:>9.2f} {avg_t:>13.1f}s")
    print()
    best_aa  = max(all_results, key=lambda m: all_results[m]["aa"])
    best_bwt = max(all_results, key=lambda m: all_results[m]["bwt"])
    print(f"  Best AA:  {best_aa.upper()}  ({all_results[best_aa]['aa']:.2f}%)")
    print(f"  Best BWT: {best_bwt.upper()}  ({all_results[best_bwt]['bwt']:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default="all",
        choices=list(RUNNERS.keys()) + ["all"],
        help="Method to run (default: all)"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("Loading data...")
    tasks = get_cifar100_tasks(num_tasks=5)

    methods = list(RUNNERS.keys()) if args.method == "all" else [args.method]

    all_results = {}
    for m in methods:
        all_results[m] = RUNNERS[m](tasks=tasks, device=device)

    if len(methods) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
