"""
Heuristic threshold sensitivity analysis for TopoAdamW.

Sweeps:
  (a) flat_sharpness_thresh ∈ [0.05, 0.075, 0.10, 0.15, 0.20]
  (b) accel_factor           ∈ [1.05, 1.10, 1.15, 1.20, 1.25]

Uses 5000-sample CIFAR-10 subset, 15 epochs, 3 seeds.
Saves results + assets/ablation_threshold.png.
Checkpoints to threshold_ablation_results.json.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from topoadamw import TopoAdamW
from models import CifarNet

CHECKPOINT = "threshold_ablation_results.json"
SEEDS = [0, 1, 2]
N_SUBSET = 5000
EPOCHS   = 15
LR       = 1e-3
WARMUP   = 50
INTERVAL = 20

# Sweep ranges
FLAT_SHARP_THRESH_SWEEP = [0.05, 0.075, 0.10, 0.15, 0.20]
ACCEL_FACTOR_SWEEP      = [1.05, 1.10, 1.15, 1.20, 1.25]


# ── Configurable heuristic wrapper ────────────────────────────────────────────

class ThresholdTopoAdamW(TopoAdamW):
    """TopoAdamW with configurable heuristic thresholds."""

    def __init__(self, *args,
                 flat_sharp_thresh=0.10,
                 flat_var_thresh=0.30,
                 sharp_thresh_high=0.50,
                 var_thresh_high=0.50,
                 sharp_thresh_med=0.20,
                 accel_factor=1.15,
                 decel_high=0.80,
                 decel_var=0.85,
                 decel_med=0.95,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._fst  = flat_sharp_thresh
        self._fvt  = flat_var_thresh
        self._sth  = sharp_thresh_high
        self._vth  = var_thresh_high
        self._stm  = sharp_thresh_med
        self._af   = accel_factor
        self._dh   = decel_high
        self._dv   = decel_var
        self._dm   = decel_med

    def _geometric_heuristic(self, features):
        s, v = features["sharpness"], features["variance"]
        if s < self._fst and v < self._fvt:
            return self._af
        elif s > self._sth:
            return self._dh
        elif v > self._vth:
            return self._dv
        elif s > self._stm:
            return self._dm
        else:
            return 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(data_dir, batch_size, num_workers, n_subset):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    os.makedirs(data_dir, exist_ok=True)
    pin = torch.cuda.is_available()
    full_train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    subset     = Subset(full_train, list(range(n_subset)))
    return (DataLoader(subset,   batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(test_set, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin))


def train_epoch(model, optimizer, loader, criterion, device):
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if isinstance(optimizer, TopoAdamW):
            optimizer.step(data=x, target=y, criterion=criterion)
        else:
            optimizer.step()
        total_loss += loss.item() * len(x)
        total += len(x)
    return total_loss / total


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total   += len(x)
    return correct / total


def run_one(make_opt, seed, train_loader, test_loader, criterion, device):
    set_seed(seed)
    model = CifarNet(10).to(device)
    optimizer = make_opt(model)
    for _ in range(EPOCHS):
        train_epoch(model, optimizer, train_loader, criterion, device)
    return evaluate(model, test_loader, criterion, device)


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2)


def run_sweep(label, param_name, param_values, make_opt_fn,
              train_loader, test_loader, criterion, device, ckpt):
    results = {}
    for val in param_values:
        accs = []
        for seed in SEEDS:
            key = f"{label}/{param_name}={val}/seed={seed}"
            if key in ckpt:
                print(f"  [SKIP] {key}")
                accs.append(ckpt[key])
                continue
            acc = run_one(
                lambda model, v=val: make_opt_fn(model, v),
                seed, train_loader, test_loader, criterion, device
            )
            ckpt[key] = acc
            save_checkpoint(ckpt)
            accs.append(acc)
            print(f"  {key}  acc={acc:.4f}")
        results[val] = accs
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device} | Subset: {N_SUBSET} | Epochs: {EPOCHS} | Seeds: {SEEDS}")

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.num_workers, N_SUBSET)
    criterion = nn.CrossEntropyLoss()
    ckpt = load_checkpoint()

    # AdamW baseline
    adamw_accs = []
    for seed in SEEDS:
        key = f"AdamW/baseline/seed={seed}"
        if key in ckpt:
            print(f"  [SKIP] {key}")
            adamw_accs.append(ckpt[key])
        else:
            def make_adamw(model):
                return optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
            acc = run_one(make_adamw, seed, train_loader, test_loader, criterion, device)
            ckpt[key] = acc
            save_checkpoint(ckpt)
            adamw_accs.append(acc)
            print(f"  AdamW baseline seed={seed} acc={acc:.4f}")

    adamw_mean = np.mean(adamw_accs)
    adamw_std  = np.std(adamw_accs, ddof=1)
    print(f"\nAdamW baseline: {adamw_mean:.4f} ± {adamw_std:.4f}")

    # ── Sweep A: flat_sharpness_thresh ────────────────────────────────────────
    print("\n=== Sweep A: flat_sharpness_thresh ===")
    def make_flat_thresh(model, fst):
        return ThresholdTopoAdamW(
            model.parameters(), model,
            lr=LR, weight_decay=5e-4,
            interval=INTERVAL, warmup_steps=WARMUP,
            max_lr_ratio=1.0, min_lr_ratio=0.2,
            flat_sharp_thresh=fst,
        )

    sweep_a = run_sweep(
        "SweepA", "flat_sharp_thresh", FLAT_SHARP_THRESH_SWEEP,
        make_flat_thresh, train_loader, test_loader, criterion, device, ckpt
    )

    # ── Sweep B: accel_factor ─────────────────────────────────────────────────
    print("\n=== Sweep B: accel_factor ===")
    def make_accel(model, af):
        return ThresholdTopoAdamW(
            model.parameters(), model,
            lr=LR, weight_decay=5e-4,
            interval=INTERVAL, warmup_steps=WARMUP,
            max_lr_ratio=1.0, min_lr_ratio=0.2,
            accel_factor=af,
        )

    sweep_b = run_sweep(
        "SweepB", "accel_factor", ACCEL_FACTOR_SWEEP,
        make_accel, train_loader, test_loader, criterion, device, ckpt
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    def print_sweep(name, param_name, sweep, default_val):
        print(f"\n{'='*55}")
        print(f"{name} — varying {param_name} (default={default_val})")
        print(f"{'AdamW baseline':>25}: {adamw_mean:.4f} ± {adamw_std:.4f}")
        for val, accs in sweep.items():
            marker = " ← default" if abs(val - default_val) < 1e-9 else ""
            print(f"  {param_name}={val:6.3f}: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}{marker}")

    print_sweep("Sweep A", "flat_sharp_thresh", sweep_a, 0.10)
    print_sweep("Sweep B", "accel_factor",       sweep_b, 1.15)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def plot_sweep(ax, sweep, param_vals, default_val, xlabel, title):
        means = [np.mean(sweep[v]) for v in param_vals]
        stds  = [np.std(sweep[v], ddof=1) for v in param_vals]
        ax.errorbar(param_vals, means, yerr=stds,
                    fmt="o-", color="#2196F3", linewidth=1.5,
                    capsize=4, label="TopoAdamW")
        ax.axhline(adamw_mean, color="#555", linestyle="--", linewidth=1.2,
                   label="AdamW baseline")
        ax.fill_between([min(param_vals), max(param_vals)],
                        adamw_mean - adamw_std, adamw_mean + adamw_std,
                        color="#555", alpha=0.1)
        ax.axvline(default_val, color="#2196F3", linestyle=":", alpha=0.6,
                   label=f"Default ({default_val})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final Val Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_sweep(axes[0], sweep_a, FLAT_SHARP_THRESH_SWEEP, 0.10,
               "flat_sharpness_thresh",
               "(a) Flat-landscape sharpness threshold")
    plot_sweep(axes[1], sweep_b, ACCEL_FACTOR_SWEEP, 1.15,
               "accel_factor",
               "(b) Acceleration factor ρ (flat regime)")

    fig.suptitle(
        f"Threshold Sensitivity — CIFAR-10 subset ({N_SUBSET} samples, "
        f"{EPOCHS} epochs, {len(SEEDS)} seeds)",
        fontsize=11
    )
    fig.tight_layout()

    os.makedirs("assets", exist_ok=True)
    out = "assets/ablation_threshold.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
