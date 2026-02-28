"""
Ablation study for TopoAdamW on CIFAR-10.

Ablations run:
  1. Probe density  — n_samples ∈ {0, 4, 8, 16, 32}
  2. Probe interval — T ∈ {10, 25, 50, 100, 200}

Results saved to:
  ablation_results.json          raw per-seed histories
  assets/ablation_density.png    bar + curve plot
  assets/ablation_interval.png   bar + curve plot

Usage:
  python ablation_study.py                        # default: 3 seeds, 20 epochs, 5000 subset
  python ablation_study.py --seeds 1 --epochs 30  # single seed, full 30 epochs
  python ablation_study.py --no-subset            # full 50 000-sample dataset (slow)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from topoadamw import TopoAdamW
from models import CifarNet


# ── Defaults ─────────────────────────────────────────────────────────────────
# Hyperparameters are scaled for a 5000-sample subset (40 batches/epoch).
# Full-dataset defaults in parentheses:
#   warmup_steps: 50  (vs 150 — keeps warmup at ~1 epoch regardless of dataset size)
#   interval:     20  (vs 50  — keeps ~2 probes/epoch with 40 batches)
BASE_CFG = dict(
    lr=1e-3,
    weight_decay=5e-4,
    warmup_steps=50,
    max_lr_ratio=1.0,
    min_lr_ratio=0.2,
    span=0.12,
    grid_size=15,
    # default probe density and interval (used as the "reference" point)
    n_samples=16,
    interval=20,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Data ─────────────────────────────────────────────────────────────────────
def build_loaders(data_dir: str, batch_size: int, subset: int, num_workers: int):
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10("data", train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)

    if subset > 0:
        train_set = Subset(train_set, list(range(min(subset, len(train_set)))))
        test_set  = Subset(test_set,  list(range(min(subset // 5, len(test_set)))))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, optimizer, loader, criterion, device):
    model.train()
    total_loss, total = 0.0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        if isinstance(optimizer, TopoAdamW):
            optimizer.step(data=data, target=target, criterion=criterion)
        else:
            optimizer.step()
        total_loss += loss.item() * len(data)
        total += len(data)
    return total_loss / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out  = model(data)
            loss = criterion(out, target)
            total_loss += loss.item() * len(data)
            correct    += out.argmax(1).eq(target).sum().item()
            total      += len(data)
    return total_loss / total, correct / total


def run_one(label, make_opt_fn, train_loader, test_loader,
            device, epochs, seed, verbose=True):
    set_seed(seed)
    model     = CifarNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_opt_fn(model)

    val_accs, val_losses, train_losses = [], [], []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl = train_epoch(model, optimizer, train_loader, criterion, device)
        vl, va = evaluate(model, test_loader, criterion, device)
        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)
        if verbose:
            print(f"  [{label}] seed={seed} ep={ep}/{epochs} "
                  f"val_acc={va:.4f} val_loss={vl:.4f} "
                  f"({time.time()-t0:.0f}s elapsed)")

    return {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accs}


# ── Optimizer factories ───────────────────────────────────────────────────────
def make_adamw(cfg):
    def factory(model):
        return optim.AdamW(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return factory


def make_topoadamw(cfg, n_samples=None, interval=None):
    n  = n_samples if n_samples is not None else cfg["n_samples"]
    iv = interval  if interval  is not None else cfg["interval"]
    def factory(model):
        return TopoAdamW(
            model.parameters(), model,
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            interval=iv,
            warmup_steps=cfg["warmup_steps"],
            max_lr_ratio=cfg["max_lr_ratio"],
            min_lr_ratio=cfg["min_lr_ratio"],
            probe_kwargs={"grid_size": cfg["grid_size"],
                          "span": cfg["span"],
                          "n_samples": n},
        )
    return factory


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_ablation(results_by_label, baseline_label, param_name,
                  param_values, out_path, title):
    """
    results_by_label : {label: {"mean_acc": [...], "std_acc": [...],
                                "final_acc_mean": float, "final_acc_std": float}}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: val accuracy curves
    ax = axes[0]
    baseline = results_by_label.get(baseline_label)
    if baseline:
        epochs = range(1, len(baseline["mean_acc"]) + 1)
        ax.plot(epochs, baseline["mean_acc"], "k--",
                label=baseline_label, linewidth=1.5)
        if baseline["std_acc"] is not None:
            lo = [m - s for m, s in zip(baseline["mean_acc"], baseline["std_acc"])]
            hi = [m + s for m, s in zip(baseline["mean_acc"], baseline["std_acc"])]
            ax.fill_between(epochs, lo, hi, alpha=0.15, color="k")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(param_values)))
    for (label, res), color in zip(
            [(l, r) for l, r in results_by_label.items() if l != baseline_label],
            colors):
        epochs = range(1, len(res["mean_acc"]) + 1)
        ax.plot(epochs, res["mean_acc"], label=label, color=color)
        if res["std_acc"] is not None:
            lo = [m - s for m, s in zip(res["mean_acc"], res["std_acc"])]
            hi = [m + s for m, s in zip(res["mean_acc"], res["std_acc"])]
            ax.fill_between(epochs, lo, hi, alpha=0.12, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Validation Accuracy Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: bar chart of final accuracy
    ax = axes[1]
    labels = [l for l in results_by_label if l != baseline_label]
    means  = [results_by_label[l]["final_acc_mean"] for l in labels]
    stds   = [results_by_label[l]["final_acc_std"]  for l in labels]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=plt.cm.viridis(np.linspace(0.2, 0.9, len(labels))),
                  alpha=0.85)

    if baseline:
        bm = baseline["final_acc_mean"]
        ax.axhline(bm, color="k", linestyle="--", linewidth=1.2,
                   label=f"{baseline_label} ({bm:.4f})")
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Final Val Accuracy")
    ax.set_title(f"Final Accuracy vs {param_name}")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{m:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(histories):
    """histories: list of {"val_acc": [...], ...} dicts, one per seed."""
    if len(histories) == 1:
        accs = histories[0]["val_acc"]
        return {
            "mean_acc":      accs,
            "std_acc":       None,
            "final_acc_mean": accs[-1],
            "final_acc_std":  0.0,
        }
    # Pad to same length (in case of early stopping)
    max_ep = max(len(h["val_acc"]) for h in histories)
    padded = [h["val_acc"] + [h["val_acc"][-1]] * (max_ep - len(h["val_acc"]))
              for h in histories]
    arr = np.array(padded)
    return {
        "mean_acc":       arr.mean(0).tolist(),
        "std_acc":        arr.std(0).tolist(),
        "final_acc_mean": float(arr[:, -1].mean()),
        "final_acc_std":  float(arr[:, -1].std()),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--seeds",      type=int, default=3)
    parser.add_argument("--subset",     type=int, default=5000,
                        help="Training samples (0 = full dataset)")
    parser.add_argument("--no-subset",  action="store_true",
                        help="Use full 50 000-sample training set")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers",type=int, default=2)
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--device",     default=None,
                        help="Force device: cpu | cuda | mps (default: auto)")
    args = parser.parse_args()

    subset = 0 if args.no_subset else args.subset
    seeds  = list(range(args.seeds))

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | Seeds: {seeds} | "
          f"Subset: {subset if subset else 'full'}")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, subset, args.num_workers)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    cfg = BASE_CFG.copy()

    # Storage: {ablation_name: {label: [history_seed0, history_seed1, ...]}}
    all_raw = {}

    # ── 1. Baseline (run once, shared by both ablations) ─────────────────────
    print("\n=== Baseline: AdamW ===")
    baseline_histories = []
    for seed in seeds:
        h = run_one("AdamW", make_adamw(cfg),
                    train_loader, test_loader, device, args.epochs, seed)
        baseline_histories.append(h)

    # ── 2. Probe density ablation ─────────────────────────────────────────────
    print("\n=== Ablation 1: Probe density (n_samples) ===")
    density_variants = [0, 4, 8, 16, 32]
    density_raw = {"AdamW": baseline_histories}

    for n in density_variants:
        label = f"n={n}" if n > 0 else "n=0\n(neighbors only)"
        label_key = f"n_samples={n}"
        print(f"\n-- n_samples={n} --")
        hs = []
        for seed in seeds:
            h = run_one(label_key, make_topoadamw(cfg, n_samples=n),
                        train_loader, test_loader, device, args.epochs, seed)
            hs.append(h)
        density_raw[label_key] = hs

    all_raw["density"] = density_raw

    # ── 3. Probe interval ablation ────────────────────────────────────────────
    print("\n=== Ablation 2: Probe interval (T) ===")
    # Scaled for 40 batches/epoch: {5,10,20,50,100} gives the same
    # probes-per-epoch spread as {10,25,50,100,200} on full CIFAR-10.
    interval_variants = [5, 10, 20, 50, 100]
    interval_raw = {"AdamW": baseline_histories}

    for iv in interval_variants:
        label_key = f"T={iv}"
        print(f"\n-- interval={iv} --")
        hs = []
        for seed in seeds:
            h = run_one(label_key, make_topoadamw(cfg, interval=iv),
                        train_loader, test_loader, device, args.epochs, seed)
            hs.append(h)
        interval_raw[label_key] = hs

    all_raw["interval"] = interval_raw

    # ── Save raw results ──────────────────────────────────────────────────────
    with open("ablation_results.json", "w") as f:
        json.dump(all_raw, f, indent=2)
    print("\nSaved raw results to ablation_results.json")

    # ── Aggregate and plot ────────────────────────────────────────────────────
    # Density
    density_agg = {k: aggregate(v) for k, v in density_raw.items()}
    plot_ablation(
        results_by_label=density_agg,
        baseline_label="AdamW",
        param_name="n_samples (random probe points)",
        param_values=density_variants,
        out_path="assets/ablation_density.png",
        title="Probe Density Ablation — CIFAR-10",
    )

    # Interval
    interval_agg = {k: aggregate(v) for k, v in interval_raw.items()}
    plot_ablation(
        results_by_label=interval_agg,
        baseline_label="AdamW",
        param_name="Probe Interval T (steps)",
        param_values=interval_variants,
        out_path="assets/ablation_interval.png",
        title="Probe Interval Ablation — CIFAR-10",
    )

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*60)
    for abl_name, agg in [("Probe Density", density_agg),
                           ("Probe Interval", interval_agg)]:
        print(f"\n{abl_name} — Final Val Accuracy (mean ± std over {len(seeds)} seeds)")
        print(f"  {'Variant':<20} {'Accuracy':>10} {'±':>4} {'Std':>8}")
        print(f"  {'-'*44}")
        for label, res in agg.items():
            print(f"  {label:<20} {res['final_acc_mean']:>10.4f} {'±':>4} "
                  f"{res['final_acc_std']:>8.4f}")


if __name__ == "__main__":
    main()
