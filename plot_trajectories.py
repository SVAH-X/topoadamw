"""
Training trajectory visualization for TopoAdamW vs AdamW.

Records per-epoch LR and per-probe sharpness/variance for TopoAdamW.
Generates assets/trajectories.png (3-panel figure).

Usage:
    python plot_trajectories.py [--seed SEED] [--device DEVICE]
"""

import argparse
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


# ── Logging wrapper ────────────────────────────────────────────────────────────

class LoggingTopoAdamW(TopoAdamW):
    """TopoAdamW that records LR, sharpness, and variance at every probe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = {
            "probe_step": [],
            "lr_at_probe": [],
            "sharpness": [],
            "variance": [],
            "center_loss": [],
            "factor": [],
        }

    def _adjust_learning_rate(self, data, target, criterion):
        data, target = data.to(self.device), target.to(self.device)

        # Sparse probe
        probe_result = self.probe.probe(data, target, criterion)
        features     = self._extract_geometric_features(probe_result)
        factor       = self._geometric_heuristic(features)

        # Update EMA
        if self.loss_ema is None:
            self.loss_ema = features["center_loss"]
        else:
            self.loss_ema = (self.ema_alpha * features["center_loss"] +
                             (1 - self.ema_alpha) * self.loss_ema)

        if features["center_loss"] > self.loss_ema * 2.0:
            factor = 0.8

        # Record BEFORE applying
        self.log["probe_step"].append(self.steps)
        self.log["lr_at_probe"].append(self.optimizer.param_groups[0]["lr"])
        self.log["sharpness"].append(features["sharpness"])
        self.log["variance"].append(features["variance"])
        self.log["center_loss"].append(features["center_loss"])
        self.log["factor"].append(factor)

        # Apply LR update
        if abs(factor - 1.0) > 0.05:
            for i, pg in enumerate(self.optimizer.param_groups):
                old_lr = pg["lr"]
                pg["lr"] = max(
                    min(old_lr * factor,
                        self.base_lrs[i] * self.max_lr_ratios[i]),
                    self.base_lrs[i] * self.min_lr_ratios[i],
                )


# ── Training helpers ───────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    pin = torch.cuda.is_available()
    os.makedirs(data_dir, exist_ok=True)
    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(test_set,  batch_size=batch_size, shuffle=False,
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
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            total_loss += criterion(out, y).item() * len(x)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += len(x)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir",    default="data")
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

    print(f"Device: {device} | Seed: {args.seed} | Epochs: {args.epochs}")

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()

    # ── AdamW ──────────────────────────────────────────────────────────────────
    set_seed(args.seed)
    model_a   = CifarNet(10).to(device)
    opt_a     = optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_adamw, acc_adamw, loss_adamw = [], [], []

    print("Training AdamW...")
    for ep in range(1, args.epochs + 1):
        train_epoch(model_a, opt_a, train_loader, criterion, device)
        vl, va = evaluate(model_a, test_loader, criterion, device)
        lr_adamw.append(opt_a.param_groups[0]["lr"])
        acc_adamw.append(va)
        loss_adamw.append(vl)
        print(f"  AdamW ep={ep}/{args.epochs} acc={va:.4f}")

    # ── TopoAdamW ──────────────────────────────────────────────────────────────
    set_seed(args.seed)
    model_t = CifarNet(10).to(device)
    opt_t   = LoggingTopoAdamW(
        model_t.parameters(), model_t,
        lr=1e-3, weight_decay=5e-4,
        interval=50, warmup_steps=150,
        max_lr_ratio=1.0, min_lr_ratio=0.2,
    )
    lr_topo, acc_topo, loss_topo = [], [], []

    print("Training TopoAdamW...")
    for ep in range(1, args.epochs + 1):
        train_epoch(model_t, opt_t, train_loader, criterion, device)
        vl, va = evaluate(model_t, test_loader, criterion, device)
        lr_topo.append(opt_t.optimizer.param_groups[0]["lr"])
        acc_topo.append(va)
        loss_topo.append(vl)
        print(f"  TopoAdamW ep={ep}/{args.epochs} acc={va:.4f} "
              f"lr={opt_t.optimizer.param_groups[0]['lr']:.6f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    epochs = list(range(1, args.epochs + 1))
    probe_log = opt_t.log

    # Convert probe steps to approximate epoch (steps per epoch ≈ 391)
    steps_per_epoch = len(train_loader)
    probe_epochs = [s / steps_per_epoch for s in probe_log["probe_step"]]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

    # Panel 1: Val accuracy
    ax = axes[0]
    ax.plot(epochs, acc_adamw, color="#555", label="AdamW", linewidth=1.5)
    ax.plot(epochs, acc_topo,  color="#2196F3", label="TopoAdamW", linewidth=1.5)
    ax.set_ylabel("Val Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Validation Accuracy")

    # Panel 2: Learning rate
    ax = axes[1]
    ax.plot(epochs, lr_adamw, color="#555",     label="AdamW (constant)", linewidth=1.5)
    ax.plot(epochs, lr_topo,  color="#2196F3",  label="TopoAdamW",        linewidth=1.5)
    # Mark probe events
    ax.scatter(probe_epochs, probe_log["lr_at_probe"],
               c=["#E53935" if f != 1.0 else "#43A047"
                  for f in probe_log["factor"]],
               zorder=5, s=18, label="Probe (red=adjust, green=neutral)")
    ax.set_ylabel("Learning Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Learning Rate Trajectory (dots = probe events)")

    # Panel 3: Sharpness and Variance
    ax = axes[2]
    ax.plot(probe_epochs, probe_log["sharpness"], color="#E53935",
            label="Sharpness", linewidth=1.2, marker=".", markersize=4)
    ax.plot(probe_epochs, probe_log["variance"],  color="#FF9800",
            label="Variance",  linewidth=1.2, marker=".", markersize=4)
    # Draw threshold lines
    ax.axhline(0.1, color="#E53935", linestyle="--", alpha=0.4, linewidth=0.8,
               label="Sharp thresh (0.1)")
    ax.axhline(0.5, color="#E53935", linestyle=":",  alpha=0.4, linewidth=0.8,
               label="Sharp thresh (0.5)")
    ax.axhline(0.5, color="#FF9800", linestyle=":",  alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Feature Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Sharpness & Variance at Each Probe (dashed = decision thresholds)")

    for ax in axes:
        ax.set_xlabel("Epoch")

    fig.suptitle(f"TopoAdamW Training Trajectories — CIFAR-10, seed={args.seed}",
                 fontsize=12)
    fig.tight_layout()

    os.makedirs("assets", exist_ok=True)
    out = "assets/trajectories.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Also save raw log
    import json
    log_data = {
        "adamw":  {"lr": lr_adamw, "val_acc": acc_adamw, "val_loss": loss_adamw},
        "topo":   {"lr": lr_topo,  "val_acc": acc_topo,  "val_loss": loss_topo,
                   "probe_log": probe_log},
    }
    with open("trajectory_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    print("Saved: trajectory_log.json")


if __name__ == "__main__":
    main()
