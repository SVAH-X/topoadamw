"""
Multi-seed benchmark for TopoAdamW — produces mean ± std for Table 1.

Saves a checkpoint after every completed run so progress survives crashes.
Resume simply by re-running the same command; completed runs are skipped.

Usage:
  python multiseed_benchmark.py                          # CIFAR-10, 5 seeds, no TDA
  python multiseed_benchmark.py --seeds 3 --include-tda
  python multiseed_benchmark.py --include-cifar100       # adds CIFAR-100 (very slow)
  python multiseed_benchmark.py --device cpu
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from topoadamw import TopoAdamW
from models import CifarNet

CHECKPOINT = "multiseed_results.json"

DATASET_CFG = {
    "cifar10": {
        "cls":         datasets.CIFAR10,
        "num_classes": 10,
        "mean":        (0.4914, 0.4822, 0.4465),
        "std":         (0.2023, 0.1994, 0.2010),
        "epochs":      50,
        "lr":          1e-3,
        "interval":    50,
        "warmup":      150,
        "min_lr":      0.2,
    },
    "cifar100": {
        "cls":         datasets.CIFAR100,
        "num_classes": 100,
        "mean":        (0.5071, 0.4867, 0.4408),
        "std":         (0.2675, 0.2565, 0.2761),
        "epochs":      100,
        "lr":          5e-4,
        "interval":    100,
        "warmup":      300,
        "min_lr":      0.3,
    },
}


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Data ─────────────────────────────────────────────────────────────────────
def build_loaders(dataset_name, data_dir, batch_size, num_workers):
    cfg = DATASET_CFG[dataset_name]
    normalize = transforms.Normalize(cfg["mean"], cfg["std"])
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    pin = torch.cuda.is_available()
    train_set = cfg["cls"](data_dir, train=True,  download=True, transform=train_tf)
    test_set  = cfg["cls"](data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


# ── Training ──────────────────────────────────────────────────────────────────
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
            out = model(data)
            loss = criterion(out, target)
            total_loss += loss.item() * len(data)
            correct    += out.argmax(1).eq(target).sum().item()
            total      += len(data)
    return total_loss / total, correct / total


def run_one(opt_label, make_opt, dataset_name, train_loader, test_loader,
            device, seed):
    cfg = DATASET_CFG[dataset_name]
    set_seed(seed)
    model     = CifarNet(cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_opt(model)
    epochs    = cfg["epochs"]

    val_accs, val_losses, train_losses = [], [], []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl       = train_epoch(model, optimizer, train_loader, criterion, device)
        vl, va   = evaluate(model, test_loader, criterion, device)
        elapsed  = time.time() - t0
        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)

        if not math.isfinite(vl):
            print(f"  [{opt_label}|{dataset_name}|seed={seed}] ep={ep}/{epochs} "
                  f"DIVERGED ({elapsed:.0f}s)")
            break

        print(f"  [{opt_label}|{dataset_name}|seed={seed}] ep={ep}/{epochs} "
              f"val_acc={va:.4f} val_loss={vl:.4f} ({elapsed:.0f}s elapsed)")

    return {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accs}


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2)


def run_key(dataset_name, opt_label, seed):
    return f"{dataset_name}/{opt_label}/seed={seed}"


# ── Aggregate + plot ──────────────────────────────────────────────────────────
def aggregate(histories):
    max_ep = max(len(h["val_acc"]) for h in histories)
    pad = lambda h: h["val_acc"] + [h["val_acc"][-1]] * (max_ep - len(h["val_acc"]))
    arr = np.array([pad(h) for h in histories])
    losses_arr = np.array([
        h["val_loss"] + [h["val_loss"][-1]] * (max_ep - len(h["val_loss"]))
        for h in histories
    ])
    return {
        "mean_acc":       arr.mean(0).tolist(),
        "std_acc":        arr.std(0).tolist(),
        "final_acc_mean": float(arr[:, -1].mean()),
        "final_acc_std":  float(arr[:, -1].std()),
        "mean_loss":      losses_arr.mean(0).tolist(),
        "std_loss":       losses_arr.std(0).tolist(),
        "final_loss_mean": float(losses_arr[:, -1].mean()),
        "final_loss_std":  float(losses_arr[:, -1].std()),
    }


def save_plot(agg_by_opt, dataset_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"AdamW": "#555", "TopoAdamW": "#2196F3", "TopoAdamW-TDA": "#FF9800"}

    for opt_label, res in agg_by_opt.items():
        col   = colors.get(opt_label, "gray")
        ep    = range(1, len(res["mean_acc"]) + 1)
        mean  = np.array(res["mean_acc"])
        std   = np.array(res["std_acc"])
        lmean = np.array(res["mean_loss"])
        lstd  = np.array(res["std_loss"])

        axes[0].plot(ep, mean, label=opt_label, color=col)
        axes[0].fill_between(ep, mean - std, mean + std, alpha=0.15, color=col)
        axes[1].plot(ep, lmean, label=opt_label, color=col)
        axes[1].fill_between(ep, lmean - lstd, lmean + lstd, alpha=0.15, color=col)

    axes[0].set_title("Val Accuracy")
    axes[1].set_title("Val Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset_name.upper()} — mean ± std over {len(list(agg_by_opt.values())[0]['mean_acc'])} epochs")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(agg_by_opt, dataset_name, n_seeds):
    print(f"\n{'='*65}")
    print(f"{dataset_name.upper()} — Final Val Accuracy & Loss (mean ± std, {n_seeds} seeds)")
    print(f"{'Optimizer':<20} {'Accuracy':>10}  {'±':>4}  {'Loss':>8}  {'±':>4}")
    print(f"{'-'*55}")
    for opt_label, res in agg_by_opt.items():
        print(f"{opt_label:<20} {res['final_acc_mean']:>10.4f}  ±{res['final_acc_std']:.4f}"
              f"  {res['final_loss_mean']:>8.4f}  ±{res['final_loss_std']:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, default=5)
    parser.add_argument("--include-tda",     action="store_true")
    parser.add_argument("--include-cifar100",action="store_true")
    parser.add_argument("--data-dir",        default="data")
    parser.add_argument("--batch-size",      type=int, default=128)
    parser.add_argument("--num-workers",     type=int, default=2)
    parser.add_argument("--device",          default=None)
    args = parser.parse_args()

    seeds    = list(range(args.seeds))
    datasets_to_run = ["cifar10"]
    if args.include_cifar100:
        datasets_to_run.append("cifar100")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device} | Seeds: {seeds} | Datasets: {datasets_to_run} | "
          f"TDA: {args.include_tda}")

    # Load any existing checkpoint
    ckpt = load_checkpoint()
    os.makedirs(args.data_dir, exist_ok=True)

    for dataset_name in datasets_to_run:
        cfg = DATASET_CFG[dataset_name]
        print(f"\n{'='*65}\nDataset: {dataset_name.upper()} | "
              f"epochs={cfg['epochs']} lr={cfg['lr']}")

        train_loader, test_loader = build_loaders(
            dataset_name, args.data_dir, args.batch_size, args.num_workers)

        def make_adamw(model):
            return optim.AdamW(model.parameters(),
                               lr=cfg["lr"], weight_decay=5e-4)

        def make_topoadamw(model):
            return TopoAdamW(
                model.parameters(), model,
                lr=cfg["lr"], weight_decay=5e-4,
                interval=cfg["interval"],
                warmup_steps=cfg["warmup"],
                max_lr_ratio=1.0,
                min_lr_ratio=cfg["min_lr"],
            )

        optimizers = [("AdamW", make_adamw), ("TopoAdamW", make_topoadamw)]

        if args.include_tda:
            try:
                import gudhi  # noqa
                def make_tda(model):
                    return TopoAdamW(
                        model.parameters(), model,
                        lr=cfg["lr"], weight_decay=5e-4,
                        interval=cfg["interval"],
                        warmup_steps=cfg["warmup"],
                        max_lr_ratio=1.0,
                        min_lr_ratio=cfg["min_lr"],
                        use_topo_cnn=True,
                    )
                optimizers.append(("TopoAdamW-TDA", make_tda))
            except ImportError:
                print("gudhi not installed — skipping TDA mode")

        for opt_label, make_opt in optimizers:
            for seed in seeds:
                key = run_key(dataset_name, opt_label, seed)
                if key in ckpt:
                    print(f"  [SKIP] {key} (already in checkpoint)")
                    continue

                print(f"\n--- {key} ---")
                history = run_one(opt_label, make_opt, dataset_name,
                                  train_loader, test_loader, device, seed)
                ckpt[key] = history
                save_checkpoint(ckpt)

        # Aggregate and report after all runs for this dataset
        agg = {}
        for opt_label, _ in optimizers:
            histories = [ckpt[run_key(dataset_name, opt_label, s)] for s in seeds
                         if run_key(dataset_name, opt_label, s) in ckpt]
            if histories:
                agg[opt_label] = aggregate(histories)

        if agg:
            out = f"assets/multiseed_{dataset_name}.png"
            save_plot(agg, dataset_name, out)
            print_summary(agg, dataset_name, len(seeds))

    print("\nDone.")


if __name__ == "__main__":
    main()
