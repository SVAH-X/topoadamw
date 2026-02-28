"""
Benchmark AdamW and TopoAdamW on CIFAR-10/100 and save comparison plots.
"""

import argparse
import math
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from topoadamw import TopoAdamW
from models import CifarNet

CONFIG_KEYS = (
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "interval",
    "warmup_steps",
    "probe_grid",
    "probe_span",
    "max_lr_ratio",
    "min_lr_ratio",
)

DATASET_SPECS = {
    "cifar10": {
        "dataset_cls": datasets.CIFAR10,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "defaults": {
            "epochs": 30,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "interval": 50,
            "warmup_steps": 150,
            "probe_grid": 15,
            "probe_span": 0.12,
            "max_lr_ratio": 1.0,
            "min_lr_ratio": 0.2,
        },
    },
    "cifar100": {
        "dataset_cls": datasets.CIFAR100,
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "defaults": {
            "epochs": 100,
            "batch_size": 128,
            "lr": 5e-4,
            "weight_decay": 5e-4,
            "interval": 100,
            "warmup_steps": 300,
            "probe_grid": 11,
            "probe_span": 0.10,
            "max_lr_ratio": 1.0,
            "min_lr_ratio": 0.3,
        },
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rolling_std(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.std(values[start : i + 1])))
    return out


def build_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    subset: int,
    dataset_name: str,
) -> Tuple[DataLoader, DataLoader]:
    spec = DATASET_SPECS[dataset_name]
    normalize = transforms.Normalize(mean=spec["mean"], std=spec["std"])
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    dataset_cls = spec["dataset_cls"]
    train_set = dataset_cls(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = dataset_cls(root=data_dir, train=False, download=True, transform=test_tf)

    if subset and subset > 0:
        train_indices = list(range(min(subset, len(train_set))))
        test_indices = list(range(min(subset, len(test_set))))
        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    optimizer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        if isinstance(optimizer, TopoAdamW):
            optimizer.step(data=data, target=target, criterion=criterion)
        else:
            optimizer.step()

        total_loss += loss.item() * data.size(0)
        total += data.size(0)

    return total_loss / max(total, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def run_experiment(
    name: str,
    make_optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int,
    num_classes: int,
) -> Dict[str, List[float]]:
    set_seed(seed)
    model = CifarNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    base_lr = optimizer.param_groups[0]["lr"]
    if isinstance(optimizer, TopoAdamW):
        min_lr = base_lr * optimizer.min_lr_ratio
        max_lr = base_lr * optimizer.max_lr_ratio
    else:
        min_lr = base_lr
        max_lr = base_lr

    print(f"\n=== {name} ===")
    print(
        f"Model=CifarNet | Device={device.type} | Epochs={epochs} | "
        f"Train batches={len(train_loader)} | Val batches={len(test_loader)}"
    )
    print(f"LR base={base_lr:.6f} | range=[{min_lr:.6f}, {max_lr:.6f}]")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            criterion=criterion,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        if not math.isfinite(val_loss):
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss=nan | val_acc={val_acc:.4f} | "
                f"lr={lr:.6f} | time={elapsed:.1f}s | status=diverged"
            )
            break

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={lr:.6f} | time={elapsed:.1f}s"
        )

    return history


def summarize_results(results: Dict[str, Dict[str, List[float]]], label: str) -> None:
    print(f"\nSummary ({label} final epoch values)")
    header = f"{'Optimizer':<10} {'Val Loss':>10} {'Acc':>8} {'Stability':>10} {'Convergence':>12}"
    print(header)
    print("-" * len(header))

    for name, history in results.items():
        val_losses = np.array(history["val_loss"], dtype=float)
        val_accs = np.array(history["val_acc"], dtype=float)

        final_loss = val_losses[-1] if len(val_losses) else float("nan")
        final_acc = val_accs[-1] if len(val_accs) else float("nan")
        stability = float(np.nanstd(val_losses)) if len(val_losses) else float("nan")
        converged = "Stable" if np.all(np.isfinite(val_losses)) else "Diverging"

        print(
            f"{name:<10} {final_loss:>10.4f} {final_acc:>8.4f} "
            f"{stability:>10.4f} {converged:>12}"
        )


def save_plot(
    results: Dict[str, Dict[str, List[float]]], out_path: str, title: str
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for name, history in results.items():
        epochs = list(range(1, len(history["train_loss"]) + 1))
        axes[0].plot(epochs, history["train_loss"], label=name)
        axes[1].plot(epochs, history["val_loss"], label=name)
        axes[2].plot(epochs, history["val_acc"], label=name)
        axes[3].plot(epochs, rolling_std(history["val_loss"]), label=name)

    axes[0].set_title("Train Loss")
    axes[1].set_title("Val Loss")
    axes[2].set_title("Val Accuracy")
    axes[3].set_title("Val Loss Std (Window=5)")

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[2].set_ylabel("Accuracy")
    axes[3].set_ylabel("Std")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoAdamW CIFAR-10/100 benchmark")
    parser.add_argument(
        "--datasets",
        default="cifar10,cifar100",
        help="Comma-separated list: cifar10,cifar100",
    )
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--probe-grid", type=int, default=None)
    parser.add_argument("--probe-span", type=float, default=None)
    parser.add_argument("--max-lr-ratio", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--subset", type=int, default=0, help="Limit data for quick runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-topo-cnn", action="store_true",
                        help="Add a TopoAdamW-TDA experiment using TDA+TopoCNN pipeline")
    parser.add_argument("--tda-grid-size", type=int, default=7,
                        help="Grid size for TDA dense probe (default: 7)")
    parser.add_argument("--tda-min-samples", type=int, default=100,
                        help="Min samples before TopoCNN trains (default: 100)")
    parser.add_argument("--tda-retrain-every", type=int, default=25,
                        help="Retrain TopoCNN every N samples (default: 25)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    requested = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in requested if name not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")

    for dataset_name in requested:
        spec = DATASET_SPECS[dataset_name]
        cfg = spec["defaults"].copy()
        for key in CONFIG_KEYS:
            value = getattr(args, key)
            if value is not None:
                cfg[key] = value

        train_loader, test_loader = build_loaders(
            data_dir=args.data_dir,
            batch_size=cfg["batch_size"],
            num_workers=args.num_workers,
            subset=args.subset,
            dataset_name=dataset_name,
        )
        print(f"\n=== Dataset: {dataset_name.upper()} ===")
        print(
            f"Data ready | Train samples={len(train_loader.dataset)} | "
            f"Val samples={len(test_loader.dataset)} | Batch size={cfg['batch_size']}"
        )
        print(
            f"Hyperparams | epochs={cfg['epochs']} | lr={cfg['lr']} | "
            f"weight_decay={cfg['weight_decay']} | interval={cfg['interval']} | "
            f"warmup_steps={cfg['warmup_steps']} | probe_grid={cfg['probe_grid']} | "
            f"probe_span={cfg['probe_span']} | max_lr_ratio={cfg['max_lr_ratio']} | "
            f"min_lr_ratio={cfg['min_lr_ratio']}"
        )

        def make_adamw(model: nn.Module):
            return optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )

        def make_topoadamw(model: nn.Module):
            return TopoAdamW(
                model.parameters(),
                model,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                interval=cfg["interval"],
                warmup_steps=cfg["warmup_steps"],
                max_lr_ratio=cfg["max_lr_ratio"],
                min_lr_ratio=cfg["min_lr_ratio"],
                probe_kwargs={"grid_size": cfg["probe_grid"], "span": cfg["probe_span"]},
            )

        def make_topoadamw_tda(model: nn.Module):
            return TopoAdamW(
                model.parameters(),
                model,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                interval=cfg["interval"],
                warmup_steps=cfg["warmup_steps"],
                max_lr_ratio=cfg["max_lr_ratio"],
                min_lr_ratio=cfg["min_lr_ratio"],
                probe_kwargs={
                    "grid_size": cfg["probe_grid"],
                    "span": cfg["probe_span"],
                    "tda_grid_size": args.tda_grid_size,
                },
                use_topo_cnn=True,
                topo_cnn_kwargs={
                    "min_samples": args.tda_min_samples,
                    "retrain_every": args.tda_retrain_every,
                },
            )

        experiments = [
            ("AdamW", make_adamw),
            ("TopoAdamW", make_topoadamw),
        ]
        if args.use_topo_cnn:
            experiments.append(("TopoAdamW-TDA", make_topoadamw_tda))

        results: Dict[str, Dict[str, List[float]]] = {}
        for name, make_opt in experiments:
            print(f"\nStarting run for {name}...")
            results[name] = run_experiment(
                name=name,
                make_optimizer=make_opt,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=cfg["epochs"],
                seed=args.seed,
                num_classes=spec["num_classes"],
            )

        if dataset_name == "cifar10":
            out_file = "comparison_results.png"
        else:
            out_file = f"comparison_results_{dataset_name}.png"
        save_plot(results, out_file, title=f"{dataset_name.upper()} Metrics")

        os.makedirs("assets", exist_ok=True)
        assets_path = os.path.join("assets", out_file)
        save_plot(results, assets_path, title=f"{dataset_name.upper()} Metrics")

        summarize_results(results, label=dataset_name.upper())
        print(f"\nSaved plots: {out_file} and {assets_path}")


if __name__ == "__main__":
    main()
