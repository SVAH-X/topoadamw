"""
Train CifarNet on CIFAR-100 using TopoAdamW.
"""

import argparse
import json
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from topoadamw import TopoAdamW
from models import CifarNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_ratio(optimizer: TopoAdamW) -> float:
    base_lr = optimizer.param_groups[0]["lr"]
    if hasattr(optimizer, "base_lrs") and optimizer.base_lrs:
        base_lr = optimizer.base_lrs[0]
    if base_lr == 0:
        return 0.0
    return optimizer.param_groups[0]["lr"] / base_lr


def build_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    subset: int,
) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)

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
    optimizer: TopoAdamW,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr_history: Optional[List[float]] = None,
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
        optimizer.step(data=data, target=target, criterion=criterion)

        if lr_history is not None:
            lr_history.append(get_lr_ratio(optimizer))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoAdamW CIFAR-100 example")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--probe-grid", type=int, default=11)
    parser.add_argument("--probe-span", type=float, default=0.10)
    parser.add_argument("--max-lr-ratio", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--subset", type=int, default=0, help="Limit data for quick runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-path",
        default="cifar100_history.json",
        help="Where to save training history (set empty to disable)",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(os.getenv("SEED", args.seed))
    set_seed(seed)

    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset,
    )

    num_classes = 100
    model = CifarNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = TopoAdamW(
        model.parameters(),
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        interval=args.interval,
        warmup_steps=args.warmup_steps,
        max_lr_ratio=args.max_lr_ratio,
        min_lr_ratio=args.min_lr_ratio,
        probe_kwargs={"grid_size": args.probe_grid, "span": args.probe_span},
        verbose=True,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr_ratio": []}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            criterion=criterion,
            device=device,
            lr_history=history["lr_ratio"],
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        lr_ratio = get_lr_ratio(optimizer)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr_ratio={lr_ratio:.3f}"
        )

    if args.history_path:
        history_dir = os.path.dirname(args.history_path)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        with open(args.history_path, "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()





