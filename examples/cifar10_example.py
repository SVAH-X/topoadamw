"""
Train CifarNet on CIFAR-10 using TopoAdamW.
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from topoadamw import TopoAdamW
from models import CifarNet


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

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

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
    parser = argparse.ArgumentParser(description="TopoAdamW CIFAR-10 example")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--probe-grid", type=int, default=15)
    parser.add_argument("--probe-span", type=float, default=0.12)
    parser.add_argument("--max-lr-ratio", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--subset", type=int, default=0, help="Limit data for quick runs")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset,
    )

    model = CifarNet().to(device)
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
    )

    for epoch in range(1, args.epochs + 1):
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
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()







