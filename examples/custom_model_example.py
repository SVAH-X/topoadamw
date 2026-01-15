"""
Train a custom model on synthetic data using TopoAdamW.
"""

import argparse
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from topoadamw import TopoAdamW


class ToyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataset(
    num_samples: int, input_dim: int, num_classes: int
) -> Tuple[TensorDataset, TensorDataset]:
    torch.manual_seed(0)
    x = torch.randn(num_samples, input_dim)
    true_w = torch.randn(input_dim, num_classes)
    logits = x @ true_w
    y = logits.argmax(dim=1)

    split = int(num_samples * 0.8)
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds = TensorDataset(x[split:], y[split:])
    return train_ds, val_ds


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
    parser = argparse.ArgumentParser(description="TopoAdamW custom model example")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = make_dataset(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ToyNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = TopoAdamW(
        model.parameters(),
        model,
        lr=args.lr,
        interval=10,
        warmup_steps=0,
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
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()




