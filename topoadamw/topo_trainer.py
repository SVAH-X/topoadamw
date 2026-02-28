"""
TopoCNN and its online bootstrapped trainer.

TopoCNN classifies a 2-channel persistence image (output of LossLandscapeTDA)
into one of three landscape regimes and returns an LR scaling factor.

TopoCNNTrainer bootstraps supervision from the existing geometric heuristic:
  1. For the first `min_samples` probes it labels each sample with the heuristic
     and stores it in a ring buffer.
  2. Once enough samples are collected TopoCNN is trained.
  3. From then on predict_factor() returns the CNN's decision.
  4. The model is periodically retrained as new samples arrive.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

# ── Landscape regime labels ───────────────────────────────────────────────────
FLAT    = 0   # flat & smooth  → accelerate (LR × 1.15)
NEUTRAL = 1   # moderate       → maintain   (LR × 1.00)
DECEL   = 2   # sharp / noisy  → slow down  (LR × 0.85)

LR_FACTORS = {FLAT: 1.15, NEUTRAL: 1.0, DECEL: 0.85}


# ── Model ─────────────────────────────────────────────────────────────────────
class TopoCNN(nn.Module):
    """
    Small CNN that maps a 2-channel persistence image to a landscape regime.

    Input:  (B, 2, H, W)   — default H = W = 50  (LossLandscapeTDA.resolution)
    Output: (B, 3) logits  — classes: FLAT (0), NEUTRAL (1), DECEL (2)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 50 → 25
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 25 → 12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),            # → 1×1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Trainer ───────────────────────────────────────────────────────────────────
class TopoCNNTrainer:
    """
    Online trainer that bootstraps TopoCNN supervision from the geometric heuristic.

    Labelling rule (mirrors _geometric_heuristic in optimizer.py):
      sharpness < 0.1 AND variance < 0.3  →  FLAT    (accelerate)
      sharpness > 0.2 OR  variance > 0.5  →  DECEL   (slow down)
      otherwise                           →  NEUTRAL (maintain)

    Parameters
    ----------
    min_samples : int
        Minimum buffer size before first training round (default 100).
    retrain_every : int
        Retrain after this many new samples (default 25).
    train_epochs : int
        Gradient steps per training session (default 20).
    buffer_capacity : int
        Ring-buffer size; oldest samples are evicted when full (default 500).
    device : torch.device or None
    verbose : bool
    """

    def __init__(
        self,
        min_samples: int = 100,
        retrain_every: int = 25,
        train_epochs: int = 20,
        buffer_capacity: int = 500,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        self.device        = device or torch.device('cpu')
        self.min_samples   = min_samples
        self.retrain_every = retrain_every
        self.train_epochs  = train_epochs
        self.verbose       = verbose

        self.cnn      = TopoCNN().to(self.device)
        self._opt     = optim.Adam(self.cnn.parameters(), lr=1e-3, weight_decay=1e-4)
        self._loss_fn = nn.CrossEntropyLoss()

        # Ring buffer (CPU tensors to avoid holding GPU memory)
        self._imgs:   list = []
        self._labels: list = []
        self._capacity     = buffer_capacity
        self._since_train  = 0
        self.is_ready      = False   # True once trained at least once

    # ── Heuristic labeller ────────────────────────────────────────────────────
    @staticmethod
    def heuristic_label(sharpness: float, variance: float) -> int:
        if sharpness < 0.1 and variance < 0.3:
            return FLAT
        if sharpness > 0.2 or variance > 0.5:
            return DECEL
        return NEUTRAL

    # ── Public API ────────────────────────────────────────────────────────────
    def add_sample(
        self,
        persistence_img: torch.Tensor,
        sharpness: float,
        variance: float,
    ) -> None:
        """Store a (persistence_image, heuristic_label) pair; retrain if due."""
        label = self.heuristic_label(sharpness, variance)

        if len(self._imgs) >= self._capacity:
            self._imgs.pop(0)
            self._labels.pop(0)

        self._imgs.append(persistence_img.detach().cpu())
        self._labels.append(label)
        self._since_train += 1

        if (len(self._imgs) >= self.min_samples
                and self._since_train >= self.retrain_every):
            self._train()

    def predict_factor(self, persistence_img: torch.Tensor) -> Optional[float]:
        """
        Return an LR scaling factor in {0.85, 1.00, 1.15}.
        Returns None if TopoCNN has not been trained yet (caller falls back
        to the heuristic).
        """
        if not self.is_ready:
            return None
        self.cnn.eval()
        with torch.no_grad():
            logits = self.cnn(persistence_img.unsqueeze(0).to(self.device))
            cls    = int(logits.argmax(dim=1).item())
        return LR_FACTORS[cls]

    # ── Internal training ─────────────────────────────────────────────────────
    def _train(self) -> None:
        imgs   = torch.stack(self._imgs).to(self.device)                    # [N,2,H,W]
        labels = torch.tensor(self._labels, dtype=torch.long,
                              device=self.device)                            # [N]

        self.cnn.train()
        for _ in range(self.train_epochs):
            self._opt.zero_grad()
            loss = self._loss_fn(self.cnn(imgs), labels)
            loss.backward()
            self._opt.step()

        self.is_ready     = True
        self._since_train = 0

        if self.verbose:
            self.cnn.eval()
            with torch.no_grad():
                preds = self.cnn(imgs).argmax(dim=1)
                acc   = (preds == labels).float().mean().item()
            counts = {c: int((labels == c).sum()) for c in (FLAT, NEUTRAL, DECEL)}
            print(
                f"[TopoCNN] Trained | n={len(self._imgs)} | acc={acc:.3f} | "
                f"flat={counts[FLAT]} neutral={counts[NEUTRAL]} decel={counts[DECEL]}"
            )
