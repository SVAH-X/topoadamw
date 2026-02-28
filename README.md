# TopoAdamW: Topology-Guided Learning Rate Controller for AdamW

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

TopoAdamW is a topology-guided **learning rate controller** built on top of AdamW. It probes the local loss landscape and dynamically scales the learning rate to improve training stability and final accuracy.

TopoAdamW **does not modify gradient directions or optimizer internals**. It acts as an external controller that adapts step size based on the geometric regime of the loss landscape.

Two modes are available:

| Mode | How it works | Overhead | When to use |
|---|---|---|---|
| **Heuristic** (default) | Sparse 25-point probe; sharpness + variance rules | ~14% | Most training runs |
| **TDA + TopoCNN** | Dense grid probe + persistence image + small CNN | ~22% | Long runs (50+ epochs) |


## Why TopoAdamW?

Training with a fixed learning rate is like driving on a mountain road with cruise control. TopoAdamW acts as an intelligent co-pilot:

1. **Topological awareness**: probes local geometric regimes (flatness, sharpness, instability) of the loss landscape every `interval` steps.
2. **Dynamic adjustment**: scales LR within `[min_lr_ratio, max_lr_ratio]` based on what it observes.
3. **Safety lock**: if the center loss spikes above 2× EMA, a divergence brake overrides any acceleration.

The result is accelerated progress in smooth regions and stabilized behavior in sharp or noisy regions.


## Performance

Benchmarked on CIFAR-10 (CifarNet, 50 epochs, lr=1e-3, batch size 128, CPU):

| Optimizer | Val Accuracy | Val Loss | vs AdamW |
|---|---|---|---|
| AdamW (baseline) | 84.76% | 0.4983 | — |
| TopoAdamW (heuristic) | 86.19% | 0.4523 | +1.43% |
| TopoAdamW-TDA (CNN) | **86.27%** | **0.4559** | **+1.51%** |

![CIFAR-10 Comparison](assets/comparison_results.png)

CIFAR-100 comparison (100 epochs, lr=5e-4):

![CIFAR-100 Comparison](assets/comparison_results_cifar100.png)

| Optimizer | Val Accuracy | Val Loss | vs AdamW |
|---|---|---|---|
| AdamW (baseline) | 57.26% | 1.9423 | — |
| TopoAdamW (heuristic) | 58.76% | 1.7980 | +1.50% |
| TopoAdamW-TDA (CNN) | 58.72% | 1.8520 | +1.46% |

**Key observation:** The TDA+TopoCNN mode holds the learning rate higher for longer (epochs 1–36 at full LR vs. the heuristic which begins cutting at epoch 30), then cuts decisively. This extra training time at the initial LR is what gives it the edge in long runs.


## Quick Start

### Installation

```bash
git clone https://github.com/SVAH-X/topoadamw.git
cd topoadamw
pip install -r requirements.txt
```

For the TDA+TopoCNN mode, also install GUDHI:

```bash
pip install topoadamw[tda]   # installs gudhi as well
# or manually:
pip install gudhi
```

### Basic Usage (Heuristic Mode)

A drop-in replacement for `torch.optim.AdamW`. Just pass `data`, `target`, and `criterion` to `step()`:

```python
import torch
import torch.nn as nn
from topoadamw import TopoAdamW

model = YourModel()
criterion = nn.CrossEntropyLoss()

optimizer = TopoAdamW(model.parameters(), model, lr=1e-3, weight_decay=5e-4)

for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step(data=data, target=target, criterion=criterion)
```

Defaults: `lr=1e-3`, `weight_decay=5e-4`, `interval=50`, `warmup_steps=150`, `max_lr_ratio=1.0`, `min_lr_ratio=0.2`.

### TDA + TopoCNN Mode

Enable with `use_topo_cnn=True`. Requires `gudhi`. The optimizer starts in heuristic fallback mode and switches to TopoCNN once enough samples have been collected (default: 100 probes).

```python
from topoadamw import TopoAdamW

optimizer = TopoAdamW(
    model.parameters(),
    model,
    lr=1e-3,
    weight_decay=5e-4,
    use_topo_cnn=True,
    topo_cnn_kwargs={
        "min_samples": 100,    # Probes collected before first CNN training
        "retrain_every": 25,   # Retrain CNN every 25 new probes
        "train_epochs": 20,    # Gradient steps per retraining session
        "buffer_capacity": 500 # Ring buffer size
    },
    probe_kwargs={
        "tda_grid_size": 7     # Dense grid size for TDA (7x7 = 49 passes)
    }
)

# Training loop is identical
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step(data=data, target=target, criterion=criterion)
```

### Convenience Function

```python
from topoadamw import create_topoadamw

# Heuristic mode
optimizer = create_topoadamw(model, lr=1e-3)

# TDA+TopoCNN mode
optimizer = create_topoadamw(model, lr=1e-3, use_topo_cnn=True)
```

### Advanced Configuration (Heuristic Mode)

```python
optimizer = TopoAdamW(
    model.parameters(),
    model,
    lr=1e-3,
    weight_decay=5e-4,
    interval=50,           # Probe every 50 optimizer steps
    warmup_steps=150,      # No LR adjustment for first 150 steps
    max_lr_ratio=1.0,      # LR cap: 1.0x the initial LR
    min_lr_ratio=0.2,      # LR floor: 0.2x the initial LR
    probe_kwargs={
        "grid_size": 15,   # Neighbor step size reference
        "span": 0.12,      # Probing region size in parameter space
        "n_samples": 16    # Random points for variance estimation
    },
    verbose=True           # Print LR decisions each probe
)
```


## How It Works

### Heuristic Mode (default)

Every `interval` steps, TopoAdamW:

1. **Probes** a sparse 25-point neighborhood (1 center + 8 neighbors + 16 random samples) in a 2D random subspace of parameter space using filter-normalized directions (Li et al., NeurIPS 2018).
2. **Extracts** sharpness (how much neighbors exceed the center) and variance (roughness across all sampled points).
3. **Applies** the following control policy:

| Landscape | Sharpness | Variance | LR Factor |
|---|---|---|---|
| Flat & smooth | < 0.1 | < 0.3 | × 1.15 |
| Very sharp | > 0.5 | — | × 0.80 |
| Moderately sharp | > 0.2 | — | × 0.95 |
| High variance | — | > 0.5 | × 0.85 |
| Neutral | — | — | × 1.00 |

A **divergence brake** overrides these rules if the current loss exceeds 2× the exponential moving average, reducing LR by 20%.

### TDA + TopoCNN Mode

When `use_topo_cnn=True`, the optimizer runs a denser pipeline:

1. **Dense probe**: evaluates a `tda_grid_size × tda_grid_size` grid (default 7×7 = 49 passes) to capture the full local loss surface.
2. **Persistence image**: computes a 2-channel 50×50 float tensor encoding the topological structure of the loss surface (H0 and H1 persistence pairs via GUDHI cubical complexes).
3. **TopoCNN**: a small CNN classifies the persistence image into one of three landscape regimes:
   - `FLAT` (0) → LR × 1.15
   - `NEUTRAL` (1) → LR × 1.00
   - `DECEL` (2) → LR × 0.85
4. **Bootstrapping**: during warmup (first `min_samples` probes), the heuristic labels each persistence image and trains TopoCNN. Once trained, TopoCNN takes over and is periodically retrained as new samples arrive.

The CNN's advantage over the heuristic is its ability to capture global topological features of the loss surface (connected components, loops) that local sharpness/variance measurements miss. In practice, this manifests as more patient LR scheduling — TopoCNN tends to keep the LR higher for longer before cutting, squeezing out additional training progress.


## Repository Structure

```
topoadamw/
├── topoadamw/
│   ├── __init__.py          # Package exports
│   ├── optimizer.py         # TopoAdam / TopoAdamW classes
│   ├── probe.py             # SubspaceProbe: sparse and dense landscape probing
│   ├── tda_utils.py         # LossLandscapeTDA: persistence image computation
│   └── topo_trainer.py      # TopoCNN model + TopoCNNTrainer (online bootstrapping)
├── examples/
│   ├── cifar10_example.py
│   ├── cifar100_example.py
│   └── custom_model_example.py
├── tests/
│   └── test_optimizer.py
├── models.py                    # CifarNet benchmark model
├── generate_comparison_plot.py  # Reproduce benchmarks
├── requirements.txt
└── README.md
```


## Reproducing Results

Run the full benchmark (AdamW vs TopoAdamW vs TopoAdamW-TDA):

```bash
# 50-epoch CIFAR-10, all three optimizers
python generate_comparison_plot.py --datasets cifar10 --epochs 50 --use-topo-cnn

# Heuristic only (faster)
python generate_comparison_plot.py --datasets cifar10 --epochs 50

# CIFAR-100
python generate_comparison_plot.py --datasets cifar100
```

Useful flags:

```
--epochs N            Override epoch count
--use-topo-cnn        Include TopoAdamW-TDA in the comparison
--tda-min-samples N   Probes before first CNN training (default: 100)
--tda-retrain-every N Retrain interval in new probes (default: 25)
--tda-grid-size N     Dense grid size for TDA (default: 7)
--subset N            Limit dataset size for quick validation
--seed N              Random seed (default: 42)
```

Run unit tests:

```bash
python -m pytest tests/
```


## Examples

```bash
python examples/cifar10_example.py
python examples/cifar100_example.py
python examples/custom_model_example.py
```


## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.20
- torchvision (for examples and benchmarks)
- GUDHI >= 3.0 — only required for `use_topo_cnn=True`

```bash
pip install torch torchvision numpy
pip install topoadamw[tda]   # optional, TDA+TopoCNN mode only
```


## Known Limitations

- **Overhead**: Heuristic mode adds ~14% per-epoch overhead; TDA+TopoCNN adds ~22%.
- **TopoCNN warmup**: With the default `min_samples=100` and `interval=50`, TopoCNN activates after roughly 13 epochs on CIFAR-10 (391 batches/epoch). Shorter runs will stay in heuristic fallback.
- **Small batches**: Works best with batch size >= 32. Very small batches produce noisy landscape estimates.
- **Single optimizer group**: LR scaling applies uniformly across all parameter groups.


## Citation

```bibtex
@software{topoadamw2025,
  author    = {Congkai Peng},
  title     = {TopoAdamW: Topology-Guided Learning Rate Control for Neural Network Training},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/SVAH-X/topoadamw}
}
```


## References

- Li et al. "Visualizing the Loss Landscape of Neural Nets." NeurIPS 2018.
- Carlsson, G. "Topology and Data." Bulletin of the AMS, 2009.
- Edelsbrunner & Harer. "Computational Topology: An Introduction." AMS, 2010.


## License

MIT License — see [LICENSE](LICENSE) for details.
