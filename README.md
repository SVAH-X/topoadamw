# TopoAdamW: Topology-Guided Learning Rate Controller for AdamW 🌄

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

TopoAdamW is a topology-guided **learning rate controller** built on top of AdamW. 
It probes the local loss landscape using Topological Data Analysis (TDA) and dynamically scales the global learning rate to improve training stability and efficiency.

TopoAdamW **does not modify gradient directions or optimizer internals**; instead, it acts as an external controller that adapts step size based on the geometric regime of training.


## Why TopoAdamW?

Training neural networks with a fixed learning rate is like driving with cruise control on a mountain road. TopoAdamW acts as an intelligent co-pilot:

1. Topological awareness: probes **local geometric regimes** (e.g., flatness, sharpness, instability) of the loss landscape.
2. Dynamic adjustment: scales LR within [min_lr_ratio, max_lr_ratio] (defaults 0.2 to 1.0; raise max_lr_ratio to allow acceleration).
3. Safety lock: if center loss spikes above 2× EMA, a divergence brake overrides acceleration and reduces LR to prevent unstable updates.

Result: accelerated progress in smooth regimes and stabilized behavior in sharp or noisy regimes.

## Performance (Proof-of-Concept Results)

We benchmarked TopoAdamW against standard AdamW on CIFAR-10 using a CifarNet architecture. Both optimizers used the same base learning rate (1e-3), weight decay (5e-4), and ran for 30 epochs.

In this setup, TopoAdamW achieved a +0.91% accuracy improvement over AdamW and avoided late-epoch loss plateaus observed in the baseline.

CIFAR-10 comparison:
![CIFAR-10 Comparison](assets/comparison_results.png)

CIFAR-100 comparison (default CIFAR-100 hyperparams):
![CIFAR-100 Comparison](assets/comparison_results_cifar100.png)

### Benchmark Metrics (CIFAR-10)

The following heuristics implement a **lightweight control policy** that maps geometric indicators to learning rate scaling factors:

| Metric | AdamW (Baseline) | TopoAdamW (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| Test Accuracy | 83.49% | 84.40% | +0.91% |
| Final Val Loss | 0.4994 | 0.4713 | Better generalization |
| Late-Stage | Plateaued around ~0.50 | Continues improving to 0.4713 | Active adaptation |
| Overhead | ~12.2s / epoch | ~14.3s / epoch | ~17% increase |

> Observation: As shown in the graph, AdamW's loss curve flattens out near Epoch 20-25, while TopoAdamW continues to lower loss through the end of training.


These rules are intentionally simple and interpretable; TopoAdamW is designed as a **training dynamics controller**, not a second-order optimizer or a minimum-finding algorithm.


## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/SVAH-X/topoadamw.git
cd topoadamw

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
import torch.optim as optim
from topoadamw import TopoAdamW

# Your model
model = YourModel()
criterion = torch.nn.CrossEntropyLoss()

# Create TopoAdamW optimizer (AdamW-compatible wrapper)
optimizer = TopoAdamW(model.parameters(), model, lr=1e-3, weight_decay=5e-4)

# Training loop
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Pass data, target, criterion for topology probing
    optimizer.step(data=data, target=target, criterion=criterion)
```

Defaults: lr=1e-3, weight_decay=5e-4, interval=50, warmup_steps=150, max_lr_ratio=1.0, min_lr_ratio=0.2.

**That's it!** TopoAdamW handles learning rate adaptation automatically.

### Advanced Configuration

```python
optimizer = TopoAdamW(
    model.parameters(),
    model,
    lr=1e-3,
    weight_decay=5e-4,
    interval=50,              # Probe landscape every 50 steps
    probe_kwargs={
        'grid_size': 15,      # Landscape resolution (15x15 grid)
        'span': 0.12          # Probing region size
    },
    max_lr_ratio=1.0,         # Cap LR at base value
    min_lr_ratio=0.2,         # Keep LR above 0.2x base value
    warmup_steps=150,         # No adjustment for first 150 steps
    verbose=True              # Print LR adjustments
)
```

## 🔬 How It Works

TopoAdamW uses Topological Data Analysis (TDA) to **identify local training regimes** in the loss landscape:

1. **Probe** 🔍: Periodically samples a local neighborhood of the loss surface around current parameters  
2. **Analyze** 📐: Extracts **topology-robust geometric summaries** (e.g., persistence-based sharpness and variability indicators)  
3. **Adapt** 🎯: Adjusts the **global learning rate multiplier**, acting as a step-size controller rather than a direction optimizer

**Note:** TopoAdamW does not aim to locate exact minima of the loss function. 
Instead, it focuses on identifying geometric regimes of training and adapting step size accordingly, complementing the underlying optimizer rather than replacing it.


### Geometric Heuristics

| Landscape Type | Sharpness | Variance | LR Adjustment | Reason |
|----------------|-----------|----------|---------------|---------|
| Flat & Smooth | < 0.1 | < 0.3 | ×1.15 | Safe to accelerate |
| Very Sharp | > 0.5 | — | ×0.80 | Avoid overshooting |
| Moderately Sharp | > 0.2 | — | ×0.95 | Gentle slowdown |
| High Variance | — | > 0.5 | ×0.85 | Stabilize |
| Neutral | — | — | ×1.00 | Keep current LR |

## 📁 Repository Structure

```
topoadamw/
├── topoadamw/               # Main package
│   ├── __init__.py
│   ├── optimizer.py        # TopoAdam optimizer
│   ├── probe.py            # Loss landscape probing
│   └── tda_utils.py        # Topological analysis
├── examples/
│   ├── cifar10_example.py      # CIFAR-10 training
│   ├── cifar100_example.py     # CIFAR-100 training
│   └── custom_model_example.py # Custom model usage
├── tests/
│   └── test_optimizer.py       # Unit tests
├── generate_comparison_plot.py # Reproduce benchmark
├── requirements.txt            # Project dependencies
├── README.md
└── LICENSE
```

## 🧪 Reproducing Results

Generate the comparison plot yourself:

```bash
python generate_comparison_plot.py
```

By default, this trains AdamW and TopoAdamW on CIFAR-10 and CIFAR-100 and produces
`comparison_results.png` (CIFAR-10) and `comparison_results_cifar100.png` (also saved under `assets/`).

To run a single dataset:

```bash
python generate_comparison_plot.py --datasets cifar100
```

Run unit tests:

```bash
python -m pytest tests/
# or
python tests/test_optimizer.py
```

## 💡 Examples

### CIFAR-10 Image Classification

```bash
python examples/cifar10_example.py
```

Trains a CNN on CIFAR-10 with full data augmentation and model checkpointing.

### CIFAR-100 Image Classification

```bash
python examples/cifar100_example.py
```

Trains CifarNet on CIFAR-100 with the recommended TopoAdamW defaults.

### Custom Model

```bash
python examples/custom_model_example.py
```

Shows how to use TopoAdamW with your own model architecture and dataset.

## ⚙️ Requirements

- **Python** ≥ 3.8
- **PyTorch** ≥ 2.0
- **NumPy** ≥ 1.20
- **GUDHI** (for TDA) - `pip install gudhi`

### Installing GUDHI

```bash
# Linux / macOS
pip install gudhi

# Windows (may require conda)
conda install -c conda-forge gudhi
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Support for more base optimizers (SGD, RMSprop)
- [ ] Adaptive probing intervals
- [ ] Multi-GPU / distributed training support
- [ ] Learned geometric heuristics
- [ ] Benchmark on larger models (ResNet, ViT, Transformers)

## 📖 Citation

If you use TopoAdam in your research, please cite:

```bibtex
@software{topoadamw2025,
  author = {Congkai Peng},
  title = {TopoAdamW: Topology-Guided Learning Rate Control for Neural Network Training},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SVAH-X/topoadamw}
}
```

## 📚 References

This work builds on research in loss landscape visualization and topological data analysis:

- Li et al. "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018)
- Carlsson, G. "Topology and Data" (AMS 2009)
- Filter normalization for loss landscape visualization

## 🐛 Known Issues / Limitations

- **GPU Memory**: Landscape probing requires additional forward passes. Reduce `probe_kwargs.grid_size` if memory is limited.
- **Overhead**: Adds ~5-10% training time overhead from periodic probing.
- **Small Batches**: Works best with batch size ≥ 32. Very small batches may have noisy landscapes.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on PyTorch's optimization framework
- Uses GUDHI library for topological computations
- Inspired by modern adaptive optimizer research























