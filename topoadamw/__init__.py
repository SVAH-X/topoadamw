"""
TopoAdam: Topology-Guided Adaptive Optimizer

A PyTorch optimizer that adapts learning rates based on loss landscape
topology using Topological Data Analysis (TDA).

Example:
    >>> from topoadamw import TopoAdamW
    >>> import torch.optim as optim
    >>> 
    >>> optimizer = TopoAdamW(model.parameters(), model, lr=1e-3)
    >>> 
    >>> # Training loop
    >>> for data, target in loader:
    >>>     optimizer.zero_grad()
    >>>     loss = criterion(model(data), target)
    >>>     loss.backward()
    >>>     optimizer.step(data=data, target=target, criterion=criterion)
"""

from .optimizer import TopoAdam, TopoAdamW, create_topoadam, create_topoadamw
from .probe import SubspaceProbe
from .tda_utils import LossLandscapeTDA

__version__ = "0.1.0"
__author__ = "Congkai Peng"
__license__ = "MIT"

__all__ = [
    "TopoAdam",
    "TopoAdamW",
    "create_topoadam",
    "create_topoadamw",
    "SubspaceProbe",
    "LossLandscapeTDA"
]

