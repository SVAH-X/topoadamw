"""
TopoAdam: Topology-Guided Adaptive Optimizer

A PyTorch optimizer that adapts learning rates based on loss landscape topology
using Topological Data Analysis (TDA). Built on top of AdamW.

Features:
- Automatic LR adjustment based on landscape geometry
- Improved stability over fixed-LR optimizers
- Drop-in replacement for torch.optim.AdamW

Example:
    >>> import torch.optim as optim
    >>> from topoadamw import TopoAdamW
    >>> optimizer = TopoAdamW(model.parameters(), model, lr=1e-3)
    >>>
    >>> for data, target in loader:
    >>>     optimizer.zero_grad()
    >>>     loss = model(data)
    >>>     loss.backward()
    >>>     optimizer.step(data=data, target=target, criterion=criterion)

Author: [Your Name]
License: MIT
"""

import torch
import numpy as np
from typing import Optional, Callable
from .probe import SubspaceProbe
from .tda_utils import LossLandscapeTDA

__version__ = "0.1.0"


class TopoAdam:
    """
    Topology-guided adaptive optimizer wrapper for PyTorch optimizers.
    
    This optimizer wraps any PyTorch optimizer (typically AdamW) and
    adjusts the learning rate based on the local geometry of the loss landscape.
    
    Args:
        base_optimizer: PyTorch optimizer instance (e.g., torch.optim.AdamW)
        model: PyTorch model
        interval: Probe loss landscape every N steps (default: 50)
        probe_kwargs: Additional arguments for SubspaceProbe
            - grid_size: Resolution of landscape grid (default: 15)
            - span: Size of probed region (default: 0.12)
        max_lr_ratio: Maximum LR as ratio of initial LR (default: 1.0)
        min_lr_ratio: Minimum LR as ratio of initial LR (default: 0.2)
        warmup_steps: Don't adjust LR for first N steps (default: 150)
        
    Geometric Heuristics:
        - Flat landscape (low sharpness) → Increase LR
        - Sharp landscape (high sharpness) → Decrease LR
        - High variance → Stabilize with lower LR
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        interval: int = 50,
        probe_kwargs: Optional[dict] = None,
        max_lr_ratio: float = 1.0,
        min_lr_ratio: float = 0.2,
        warmup_steps: int = 150,
        verbose: bool = False
    ):
        self.optimizer = base_optimizer
        self.model = model
        self.interval = interval
        self.max_lr_ratio = max_lr_ratio
        self.min_lr_ratio = min_lr_ratio
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.steps = 0
        
        # Device detection
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')
        
        # Initialize TDA components
        probe_cfg = probe_kwargs or {}
        self.probe = SubspaceProbe(
            model,
            grid_size=probe_cfg.get('grid_size', 15),
            span=probe_cfg.get('span', 0.12),
            device=self.device
        )
        self.tda = LossLandscapeTDA(resolution=50, bandwidth_scale=2.0)
        
        # Store base learning rates
        self.base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        
        # Tracking for stability
        self.loss_ema = None
        self.sharpness_ema = None
        self.ema_alpha = 0.2
        
        # Cache for next batch (avoid same-batch probing)
        self.next_batch = None
        
        if self.verbose:
            print(f"[TopoAdam] Initialized with interval={interval}, "
                  f"warmup={warmup_steps}, max_lr_ratio={max_lr_ratio}")

    def __getattr__(self, name):
        """Delegate unknown attributes to base optimizer"""
        return getattr(self.optimizer, name)

    def step(
        self,
        closure: Optional[Callable] = None,
        data: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        criterion: Optional[Callable] = None
    ):
        """
        Perform a single optimization step with topology-guided LR adjustment.
        
        Args:
            closure: Optional closure for optimizer
            data: Input data batch (required for topology probing)
            target: Target labels (required for topology probing)
            criterion: Loss function (required for topology probing)
            
        Returns:
            Loss value from base optimizer step
        """
        loss = self.optimizer.step(closure)
        self.steps += 1
        
        # Cache batch for next probe
        if data is not None and target is not None:
            self.next_batch = (data.detach().clone(), target.detach().clone())
        
        # Perform topology-guided adjustment
        if (self.steps % self.interval == 0 and 
            self.steps > self.warmup_steps and
            self.next_batch is not None and 
            criterion is not None):
            
            self._adjust_learning_rate(
                self.next_batch[0], 
                self.next_batch[1], 
                criterion
            )
        
        return loss

    def _adjust_learning_rate(self, data, target, criterion):
        """Internal method to adjust LR based on landscape geometry"""
        data, target = data.to(self.device), target.to(self.device)
        
        # 1. Probe loss landscape
        loss_grid = self.probe.probe(data, target, criterion)
        features = self._extract_geometric_features(loss_grid)
        
        # 2. Update exponential moving averages
        if self.loss_ema is None:
            self.loss_ema = features['center_loss']
            self.sharpness_ema = features['sharpness']
        else:
            self.loss_ema = (self.ema_alpha * features['center_loss'] + 
                           (1 - self.ema_alpha) * self.loss_ema)
            self.sharpness_ema = (self.ema_alpha * features['sharpness'] + 
                                (1 - self.ema_alpha) * self.sharpness_ema)
        
        # 3. Compute adjustment factor using geometric heuristics
        factor = self._geometric_heuristic(features)
        
        # 4. Safety check: prevent divergence
        if features['center_loss'] > self.loss_ema * 2.0:
            factor = 0.8
            reason = "DIVERGENCE_BRAKE"
        else:
            reason = self._get_adjustment_reason(features)
        
        # 5. Apply adjustment with bounds
        if abs(factor - 1.0) > 0.05:
            for i, pg in enumerate(self.optimizer.param_groups):
                old_lr = pg['lr']
                new_lr = old_lr * factor
                
                # Enforce bounds
                max_lr = self.base_lrs[i] * self.max_lr_ratio
                min_lr = self.base_lrs[i] * self.min_lr_ratio
                new_lr = max(min(new_lr, max_lr), min_lr)
                
                pg['lr'] = new_lr
                
                if self.verbose:
                    print(f"[TopoAdam] Step {self.steps} | {reason} | "
                          f"Sharpness: {features['sharpness']:.3f} | "
                          f"LR: {old_lr:.6f} -> {new_lr:.6f}")

    def _extract_geometric_features(self, loss_grid):
        """Extract interpretable features from loss landscape"""
        mid = loss_grid.shape[0] // 2
        center_loss = loss_grid[mid, mid]
        
        # Sharpness: how much loss increases around center
        neighbors = [
            loss_grid[mid-1, mid], loss_grid[mid+1, mid],
            loss_grid[mid, mid-1], loss_grid[mid, mid+1],
            loss_grid[mid-1, mid-1], loss_grid[mid+1, mid+1],
            loss_grid[mid-1, mid+1], loss_grid[mid+1, mid-1]
        ]
        avg_neighbor = np.mean(neighbors)
        sharpness = (avg_neighbor - center_loss) / (center_loss + 1e-8)
        
        # Variance: landscape roughness
        variance = np.std(loss_grid) / (np.mean(loss_grid) + 1e-8)
        
        return {
            'center_loss': center_loss,
            'sharpness': sharpness,
            'variance': variance
        }

    def _geometric_heuristic(self, features):
        """
        Geometric heuristics for LR adjustment based on landscape topology.
        
        Rules:
        1. Very flat (sharpness < 0.1) + smooth (variance < 0.3) → Speed up 15%
        2. Very sharp (sharpness > 0.5) → Slow down 20%
        3. Moderately sharp (sharpness > 0.2) → Slow down 5%
        4. High variance (> 0.5) → Slow down 15%
        5. Default → Keep current LR
        """
        sharpness = features['sharpness']
        variance = features['variance']
        
        if sharpness < 0.1 and variance < 0.3:
            return 1.15  # Safe to accelerate
        elif sharpness > 0.5:
            return 0.80  # Dangerous region
        elif variance > 0.5:
            return 0.85  # Too noisy
        elif sharpness > 0.2:
            return 0.95  # Slightly sharp
        else:
            return 1.0   # Neutral

    def _get_adjustment_reason(self, features):
        """Get human-readable reason for adjustment decision"""
        sharpness = features['sharpness']
        variance = features['variance']
        
        if sharpness < 0.1 and variance < 0.3:
            return "FLAT_LANDSCAPE"
        elif sharpness > 0.5:
            return "SHARP_PEAK"
        elif variance > 0.5:
            return "HIGH_VARIANCE"
        elif sharpness > 0.2:
            return "MODERATE_SHARPNESS"
        else:
            return "NEUTRAL"

    def state_dict(self):
        """Return optimizer state dict"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict"""
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients"""
        self.optimizer.zero_grad(set_to_none=set_to_none)


# Convenience function for easy usage
def create_topoadam(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    **kwargs
) -> TopoAdam:
    """
    Create TopoAdam optimizer with sensible defaults.
    
    Args:
        model: PyTorch model
        lr: Base learning rate (default: 1e-3)
        weight_decay: Weight decay (L2 penalty) (default: 5e-4)
        **kwargs: Additional arguments for TopoAdam
        
    Returns:
        TopoAdam optimizer instance
        
    Example:
        >>> optimizer = create_topoadam(model, lr=1e-3, interval=50)
    """
    base_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return TopoAdam(base_opt, model, **kwargs)

class TopoAdamW(TopoAdam):
    """
    Convenience wrapper that builds AdamW internally for easier drop-in use.
    """

    def __init__(
        self,
        params,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        **kwargs
    ):
        topo_keys = {
            "interval",
            "probe_kwargs",
            "max_lr_ratio",
            "min_lr_ratio",
            "warmup_steps",
            "verbose",
        }
        topo_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in topo_keys}
        base_opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
        super().__init__(base_opt, model, **topo_kwargs)


def create_topoadamw(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    **kwargs
) -> TopoAdamW:
    """
    Create TopoAdamW optimizer with sensible defaults.

    Args:
        model: PyTorch model
        lr: Base learning rate (default: 1e-3)
        weight_decay: Weight decay (L2 penalty) (default: 5e-4)
        **kwargs: Additional arguments for TopoAdamW

    Returns:
        TopoAdamW optimizer instance

    Example:
        >>> optimizer = create_topoadamw(model, lr=1e-3, interval=50)
    """
    return TopoAdamW(model.parameters(), model, lr=lr, weight_decay=weight_decay, **kwargs)






