import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SubspaceProbe:
    def __init__(self, model, grid_size=30, span=1.0, n_samples=16, device=None):
        self.model = model
        self.grid_size = grid_size
        self.span = span
        self.n_samples = n_samples
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except Exception:
                self.device = torch.device('cpu')
        else:
            self.device = device

    def _get_filter_normalized_direction(self):
        """
        Generate a filter-wise normalized random direction vector,
        returned as a single flat tensor of shape [D] (total param count).

        Based on filter-wise normalization from Li et al. NeurIPS 2018.
        """
        parts = []
        for param in self.model.parameters():
            d = torch.randn_like(param, device=self.device)

            if param.dim() == 4:  # Conv2D: [Out, In, K, K]
                p_norm = param.view(param.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                d_norm = d.view(d.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                d = d * (p_norm / (d_norm + 1e-10))
            elif param.dim() == 2:  # Linear: [Out, In]
                p_norm = param.norm(dim=1, keepdim=True)
                d_norm = d.norm(dim=1, keepdim=True)
                d = d * (p_norm / (d_norm + 1e-10))
            else:  # Bias or other 1D params
                p_norm = param.norm()
                d_norm = d.norm()
                d = d * (p_norm / (d_norm + 1e-10))

            parts.append(d.reshape(-1))

        return torch.cat(parts)  # [D]

    def probe(self, data, target, criterion):
        """
        Sparse landscape probe: evaluates center + 8 neighbors + n_samples
        random points.

        Total forward passes: 1 + 8 + n_samples  (default: 25)
        vs. the old dense grid:                   grid_size² (e.g. 225 for 15×15)

        Neighbor step size matches the original grid spacing:
            step = 2 * span / (grid_size - 1)

        Returns:
            dict with keys:
                'center'   : float            — loss at (0, 0)
                'neighbors': list[float] × 8  — losses at the 8 adjacent points
                'samples'  : list[float]       — losses at n_samples random points
        """
        was_training = self.model.training
        self.model.eval()

        dir1 = self._get_filter_normalized_direction()
        dir2 = self._get_filter_normalized_direction()

        # Neighbor step = one grid-cell width in the original dense grid
        step = 2.0 * self.span / (self.grid_size - 1)

        # Build the evaluation coordinates: center, 8 neighbors, random samples
        coords = [(0.0, 0.0)]  # center
        coords += [
            ( step,    0.0), (-step,    0.0),
            ( 0.0,    step), ( 0.0,   -step),
            ( step,   step), (-step,   step),
            ( step,  -step), (-step,  -step),
        ]
        if self.n_samples > 0:
            u_rand = ((torch.rand(self.n_samples, device=self.device) * 2 - 1) * self.span).tolist()
            v_rand = ((torch.rand(self.n_samples, device=self.device) * 2 - 1) * self.span).tolist()
            coords += list(zip(u_rand, v_rand))

        # Backup as a single flat vector
        orig_vec = parameters_to_vector(self.model.parameters()).detach().clone()
        # Single reusable buffer — zero allocations inside the loop
        buf = torch.empty_like(orig_vec)
        losses = []

        try:
            with torch.no_grad():
                for u, v in coords:
                    # buf = orig + u*dir1 + v*dir2
                    torch.add(orig_vec, dir1, alpha=u, out=buf)
                    buf.add_(dir2, alpha=v)
                    vector_to_parameters(buf, self.model.parameters())
                    out = self.model(data)
                    loss = criterion(out, target)
                    losses.append(loss.item())
        except Exception as e:
            print(f"[Probe Error] {e}")
            raise e
        finally:
            with torch.no_grad():
                vector_to_parameters(orig_vec, self.model.parameters())
            self.model.train(mode=was_training)

        return {
            'center':    losses[0],
            'neighbors': losses[1:9],
            'samples':   losses[9:],
        }

    def grid_probe(self, data, target, criterion, grid_size: int = 7):
        """
        Dense grid probe for TDA.

        Evaluates loss on a (grid_size × grid_size) regular grid of parameter
        perturbations and returns a (grid_size × grid_size) numpy float array.

        Total forward passes: grid_size²  (default 49 for grid_size=7).
        The same vectorised flat-buffer approach as probe() is used, with the
        row-base optimisation to halve the number of full vector additions.
        """
        was_training = self.model.training
        self.model.eval()

        dir1 = self._get_filter_normalized_direction()
        dir2 = self._get_filter_normalized_direction()

        coords    = torch.linspace(-self.span, self.span, grid_size, device=self.device)
        loss_grid = np.zeros((grid_size, grid_size))

        orig_vec = parameters_to_vector(self.model.parameters()).detach().clone()
        row_base = torch.empty_like(orig_vec)
        buf      = torch.empty_like(orig_vec)

        try:
            with torch.no_grad():
                for i in range(grid_size):
                    u = coords[i].item()
                    torch.add(orig_vec, dir1, alpha=u, out=row_base)
                    for j in range(grid_size):
                        v = coords[j].item()
                        torch.add(row_base, dir2, alpha=v, out=buf)
                        vector_to_parameters(buf, self.model.parameters())
                        out  = self.model(data)
                        loss = criterion(out, target)
                        loss_grid[i, j] = loss.item()
        except Exception as e:
            print(f"[Probe Error] {e}")
            raise e
        finally:
            with torch.no_grad():
                vector_to_parameters(orig_vec, self.model.parameters())
            self.model.train(mode=was_training)

        return loss_grid
