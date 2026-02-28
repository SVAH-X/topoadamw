import numpy as np
import torch

class LossLandscapeTDA:
    def __init__(self, resolution=50, bandwidth_scale=2.0):
        self.resolution = resolution
        # Larger bandwidth spreads each point's Gaussian, making features more visible
        self.sigma = bandwidth_scale / resolution

        # Coordinate grid for persistence image rendering
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.xv, self.yv = np.meshgrid(x, y)
        self.sigma_sq = 2 * (self.sigma ** 2)

        # Sensitivity cap: a loss ratio of max_ratio_cap maps to the image corner.
        # At 5.0, a 5x loss increase pushes a point to (1, 1); moderate changes
        # still spread across the image rather than clustering near the origin.
        self.max_ratio_cap = 5.0

    def compute_persistence_image(self, loss_grid):
        """
        High-Sensitivity Ratio-Based TDA.
        """
        try:
            import gudhi as gd
        except ImportError as exc:
            raise ImportError(
                "gudhi is required for TDA computation. Install it with: pip install gudhi"
            ) from exc

        # 1. Base values
        mid = loss_grid.shape[0] // 2
        center_loss = loss_grid[mid, mid]
        if center_loss <= 1e-8: center_loss = 1e-8

        # 2. Compute persistence via cubical complex
        cc = gd.CubicalComplex(dimensions=loss_grid.shape,
                               top_dimensional_cells=loss_grid.flatten())
        persistence = cc.persistence(homology_coeff_field=2, min_persistence=0)

        h0 = np.array([p[1] for p in persistence if p[0] == 0])
        if len(h0) == 0: return torch.zeros((2, self.resolution, self.resolution))

        # Replace infinite death values with the grid maximum
        L_max = np.max(loss_grid)
        h0[np.isinf(h0[:, 1]), 1] = L_max

        # 3. Log-ratio coordinate transform (high sensitivity)

        # X-axis: downhill depth — log(center / birth)
        ratio_down = center_loss / (h0[:, 0] + 1e-8)
        ratio_down = np.maximum(ratio_down, 1.0)
        x_coords = np.log(ratio_down) / np.log(self.max_ratio_cap)

        # Y-axis: uphill height — log(death / center)
        ratio_up = h0[:, 1] / (center_loss + 1e-8)
        ratio_up = np.maximum(ratio_up, 1.0)
        y_coords = np.log(ratio_up) / np.log(self.max_ratio_cap)

        # Clip to [0, 1]
        x_coords = np.clip(x_coords, 0, 1)
        y_coords = np.clip(y_coords, 0, 1)

        # 4. Render persistence image
        pixel_grid = np.zeros((self.resolution, self.resolution))

        # Weight by distance from origin: steeper features are brighter
        pers_weights = np.sqrt(x_coords**2 + y_coords**2)
        if pers_weights.max() > 0:
            pers_weights /= pers_weights.max()

        for x_c, y_c, w in zip(x_coords, y_coords, pers_weights):
            gauss = np.exp(-((self.xv - x_c)**2 + (self.yv - y_c)**2) / self.sigma_sq)
            pixel_grid += w * gauss

        # Flip vertically (persistence image convention) and normalize
        pixel_grid = np.flipud(pixel_grid).copy()

        if pixel_grid.max() > 0:
            pixel_grid /= pixel_grid.max()
            
        tensor_img = torch.tensor(pixel_grid, dtype=torch.float32)
        # 堆叠成双通道，兼容 TopoCNN 输入
        return torch.stack([tensor_img, tensor_img])
    
    
    
    