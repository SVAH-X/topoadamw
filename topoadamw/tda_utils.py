import numpy as np
import torch

class LossLandscapeTDA:
    def __init__(self, resolution=50, bandwidth_scale=2.0):
        self.resolution = resolution
        # 增大光斑尺寸，让特征更明显
        self.sigma = bandwidth_scale / resolution 
        
        # 坐标网格
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.xv, self.yv = np.meshgrid(x, y)
        self.sigma_sq = 2 * (self.sigma ** 2)
        
        # --- 🔧 关键参数：灵敏度 ---
        # 设为 5.0。意思是只要 Loss 变成原来的 5 倍，点就飞到最右上角。
        # 这样对于温和的 Loss 变化，点也会散开，不会挤在左下角。
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

        # 1. 基础数据
        mid = loss_grid.shape[0] // 2
        center_loss = loss_grid[mid, mid]
        if center_loss <= 1e-8: center_loss = 1e-8

        # 2. TDA 计算
        cc = gd.CubicalComplex(dimensions=loss_grid.shape,
                               top_dimensional_cells=loss_grid.flatten())
        persistence = cc.persistence(homology_coeff_field=2, min_persistence=0)
        
        h0 = np.array([p[1] for p in persistence if p[0] == 0])
        if len(h0) == 0: return torch.zeros((2, self.resolution, self.resolution))
        
        # 处理无穷大
        L_max = np.max(loss_grid)
        h0[np.isinf(h0[:, 1]), 1] = L_max
        
        # 3. 坐标变换 (高灵敏度)
        
        # X轴: 下坡深度 log(Center / Birth)
        # 加上 1e-8 防止除以0
        ratio_down = center_loss / (h0[:, 0] + 1e-8)
        # 限制比率范围 [1.0, max_ratio]
        ratio_down = np.maximum(ratio_down, 1.0)
        x_coords = np.log(ratio_down) / np.log(self.max_ratio_cap)
        
        # Y轴: 上坡高度 log(Death / Center)
        ratio_up = h0[:, 1] / (center_loss + 1e-8)
        ratio_up = np.maximum(ratio_up, 1.0)
        y_coords = np.log(ratio_up) / np.log(self.max_ratio_cap)
        
        # 截断到 [0, 1]
        x_coords = np.clip(x_coords, 0, 1)
        y_coords = np.clip(y_coords, 0, 1)

        # 4. 生成图片
        pixel_grid = np.zeros((self.resolution, self.resolution))
        
        # 权重：越陡峭越亮
        pers_weights = np.sqrt(x_coords**2 + y_coords**2) # 离原点越远越亮
        if pers_weights.max() > 0:
            pers_weights /= pers_weights.max()
            
        for x_c, y_c, w in zip(x_coords, y_coords, pers_weights):
            gauss = np.exp(-((self.xv - x_c)**2 + (self.yv - y_c)**2) / self.sigma_sq)
            pixel_grid += w * gauss
            
        # 翻转 + Copy
        pixel_grid = np.flipud(pixel_grid).copy()
        
        # 归一化亮度
        if pixel_grid.max() > 0:
            pixel_grid /= pixel_grid.max()
            
        tensor_img = torch.tensor(pixel_grid, dtype=torch.float32)
        # 堆叠成双通道，兼容 TopoCNN 输入
        return torch.stack([tensor_img, tensor_img])
    
    
    
    