import torch
import numpy as np
import copy

class SubspaceProbe:
    def __init__(self, model, grid_size=30, span=1.0, device=None):
        self.model = model
        self.grid_size = grid_size
        self.span = span
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except:
                self.device = torch.device('cpu')
        else:
            self.device = device

    def _get_filter_normalized_direction(self):
        """
        根据论文 Section 3 和 8.3 实现 Filter-Wise Normalization。
        生成与模型参数形状一致的随机方向向量，并根据参数的范数进行归一化。
        """
        direction = []
        for param in self.model.parameters():
            d = torch.randn_like(param, device=self.device)
            
            # Filter-wise normalization logic [cite: 514-533]
            if param.dim() == 4: # Conv2D: [Out, In, K, K]
                # 对每个输出 Filter (dim 0) 独立归一化
                p_norm = param.view(param.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                d_norm = d.view(d.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                d = d * (p_norm / (d_norm + 1e-10))
            elif param.dim() == 2: # Linear: [Out, In]
                # 对每个神经元 (dim 0) 独立归一化
                p_norm = param.norm(dim=1, keepdim=True)
                d_norm = d.norm(dim=1, keepdim=True)
                d = d * (p_norm / (d_norm + 1e-10))
            else: # Bias 或其他 1D 参数
                p_norm = param.norm()
                d_norm = d.norm()
                d = d * (p_norm / (d_norm + 1e-10))
            
            direction.append(d)
        return direction

    def probe(self, data, target, criterion):
        """
        生成 Loss Landscape Grid。
        """
        was_training = self.model.training
        self.model.eval() # 冻结 BN
        
        # 1. 生成两个正交的、经过 Filter-Norm 的方向向量 [cite: 500-501]
        dir1 = self._get_filter_normalized_direction()
        dir2 = self._get_filter_normalized_direction()
        
        # 2. 准备网格坐标
        coords = torch.linspace(-self.span, self.span, self.grid_size, device=self.device)
        # 注意：这里我们手动循环，避免生成巨大的 Meshgrid 显存爆炸
        
        loss_grid = np.zeros((self.grid_size, self.grid_size))
        
        # 备份原始参数 (为了速度，不使用 deepcopy model，只备份参数 tensor)
        orig_params = [p.clone() for p in self.model.parameters()]
        
        try:
            with torch.no_grad():
                for i in range(self.grid_size):
                    u = coords[i]
                    for j in range(self.grid_size):
                        v = coords[j]
                        
                        # Apply Perturbation: theta + u*d1 + v*d2 [cite: 547]
                        # 直接原地修改参数，速度最快
                        for idx, param in enumerate(self.model.parameters()):
                            # param.data = orig + u * dir1 + v * dir2
                            # 使用 addcmul 优化性能: param = orig + u*d1
                            # 这里为了清晰写成标准加法
                            perturbation = u * dir1[idx] + v * dir2[idx]
                            param.copy_(orig_params[idx] + perturbation)
                        
                        # Compute Loss
                        out = self.model(data)
                        loss = criterion(out, target)
                        loss_grid[i, j] = loss.item()
        
        except Exception as e:
            print(f"[Probe Error] {e}")
            raise e
        finally:
            # 恢复原始参数
            with torch.no_grad():
                for idx, param in enumerate(self.model.parameters()):
                    param.copy_(orig_params[idx])
            self.model.train(mode=was_training)
        
        return loss_grid
    
    
    
    