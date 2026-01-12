import torch
import torch.nn as nn

class TopoCNN(nn.Module):
    def __init__(self):
        super(TopoCNN, self).__init__()
        
        # 输入: (Batch, 2, 50, 50) -> H0 和 H1 两个通道
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 25x25
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 12x12
            
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Global Average Pooling: 不管图里特征在哪，只管有没有
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # 输出 3 类: 0=减速, 1=保持, 2=加速
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    
    
class CifarNet(nn.Module):
    """
    一个用于 CIFAR-10 的轻量级卷积网络 (Target Model)。
    比 SimpleNet 强，但比 ResNet 弱，方便观察优化器的影响。
    """
    def __init__(self):
        super(CifarNet, self).__init__()
        self.features = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    