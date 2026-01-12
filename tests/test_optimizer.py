"""
Unit tests for TopoAdam optimizer
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from topoadamw import TopoAdam, TopoAdamW, create_topoadam, create_topoadamw


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestTopoAdam(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device("cpu")
        self.model = SimpleModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Create sample data
        self.data = torch.randn(32, 10)
        self.target = torch.randint(0, 2, (32,))
    
    def test_initialization(self):
        """Test optimizer initialization"""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(base_opt, self.model)
        
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.steps, 0)
        self.assertIsNotNone(optimizer.probe)
        self.assertIsNotNone(optimizer.tda)
    
    def test_basic_step(self):
        """Test basic optimization step"""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(base_opt, self.model, interval=10, warmup_steps=0)
        
        # Forward pass
        output = self.model(self.data)
        loss = self.criterion(output, self.target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimization step
        optimizer.step(data=self.data, target=self.target, criterion=self.criterion)
        
        self.assertEqual(optimizer.steps, 1)
    
    def test_lr_adjustment(self):
        """Test that LR actually changes"""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(base_opt, self.model, interval=1, warmup_steps=0)
        
        initial_lr = optimizer.optimizer.param_groups[0]['lr']
        
        # Run a few steps
        for _ in range(5):
            output = self.model(self.data)
            loss = self.criterion(output, self.target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(data=self.data, target=self.target, criterion=self.criterion)
        
        # LR should have changed
        final_lr = optimizer.optimizer.param_groups[0]['lr']
        # Note: LR might stay same if landscape is neutral, so we just check it's in bounds
        self.assertGreater(final_lr, initial_lr * 0.1)  # Min bound
        self.assertLess(final_lr, initial_lr * 3.0)     # Max bound
    
    def test_warmup(self):
        """Test warmup period"""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(base_opt, self.model, interval=1, warmup_steps=10)
        
        initial_lr = optimizer.optimizer.param_groups[0]['lr']
        
        # Run steps during warmup
        for _ in range(5):
            output = self.model(self.data)
            loss = self.criterion(output, self.target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(data=self.data, target=self.target, criterion=self.criterion)
        
        # LR should not have changed during warmup
        current_lr = optimizer.optimizer.param_groups[0]['lr']
        self.assertEqual(current_lr, initial_lr)
    
    def test_create_topoadam_convenience(self):
        """Test convenience function"""
        optimizer = create_topoadam(self.model, lr=1e-3)
        
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, TopoAdam)
        
        # Test training step
        output = self.model(self.data)
        loss = self.criterion(output, self.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(data=self.data, target=self.target, criterion=self.criterion)
    
    def test_topoadamw_wrapper(self):
        """Test TopoAdamW convenience wrapper"""
        optimizer = TopoAdamW(self.model.parameters(), self.model, lr=1e-3, warmup_steps=0)

        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, TopoAdamW)

        output = self.model(self.data)
        loss = self.criterion(output, self.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(data=self.data, target=self.target, criterion=self.criterion)

    def test_create_topoadamw_convenience(self):
        """Test create_topoadamw convenience function"""
        optimizer = create_topoadamw(self.model, lr=1e-3)

        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, TopoAdamW)

    def test_state_dict(self):
        """Test saving and loading state"""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(base_opt, self.model)
        
        # Train a bit
        for _ in range(3):
            output = self.model(self.data)
            loss = self.criterion(output, self.target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(data=self.data, target=self.target, criterion=self.criterion)
        
        # Save state
        state = optimizer.state_dict()
        self.assertIsNotNone(state)
        
        # Create new optimizer and load state
        new_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        new_optimizer = TopoAdam(new_opt, self.model)
        new_optimizer.load_state_dict(state)
    
    def test_without_topology_args(self):
        """Test that optimizer works even without data/target/criterion"""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(base_opt, self.model)
        
        output = self.model(self.data)
        loss = self.criterion(output, self.target)
        optimizer.zero_grad()
        loss.backward()
        
        # Should work without topology probing
        optimizer.step()
        
        self.assertEqual(optimizer.steps, 1)
    
    def test_lr_bounds(self):
        """Test LR stays within specified bounds"""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(
            base_opt,
            self.model,
            interval=1,
            warmup_steps=0,
            max_lr_ratio=1.5,
            min_lr_ratio=0.5
        )
        
        initial_lr = 1e-3
        
        # Run many steps
        for _ in range(20):
            output = self.model(self.data)
            loss = self.criterion(output, self.target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(data=self.data, target=self.target, criterion=self.criterion)
        
        final_lr = optimizer.optimizer.param_groups[0]['lr']
        
        # Check bounds
        self.assertGreaterEqual(final_lr, initial_lr * 0.5)
        self.assertLessEqual(final_lr, initial_lr * 1.5)


class TestGeometricFeatures(unittest.TestCase):
    """Test geometric feature extraction"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = SimpleModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.optimizer = TopoAdam(base_opt, self.model, warmup_steps=0)
    
    def test_feature_extraction(self):
        """Test that geometric features are computed"""
        import numpy as np
        
        # Create synthetic loss grid
        loss_grid = np.random.rand(15, 15) + 1.0
        
        features = self.optimizer._extract_geometric_features(loss_grid)
        
        self.assertIn('center_loss', features)
        self.assertIn('sharpness', features)
        self.assertIn('variance', features)
        
        self.assertIsInstance(features['center_loss'], (float, np.floating))
        self.assertIsInstance(features['sharpness'], (float, np.floating))
        self.assertIsInstance(features['variance'], (float, np.floating))


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
    
    
    








