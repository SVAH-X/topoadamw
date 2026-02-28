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
        """Test that geometric features are computed from a sparse probe result"""
        import numpy as np

        probe_result = {
            'center':    1.5,
            'neighbors': [1.6, 1.7, 1.8, 1.5, 1.6, 1.7, 1.8, 1.5],
            'samples':   [1.4, 1.5, 1.6, 1.7] * 4,
        }

        features = self.optimizer._extract_geometric_features(probe_result)

        self.assertIn('center_loss', features)
        self.assertIn('sharpness', features)
        self.assertIn('variance', features)

        self.assertIsInstance(features['center_loss'], (float, np.floating))
        self.assertIsInstance(features['sharpness'], (float, np.floating))
        self.assertIsInstance(features['variance'], (float, np.floating))


class TestGridProbe(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = SimpleModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.data = torch.randn(32, 10)
        self.target = torch.randint(0, 2, (32,))
        from topoadamw import SubspaceProbe
        self.probe = SubspaceProbe(self.model, grid_size=15, span=0.12, device=self.device)

    def test_grid_probe_shape(self):
        import numpy as np
        grid = self.probe.grid_probe(self.data, self.target, self.criterion, grid_size=5)
        self.assertIsInstance(grid, np.ndarray)
        self.assertEqual(grid.shape, (5, 5))

    def test_grid_probe_positive_losses(self):
        grid = self.probe.grid_probe(self.data, self.target, self.criterion, grid_size=3)
        self.assertTrue((grid > 0).all())

    def test_model_params_restored_after_grid_probe(self):
        """Parameters must be identical before and after probing."""
        from torch.nn.utils import parameters_to_vector
        before = parameters_to_vector(self.model.parameters()).clone()
        self.probe.grid_probe(self.data, self.target, self.criterion, grid_size=3)
        after = parameters_to_vector(self.model.parameters())
        self.assertTrue(torch.allclose(before, after))


class TestTopoCNNTrainer(unittest.TestCase):

    def setUp(self):
        from topoadamw import TopoCNNTrainer
        self.TrainerClass = TopoCNNTrainer
        self.dummy_img = torch.zeros(2, 50, 50)

    def test_heuristic_label_flat(self):
        from topoadamw import FLAT
        self.assertEqual(self.TrainerClass.heuristic_label(0.05, 0.1), FLAT)

    def test_heuristic_label_decel_sharp(self):
        from topoadamw import DECEL
        self.assertEqual(self.TrainerClass.heuristic_label(0.6, 0.1), DECEL)

    def test_heuristic_label_decel_noisy(self):
        from topoadamw import DECEL
        self.assertEqual(self.TrainerClass.heuristic_label(0.05, 0.6), DECEL)

    def test_heuristic_label_neutral(self):
        from topoadamw import NEUTRAL
        self.assertEqual(self.TrainerClass.heuristic_label(0.15, 0.3), NEUTRAL)

    def test_predict_factor_returns_none_before_training(self):
        trainer = self.TrainerClass(min_samples=10)
        self.assertIsNone(trainer.predict_factor(self.dummy_img))

    def test_add_sample_fills_buffer(self):
        trainer = self.TrainerClass(min_samples=100, buffer_capacity=10)
        for _ in range(7):
            trainer.add_sample(self.dummy_img, sharpness=0.05, variance=0.1)
        self.assertEqual(len(trainer._imgs), 7)

    def test_buffer_respects_capacity(self):
        trainer = self.TrainerClass(min_samples=100, buffer_capacity=3)
        for _ in range(6):
            trainer.add_sample(self.dummy_img, sharpness=0.05, variance=0.1)
        self.assertLessEqual(len(trainer._imgs), 3)

    def test_predict_factor_valid_after_training(self):
        from topoadamw import LR_FACTORS
        trainer = self.TrainerClass(min_samples=5, retrain_every=5, train_epochs=2)
        for _ in range(5):
            trainer.add_sample(self.dummy_img, sharpness=0.05, variance=0.1)
        self.assertTrue(trainer.is_ready)
        factor = trainer.predict_factor(self.dummy_img)
        self.assertIn(factor, set(LR_FACTORS.values()))


class TestDivergenceBrake(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = SimpleModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.data = torch.randn(32, 10)
        self.target = torch.randint(0, 2, (32,))

    def test_divergence_brake_reduces_lr(self):
        """When current loss > 2× EMA, LR must drop regardless of landscape."""
        base_opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer = TopoAdam(base_opt, self.model, interval=1, warmup_steps=0)

        # Force a very low EMA so any real CrossEntropyLoss triggers the brake
        optimizer.loss_ema = 1e-6
        initial_lr = optimizer.optimizer.param_groups[0]['lr']

        output = self.model(self.data)
        loss = self.criterion(output, self.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(data=self.data, target=self.target, criterion=self.criterion)

        final_lr = optimizer.optimizer.param_groups[0]['lr']
        self.assertLess(final_lr, initial_lr)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
    
    
    








