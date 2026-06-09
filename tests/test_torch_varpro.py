"""Unit tests for torch_varpro."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch_varpro import (
    VarProClassifier,
    VarProRegressor,
    IncrementalRidgeReadout,
    ProximalRidgeReadout,
    augment_features,
    fit_readout_from_loader,
    solve_ce_readout,
    solve_ridge_readout,
    solve_sparse_readout,
    varpro_ce_step,
    varpro_incremental_step,
    varpro_proximal_step,
    varpro_sparse_step,
    varpro_step,
)


def _small_net():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _feature_net():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU())


class TestSolvers(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_ridge_readout_shapes(self):
        F = torch.randn(16, 6)
        Y = torch.randn(16, 3)
        W_bias = solve_ridge_readout(F, Y, ridge=1e-3, bias=True)
        W_no_bias = solve_ridge_readout(F, Y, ridge=1e-3, bias=False)
        self.assertEqual(W_bias.shape, (7, 3))
        self.assertEqual(W_no_bias.shape, (6, 3))

    def test_ridge_readout_zero_ridge_uses_pseudoinverse(self):
        F = torch.randn(20, 4)
        Y = torch.randn(20, 2)
        W = solve_ridge_readout(F, Y, ridge=0.0, bias=False)
        self.assertEqual(W.shape, (4, 2))
        self.assertTrue(torch.isfinite(W).all())

    def test_sparse_readout_l1_shape(self):
        F = torch.randn(12, 4)
        Y = torch.randn(12, 2)
        W = solve_sparse_readout(F, Y, sparsity=1e-2, penalty="l1", bias=True, max_iter=20)
        self.assertEqual(W.shape, (5, 2))
        self.assertTrue(torch.isfinite(W).all())

    def test_sparse_readout_l1half_irls(self):
        F = torch.randn(12, 4)
        Y = torch.randn(12, 2)
        W = solve_sparse_readout(F, Y, sparsity=1e-2, penalty="l1/2", bias=True,
                                 max_iter=10, solver="irls")
        self.assertEqual(W.shape, (5, 2))
        self.assertTrue(torch.isfinite(W).all())

    def test_ridge_readout_gradient_path(self):
        F = torch.randn(12, 4, requires_grad=True)
        Y = torch.randn(12, 2)
        W = solve_ridge_readout(F, Y, ridge=1e-3, bias=True, detach_features=False)
        loss = (augment_features(F, bias=True) @ W).square().mean()
        loss.backward()
        self.assertIsNotNone(F.grad)
        self.assertGreater(F.grad.norm().item(), 0.0)

    def test_ce_readout_shapes(self):
        F = torch.randn(10, 6)
        y = torch.randint(0, 3, (10,))
        W, b = solve_ce_readout(F, y, num_classes=3, ridge=1e-2, max_iter=5)
        self.assertEqual(W.shape, (6, 3))
        self.assertEqual(b.shape, (3,))

    def test_ce_readout_newton_cg(self):
        F = torch.randn(10, 6)
        y = torch.randint(0, 3, (10,))
        W, b = solve_ce_readout(F, y, num_classes=3, ridge=1e-2, max_iter=3,
                                 solver="newton_cg", cg_max_iter=5)
        self.assertEqual(W.shape, (6, 3))
        self.assertTrue(torch.isfinite(W).all())


class TestVarProRegressor(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_forward_shape(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        X = torch.randn(10, 4)
        self.assertEqual(model(X).shape, (10, 2))

    def test_step_updates_feature_net(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        optimizer = torch.optim.SGD(model.feature_net.parameters(), lr=1e-2)
        X = torch.randn(16, 4)
        Y = torch.randn(16, 2)
        before = [p.detach().clone() for p in model.feature_net.parameters()]
        loss = model.step(X, Y, optimizer, ridge=1e-3)
        after = list(model.feature_net.parameters())
        self.assertIsInstance(loss, float)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(after, before)))
        self.assertTrue(model.W.requires_grad)

    def test_sparse_step_produces_sparse_readout(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = torch.randn(32, 4)
        Y = torch.randn(32, 2)
        for _ in range(5):
            model.step(X, Y, optimizer, sparse_penalty="l1", sparsity=1.0, ridge=0.0)
        sparsity_ratio = (model.W.abs() < 1e-6).float().mean().item()
        self.assertGreater(sparsity_ratio, 0.1)

    def test_sparse_step_updates_readout_after_inner_solve(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        X = torch.randn(16, 4)
        Y = torch.randn(16, 2)
        projected = solve_sparse_readout(
            model.feature_net(X).detach(), Y, sparsity=1e-2, ridge=1e-3,
            bias=True, max_iter=10,
        )
        model.step(X, Y, optimizer, sparse_penalty="l1", sparsity=1e-2,
                   ridge=1e-3, sparse_max_iter=10)
        actual = torch.cat([model.W.detach(), model.b.detach().unsqueeze(0)], dim=0)
        self.assertFalse(torch.allclose(actual, projected))

    def test_sparse_step_requires_readout_in_optimizer(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        optimizer = torch.optim.SGD(model.feature_net.parameters(), lr=1e-2)
        with self.assertRaisesRegex(ValueError, "must include the readout"):
            model.step(torch.randn(8, 4), torch.randn(8, 2), optimizer,
                       sparse_penalty="l1")

    def test_fit_readout_from_loader(self):
        feature_net = _feature_net()
        model = VarProRegressor(feature_net, feature_dim=8, output_dim=2)
        X = torch.randn(20, 4)
        Y = torch.randn(20, 2)
        loader = DataLoader(TensorDataset(X, Y), batch_size=5)
        features = model.feature_net(X).detach()
        W_full = solve_ridge_readout(features, Y, ridge=1e-3, bias=True)
        W_by_loader = model.fit_readout(loader, ridge=1e-3)
        self.assertTrue(torch.allclose(W_by_loader, W_full, atol=1e-5))

    def test_make_incremental_state(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        state = model.make_incremental_state(ridge=1e-3)
        self.assertIsInstance(state, IncrementalRidgeReadout)
        self.assertEqual(state.feature_dim, 8)

    def test_make_proximal_state(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        state = model.make_proximal_state(prox_strength=1.0)
        self.assertIsInstance(state, ProximalRidgeReadout)


class TestVarProClassifier(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_forward_shape(self):
        model = VarProClassifier(_feature_net(), feature_dim=8, num_classes=3)
        X = torch.randn(9, 4)
        self.assertEqual(model(X).shape, (9, 3))

    def test_step_updates_feature_net(self):
        model = VarProClassifier(_feature_net(), feature_dim=8, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = torch.randn(9, 4)
        y = torch.randint(0, 3, (9,))
        before = [p.detach().clone() for p in model.feature_net.parameters()]
        loss = model.step(X, y, optimizer, ridge=1e-2, inner_iter=3)
        after = list(model.feature_net.parameters())
        self.assertIsInstance(loss, float)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(after, before)))
        self.assertTrue(model.W.requires_grad)

    def test_step_newton_cg(self):
        model = VarProClassifier(_feature_net(), feature_dim=8, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = torch.randn(9, 4)
        y = torch.randint(0, 3, (9,))
        loss = model.step(X, y, optimizer, ridge=1e-2, inner_iter=2, inner_solver="newton_cg", cg_max_iter=5)
        self.assertIsInstance(loss, float)

    def test_step_requires_readout_in_optimizer(self):
        model = VarProClassifier(_feature_net(), feature_dim=8, num_classes=3)
        optimizer = torch.optim.Adam(model.feature_net.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "must include the readout"):
            model.step(torch.randn(9, 4), torch.randint(0, 3, (9,)), optimizer)


class TestFinalLinearAPI(unittest.TestCase):
    """Test the model.net = nn.Sequential[..., Linear] convention."""

    def setUp(self):
        torch.manual_seed(0)

    def _make_model(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = _small_net()
            def forward(self, X):
                return self.net(X)
        return Net()

    def test_varpro_step_updates_feature_layers(self):
        model = self._make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        X = torch.randn(12, 4)
        Y = torch.randn(12, 2)
        feature_params = list(list(model.net.children())[0].parameters()) + list(list(model.net.children())[1].parameters())
        before = [p.detach().clone() for p in feature_params]
        loss = varpro_step(model, optimizer, X, Y, ridge=1e-3)
        after = [p.detach().clone() for p in feature_params]
        self.assertIsInstance(loss, float)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(before, after)))

    def test_varpro_sparse_step(self):
        model = self._make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        X = torch.randn(12, 4)
        Y = torch.randn(12, 2)
        loss = varpro_sparse_step(model, optimizer, X, Y, sparsity=1e-2, ridge=1e-3, max_iter=10)
        self.assertIsInstance(loss, float)


class TestIncrementalReadout(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_cumulative_mode_matches_full_solve(self):
        F = torch.randn(24, 6)
        Y = torch.randn(24, 3)
        state = IncrementalRidgeReadout(feature_dim=6, output_dim=3, bias=True, ridge=1e-3, mode="cumulative")
        state.update(F[:12], Y[:12])
        state.update(F[12:], Y[12:])
        W_incr = state.solve()
        W_full = solve_ridge_readout(F, Y, ridge=1e-3, bias=True)
        self.assertTrue(torch.allclose(W_incr, W_full, atol=1e-5))

    def test_ema_mode_shape(self):
        state = IncrementalRidgeReadout(feature_dim=6, output_dim=3, bias=True, ridge=1e-3, mode="ema", momentum=0.9)
        state.update(torch.randn(8, 6), torch.randn(8, 3))
        state.update(torch.randn(8, 6), torch.randn(8, 3))
        self.assertEqual(state.solve().shape, (7, 3))

    def test_window_mode_shape(self):
        state = IncrementalRidgeReadout(feature_dim=6, output_dim=3, bias=True, ridge=1e-3, mode="window", window_size=2)
        for _ in range(4):
            state.update(torch.randn(8, 6), torch.randn(8, 3))
        self.assertEqual(state.solve().shape, (7, 3))

    def test_incremental_step_wrapper(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        optimizer = torch.optim.SGD(model.feature_net.parameters(), lr=1e-2)
        state = model.make_incremental_state(ridge=1e-3)
        X = torch.randn(16, 4)
        Y = torch.randn(16, 2)
        loss = varpro_incremental_step(model, optimizer, X, Y, state)
        self.assertIsInstance(loss, float)


class TestProximalReadout(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_high_prox_strength_stays_close(self):
        W_prev = torch.randn(9, 2)
        state = ProximalRidgeReadout(feature_dim=8, output_dim=2, bias=True, prox_strength=1e6, W_init=W_prev)
        W_next = state.update(torch.randn(10, 8), torch.randn(10, 2))
        self.assertTrue(torch.allclose(W_next[:-1], W_prev[:-1], atol=1e-4))

    def test_proximal_step_wrapper(self):
        model = VarProRegressor(_feature_net(), feature_dim=8, output_dim=2)
        optimizer = torch.optim.SGD(model.feature_net.parameters(), lr=1e-2)
        state = model.make_proximal_state(prox_strength=1.0)
        X = torch.randn(16, 4)
        Y = torch.randn(16, 2)
        loss = varpro_proximal_step(model, optimizer, X, Y, state)
        self.assertIsInstance(loss, float)


if __name__ == "__main__":
    unittest.main()
