"""PyTorch modules for Variable Projection training."""

import torch
import torch.nn as nn


class VarProRegressor(nn.Module):
    """Feature network with an analytically-optimal linear readout (MSE).

    The readout weights ``W`` and ``b`` are stored as non-differentiable
    buffers; they are updated by the VarPro solve, not by backprop.

    Args:
        feature_net: Any ``nn.Module`` mapping inputs to a feature vector.
        feature_dim: Output dimension of ``feature_net``.
        output_dim: Number of output targets.
        bias: Whether to include a bias in the readout.

    Usage::

        model = VarProRegressor(my_net, feature_dim=128, output_dim=1)
        optimizer = torch.optim.Adam(model.feature_net.parameters(), lr=1e-3)

        for X_batch, Y_batch in loader:
            loss = model.step(X_batch, Y_batch, optimizer)

        # Sparse readout:
        for X_batch, Y_batch in loader:
            loss = model.step(X_batch, Y_batch, optimizer,
                              sparse_penalty="l1", sparsity=1e-3)
    """

    def __init__(self, feature_net: nn.Module, feature_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.feature_net = feature_net
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bias = bias
        self.register_buffer("W", torch.zeros(feature_dim, output_dim))
        self.register_buffer("b", torch.zeros(output_dim) if bias else None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pred = self.feature_net(X) @ self.W
        if self.bias:
            pred = pred + self.b
        return pred

    def step(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn=None,
        ridge: float = 1e-4,
        one_pass: bool = True,
        regularize_bias: bool = False,
        project_after_step: bool = False,
        implicit_readout_gradient: bool = False,
        # sparse options
        sparse_penalty: str | None = None,
        sparsity: float = 1e-4,
        sparse_solver: str = "proxgd",
        sparse_max_iter: int = 100,
        sparse_tol: float = 1e-6,
        # stateful mini-batch options
        incremental_state=None,
        proximal_state=None,
    ) -> float:
        """Perform one VarPro optimization step.

        Passing ``sparse_penalty`` activates sparse VarPro (l1 or l1/2).
        Passing ``incremental_state`` or ``proximal_state`` activates the
        corresponding stateful mini-batch variant.
        """
        from .functional import (
            varpro_incremental_step,
            varpro_proximal_step,
            varpro_sparse_step,
            varpro_step,
        )

        if incremental_state is not None and proximal_state is not None:
            raise ValueError("Pass either incremental_state or proximal_state, not both.")

        if incremental_state is not None:
            return varpro_incremental_step(self, optimizer, X, Y, incremental_state,
                                           loss_fn=loss_fn, project_after_step=project_after_step)
        if proximal_state is not None:
            return varpro_proximal_step(self, optimizer, X, Y, proximal_state,
                                        loss_fn=loss_fn, project_after_step=project_after_step)
        if sparse_penalty is not None:
            return varpro_sparse_step(
                self, optimizer, X, Y,
                sparsity=sparsity, ridge=ridge, penalty=sparse_penalty, loss_fn=loss_fn,
                max_iter=sparse_max_iter, tol=sparse_tol, regularize_bias=regularize_bias,
                one_pass=one_pass, project_after_step=project_after_step,
                implicit_readout_gradient=implicit_readout_gradient, solver=sparse_solver,
            )
        return varpro_step(
            self, optimizer, X, Y,
            ridge=ridge, loss_fn=loss_fn, one_pass=one_pass,
            regularize_bias=regularize_bias, project_after_step=project_after_step,
            implicit_readout_gradient=implicit_readout_gradient,
        )

    def fit_readout(self, data_loader, ridge: float = 1e-4, regularize_bias: bool = False, device=None):
        """Fit the readout by scanning a DataLoader (no gradient step)."""
        from .functional import fit_readout_from_loader
        return fit_readout_from_loader(self, data_loader, ridge=ridge,
                                       regularize_bias=regularize_bias, device=device)

    def make_incremental_state(self, ridge: float = 1e-4, regularize_bias: bool = False,
                                mode: str = "cumulative", momentum: float = 0.95, window_size: int = 8):
        from .incremental import IncrementalRidgeReadout
        return IncrementalRidgeReadout(
            feature_dim=self.feature_dim, output_dim=self.output_dim, bias=self.bias,
            ridge=ridge, regularize_bias=regularize_bias,
            mode=mode, momentum=momentum, window_size=window_size,
            device=self.W.device, dtype=self.W.dtype,
        )

    def make_proximal_state(self, prox_strength: float = 1.0, regularize_bias: bool = False):
        from .incremental import ProximalRidgeReadout
        W_init = torch.cat([self.W, self.b.unsqueeze(0)], dim=0) if self.bias else self.W
        return ProximalRidgeReadout(
            feature_dim=self.feature_dim, output_dim=self.output_dim, bias=self.bias,
            prox_strength=prox_strength, regularize_bias=regularize_bias,
            device=self.W.device, dtype=self.W.dtype, W_init=W_init,
        )


class VarProClassifier(nn.Module):
    """Feature network with an analytically-optimal softmax readout (CE).

    The readout ``W`` and ``b`` are non-differentiable buffers solved by an
    inner convex optimization (L-BFGS or Newton-CG) at each step.

    Args:
        feature_net: Any ``nn.Module`` mapping inputs to a feature vector.
        feature_dim: Output dimension of ``feature_net``.
        num_classes: Number of output classes.

    Usage::

        model = VarProClassifier(my_net, feature_dim=128, num_classes=10)
        optimizer = torch.optim.Adam(model.feature_net.parameters(), lr=1e-3)

        for X_batch, y_batch in loader:
            loss = model.step(X_batch, y_batch, optimizer)
    """

    def __init__(self, feature_net: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_net = feature_net
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.register_buffer("W", torch.zeros(feature_dim, num_classes))
        self.register_buffer("b", torch.zeros(num_classes))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.feature_net(X) @ self.W + self.b

    def step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        ridge: float = 1e-2,
        inner_iter: int = 25,
        regularize_bias: bool = True,
        inner_solver: str = "lbfgs",
        cg_max_iter: int = 50,
        cg_tol: float = 1e-6,
        damping: float = 1e-4,
        newton_tol: float = 1e-6,
    ) -> float:
        """One VarPro cross-entropy step."""
        from .functional import varpro_ce_step
        return varpro_ce_step(
            self, optimizer, X, y,
            ridge=ridge, inner_iter=inner_iter, regularize_bias=regularize_bias,
            inner_solver=inner_solver, cg_max_iter=cg_max_iter, cg_tol=cg_tol,
            damping=damping, newton_tol=newton_tol,
        )
