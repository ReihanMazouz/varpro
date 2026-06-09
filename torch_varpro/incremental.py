"""Stateful readout trackers for mini-batch VarPro."""

from collections import deque

import torch

from .solvers import augment_features


class IncrementalRidgeReadout:
    """Accumulate ridge-readout sufficient statistics across mini-batches.

    Three update modes:

    * ``"cumulative"`` — keep the running sum of all batches seen so far.
      Equivalent to a full-data solve once all data has been seen. Best
      default choice.
    * ``"ema"`` — exponential moving average of per-sample statistics.
      Adapts quickly when the feature extractor changes rapidly.
    * ``"window"`` — fixed-size sliding window of recent batches.

    Usage::

        state = IncrementalRidgeReadout(feature_dim=128, output_dim=10,
                                        bias=True, ridge=1e-4)
        for X_batch, Y_batch in loader:
            features = model.feature_net(X_batch).detach()
            state.update(features, Y_batch)
            W = state.solve()          # (feature_dim+1, output_dim)
            model.W.copy_(W[:-1])
            model.b.copy_(W[-1])
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        bias: bool = True,
        ridge: float = 1e-4,
        regularize_bias: bool = False,
        mode: str = "cumulative",
        momentum: float = 0.95,
        window_size: int = 8,
        device=None,
        dtype=torch.float32,
    ):
        if mode not in {"cumulative", "ema", "window"}:
            raise ValueError("mode must be 'cumulative', 'ema', or 'window'.")

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bias = bias
        self.ridge = ridge
        self.regularize_bias = regularize_bias
        self.mode = mode
        self.momentum = momentum
        self.window_size = window_size
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        self.dim = feature_dim + int(bias)
        self.reset()

    def reset(self):
        self.A = torch.zeros(self.dim, self.dim, device=self.device, dtype=self.dtype)
        self.B = torch.zeros(self.dim, self.output_dim, device=self.device, dtype=self.dtype)
        self.n_samples = 0.0
        self.initialized = False
        self._window: deque = deque(maxlen=self.window_size)

    @torch.no_grad()
    def update(self, features: torch.Tensor, target: torch.Tensor):
        features = features.detach().to(device=self.device, dtype=self.dtype)
        target = target.detach().to(device=self.device, dtype=self.dtype)
        design = augment_features(features, bias=self.bias)
        n = float(design.shape[0])
        if n == 0:
            raise ValueError("Empty batch.")

        A_b = design.T @ design
        B_b = design.T @ target

        if self.mode == "cumulative":
            self.A += A_b
            self.B += B_b
            self.n_samples += n
        elif self.mode == "ema":
            A_m, B_m = A_b / n, B_b / n
            if not self.initialized:
                self.A.copy_(A_m)
                self.B.copy_(B_m)
            else:
                self.A.mul_(self.momentum).add_(A_m, alpha=1.0 - self.momentum)
                self.B.mul_(self.momentum).add_(B_m, alpha=1.0 - self.momentum)
            self.n_samples = 1.0
        else:  # window
            self._window.append((A_b, B_b, n))
            self.A.zero_()
            self.B.zero_()
            self.n_samples = 0.0
            for Ai, Bi, ni in self._window:
                self.A += Ai
                self.B += Bi
                self.n_samples += ni
        self.initialized = True

    @torch.no_grad()
    def solve(self) -> torch.Tensor:
        """Return the optimal readout W of shape ``(dim, output_dim)``."""
        if not self.initialized or self.n_samples <= 0:
            raise ValueError("No accumulated statistics — call update() first.")

        lhs = self.A.clone()
        if self.ridge != 0.0:
            gamma = torch.eye(self.dim, device=self.device, dtype=self.dtype)
            if self.bias and not self.regularize_bias:
                gamma[-1, -1] = 0.0
            lhs = lhs + self.n_samples * self.ridge * gamma

        if self.ridge == 0.0:
            return torch.linalg.pinv(lhs) @ self.B
        try:
            return torch.linalg.solve(lhs, self.B)
        except RuntimeError:
            jitter = torch.finfo(self.dtype).eps * max(self.n_samples, 1.0)
            return torch.linalg.pinv(lhs + jitter * torch.eye(self.dim, device=self.device, dtype=self.dtype)) @ self.B


class ProximalRidgeReadout:
    """Mini-batch proximal ridge readout.

    At each batch, solves:

        1/2 ||Phi W - Y||_F^2  +  (prox_strength/2) ||W - W_prev||_Gamma^2

    and stores the result as the new ``W_prev``. This anchors successive
    mini-batch solves near the previous solution, preventing large jumps.

    Usage::

        state = ProximalRidgeReadout(feature_dim=128, output_dim=10,
                                     bias=True, prox_strength=1.0)
        for X_batch, Y_batch in loader:
            features = model.feature_net(X_batch).detach()
            W = state.update(features, Y_batch)   # (dim, output_dim)
            model.W.copy_(W[:-1])
            model.b.copy_(W[-1])
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        bias: bool = True,
        prox_strength: float = 1.0,
        regularize_bias: bool = False,
        device=None,
        dtype=torch.float32,
        W_init: torch.Tensor | None = None,
    ):
        if prox_strength < 0:
            raise ValueError("prox_strength must be non-negative.")

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bias = bias
        self.prox_strength = prox_strength
        self.regularize_bias = regularize_bias
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        self.dim = feature_dim + int(bias)
        self.W = torch.zeros(self.dim, output_dim, device=self.device, dtype=self.dtype)
        if W_init is not None:
            W_init = W_init.detach().to(device=self.device, dtype=self.dtype)
            if W_init.shape != (self.dim, output_dim):
                raise ValueError(f"W_init shape mismatch: expected {(self.dim, output_dim)}.")
            self.W.copy_(W_init)

    @torch.no_grad()
    def update(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Solve the proximal problem and update the internal state."""
        features = features.detach().to(device=self.device, dtype=self.dtype)
        target = target.detach().to(device=self.device, dtype=self.dtype)
        design = augment_features(features, bias=self.bias)

        lhs = design.T @ design
        rhs = design.T @ target

        if self.prox_strength != 0.0:
            gamma = torch.eye(self.dim, device=self.device, dtype=self.dtype)
            if self.bias and not self.regularize_bias:
                gamma[-1, -1] = 0.0
            lhs = lhs + self.prox_strength * gamma
            rhs = rhs + self.prox_strength * gamma @ self.W

        try:
            W_next = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            jitter = torch.finfo(self.dtype).eps * max(float(design.shape[0]), 1.0)
            W_next = torch.linalg.pinv(lhs + jitter * torch.eye(self.dim, device=self.device, dtype=self.dtype)) @ rhs

        self.W.copy_(W_next)
        return W_next
