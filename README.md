# torch_varpro

**Variable Projection (VarPro) training for PyTorch deep learning models.**

VarPro exploits the structure of models that are *linear in their last layer*: at each gradient step, instead of backpropagating through all parameters simultaneously, it *analytically solves* for the optimal linear readout W★ given the current features, then holds W★ fixed and backpropagates only through the feature network φ.

This turns a joint non-convex problem into a sequence of:
1. **Inner solve** — find W★ in closed form (ridge) or via proximal iterations (sparse).
2. **Outer gradient step** — update the feature network with W★ treated as a constant.

---

## Installation

```bash
pip install torch  # only dependency
```

Then add `torch_varpro/` to your Python path (or install as a local package).

---

## Core concept

For a model `f(X) = φ(X; θ) W + b`, the VarPro step decomposes as:

```
W★ = argmin_W  loss(φ(X; θ) W, Y)      ← solved analytically
θ  ← θ - lr · ∇_θ loss(φ(X; θ) W★, Y) ← standard gradient step
```

The key: **gradients never flow through the readout solve**, so `W` is not a
parameter passed to the optimizer.

---

## Quick start

### Classic VarPro (ridge readout)

```python
import torch
import torch.nn as nn
from torch_varpro import VarProRegressor

feature_net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
model = VarProRegressor(feature_net, feature_dim=128, output_dim=10)
optimizer = torch.optim.Adam(model.feature_net.parameters(), lr=1e-3)

for X_batch, Y_batch in train_loader:   # Y_batch must be float (N, C), not class indices
    loss = model.step(X_batch, Y_batch, optimizer, ridge=1e-4)
```

### Sparse VarPro (l1 penalty)

```python
for X_batch, Y_batch in train_loader:
    loss = model.step(
        X_batch, Y_batch, optimizer,
        sparse_penalty="l1",   # or "l1/2" for non-convex
        sparsity=1e-3,
        ridge=0.0,
    )
```

### Cross-entropy classification

```python
from torch_varpro import VarProClassifier

feature_net = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
model = VarProClassifier(feature_net, feature_dim=256, num_classes=10)
optimizer = torch.optim.Adam(model.feature_net.parameters(), lr=1e-3)

for X_batch, y_batch in train_loader:   # y_batch = integer class labels
    loss = model.step(X_batch, y_batch, optimizer, ridge=1e-2)
```

### Using any `nn.Sequential` model (no wrapper needed)

If your model ends with `nn.Linear`, you can apply VarPro directly:

```python
from torch_varpro import varpro_step, varpro_sparse_step

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    def forward(self, X):
        return self.net(X)

model = MyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Classic VarPro
for X_batch, Y_batch in train_loader:
    loss = varpro_step(model, optimizer, X_batch, Y_batch, ridge=1e-4)

# Sparse VarPro
for X_batch, Y_batch in train_loader:
    loss = varpro_sparse_step(model, optimizer, X_batch, Y_batch,
                              sparsity=1e-3, penalty="l1")
```

---

## Mini-batch variants

When the full dataset doesn't fit in memory, two stateful mini-batch strategies are available.

### Incremental (accumulates normal equations)

```python
state = model.make_incremental_state(ridge=1e-4, mode="cumulative")
# modes: "cumulative" | "ema" | "window"

for X_batch, Y_batch in train_loader:
    loss = model.step(X_batch, Y_batch, optimizer, incremental_state=state)
```

### Proximal (anchors each solve to the previous readout)

```python
state = model.make_proximal_state(prox_strength=1.0)

for X_batch, Y_batch in train_loader:
    loss = model.step(X_batch, Y_batch, optimizer, proximal_state=state)
```

---

## Functional API

All step functions are also available as plain functions, without needing the model wrappers:

```python
from torch_varpro import (
    varpro_step,           # ridge, MSE
    varpro_sparse_step,    # sparse (l1 / l1/2), MSE
    varpro_ce_step,        # cross-entropy
    varpro_incremental_step,
    varpro_proximal_step,
)
```

---

## Low-level solvers

```python
from torch_varpro import solve_ridge_readout, solve_sparse_readout, solve_ce_readout

# Ridge: returns W of shape (feature_dim+1, output_dim) with bias
W = solve_ridge_readout(features, Y, ridge=1e-4, bias=True)

# Sparse l1 (FISTA):
W = solve_sparse_readout(features, Y, sparsity=1e-3, penalty="l1",
                          bias=True, max_iter=100, solver="proxgd")

# Sparse l1/2 (IRLS, non-convex):
W = solve_sparse_readout(features, Y, sparsity=1e-3, penalty="l1/2",
                          bias=True, max_iter=50, solver="irls")

# Cross-entropy (L-BFGS):
W, b = solve_ce_readout(features, y, num_classes=10, ridge=1e-2, solver="lbfgs")
```

---

## API reference

### `VarProRegressor`

| Argument | Default | Description |
|---|---|---|
| `feature_net` | — | Feature extractor module |
| `feature_dim` | — | Feature dimension |
| `output_dim` | — | Number of outputs |
| `bias` | `True` | Include bias in readout |

**`.step(X, Y, optimizer, ...)`** — one VarPro step. Key kwargs:

| Kwarg | Default | Description |
|---|---|---|
| `ridge` | `1e-4` | Ridge regularization |
| `sparse_penalty` | `None` | `"l1"` or `"l1/2"` to activate sparse VarPro |
| `sparsity` | `1e-4` | Sparsity weight λ |
| `sparse_solver` | `"proxgd"` | `"proxgd"` (FISTA) or `"irls"` |
| `one_pass` | `True` | Solve and backprop in one forward pass |
| `incremental_state` | `None` | Pass an `IncrementalRidgeReadout` |
| `proximal_state` | `None` | Pass a `ProximalRidgeReadout` |

### `VarProClassifier`

Same structure, but for classification. `.step(X, y, optimizer, ...)` takes integer labels and uses cross-entropy.

| Kwarg | Default | Description |
|---|---|---|
| `ridge` | `1e-2` | Ridge on W |
| `inner_iter` | `25` | L-BFGS / Newton iterations |
| `inner_solver` | `"lbfgs"` | `"lbfgs"` or `"newton_cg"` |

---

## Running tests

```bash
cd /path/to/varpro
python -m pytest tests/ -v
# or
python tests/test_torch_varpro.py
```

---

## Algorithm summary

```
For each batch (X, Y):

  ┌── Inner solve (no gradient) ──────────────────────────────────────────┐
  │  φ = feature_net(X).detach()                                          │
  │                                                                        │
  │  Classic:  W★ = (φᵀφ + nλI)⁻¹ φᵀY                                   │
  │  Sparse:   W★ = argmin 1/(2n)||φW-Y||² + λ||W||₁   (FISTA or IRLS)   │
  │  CE:       W★ = argmin CE(φW+b, y) + λ||W||²        (L-BFGS)         │
  └───────────────────────────────────────────────────────────────────────┘
  ┌── Outer gradient step ────────────────────────────────────────────────┐
  │  φ_grad = feature_net(X)         ← recompute with gradient           │
  │  loss = loss_fn(φ_grad @ W★, Y)  ← W★ is a constant here            │
  │  loss.backward()                  ← gradient w.r.t. feature_net only │
  │  optimizer.step()                                                     │
  └───────────────────────────────────────────────────────────────────────┘
```
