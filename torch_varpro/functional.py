"""One-step VarPro training functions.

Each function performs a single VarPro optimization step:
  1. Compute features phi(X).
  2. Solve for the optimal linear readout W* on the **detached** features.
  3. Set W* into the model (no gradient on W).
  4. Compute the loss with W* fixed, backprop through phi(X), step the optimizer.

Two model conventions are supported:

* **VarProRegressor / VarProClassifier** — explicit ``feature_net`` + separate
  ``W`` / ``b`` buffers.
* **nn.Sequential-ended** — any ``model`` whose ``model.net`` is an
  ``nn.Sequential`` ending in ``nn.Linear``; VarPro intercepts that final layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .solvers import augment_features, solve_ce_readout, solve_ridge_readout, solve_sparse_readout


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * mean_i ||pred_i - target_i||^2."""
    target = target.to(device=pred.device, dtype=pred.dtype)
    return 0.5 * (pred - target).square().reshape(pred.shape[0], -1).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Classic VarPro (ridge readout)
# ---------------------------------------------------------------------------

def varpro_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    ridge: float = 1e-4,
    loss_fn=None,
    one_pass: bool = True,
    regularize_bias: bool = False,
    project_after_step: bool = False,
    implicit_readout_gradient: bool = False,
) -> float:
    """One VarPro step for MSE / regression.

    Args:
        model: Either a ``VarProRegressor`` or an ``nn.Module`` with
            ``model.net`` as an ``nn.Sequential`` ending in ``nn.Linear``.
        optimizer: Optimizer over the feature-network parameters.
        X: Input batch, shape ``(N, ...)``.
        Y: Target batch, shape ``(N, C)`` (must be float, not class indices).
        ridge: L2 regularization on the readout.
        loss_fn: Optional custom loss; defaults to ``mse_loss``.
        one_pass: If True (default), solve W* then backprop in a single
            forward pass.  If False, solve first, then do a fresh forward pass
            (slightly higher cost, sometimes more stable).
        regularize_bias: Whether to penalize the bias in the ridge solve.
        project_after_step: Re-solve for W* after the gradient step.
        implicit_readout_gradient: Keep gradient path through the ridge solve
            (implicit differentiation). Disabled by default.

    Returns:
        Scalar training loss.
    """
    if Y.ndim == 1:
        raise ValueError("Y must be 2-D (N, C). For classification use varpro_ce_step.")

    if _is_wrapper(model):
        return _wrapper_mse_step(
            model, optimizer, X, Y,
            ridge=ridge, loss_fn=loss_fn, one_pass=one_pass,
            regularize_bias=regularize_bias, project_after_step=project_after_step,
            implicit_readout_gradient=implicit_readout_gradient,
        )
    return _final_linear_mse_step(
        model, optimizer, X, Y,
        ridge=ridge, loss_fn=loss_fn, one_pass=one_pass,
        regularize_bias=regularize_bias, project_after_step=project_after_step,
        implicit_readout_gradient=implicit_readout_gradient,
    )


# ---------------------------------------------------------------------------
# Sparse VarPro
# ---------------------------------------------------------------------------

def varpro_sparse_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    sparsity: float = 1e-4,
    ridge: float = 0.0,
    penalty: str = "l1",
    loss_fn=None,
    max_iter: int = 100,
    tol: float = 1e-6,
    regularize_bias: bool = False,
    one_pass: bool = True,
    project_after_step: bool = False,
    implicit_readout_gradient: bool = False,
    solver: str = "proxgd",
) -> float:
    """One VarPro step with a sparse proximal readout solve.

    The readout is solved with an l1 or l1/2 penalty (FISTA or IRLS),
    then held fixed during the backward pass.

    Args:
        sparsity: Sparsity regularization weight (lambda).
        ridge: Optional additional L2 term on the readout.
        penalty: ``"l1"`` (convex, FISTA default) or ``"l1/2"`` / ``"l0.5"``
            (non-convex, use ``solver="irls"``).
        solver: ``"proxgd"`` (FISTA) or ``"irls"``.

    Returns:
        Scalar training loss.
    """
    if Y.ndim == 1:
        raise ValueError("Y must be 2-D (N, C).")

    if _is_wrapper(model):
        return _wrapper_sparse_step(
            model, optimizer, X, Y,
            ridge=ridge, sparsity=sparsity, penalty=penalty, loss_fn=loss_fn,
            max_iter=max_iter, tol=tol, regularize_bias=regularize_bias,
            one_pass=one_pass, project_after_step=project_after_step,
            implicit_readout_gradient=implicit_readout_gradient, solver=solver,
        )
    return _final_linear_sparse_step(
        model, optimizer, X, Y,
        ridge=ridge, sparsity=sparsity, penalty=penalty, loss_fn=loss_fn,
        max_iter=max_iter, tol=tol, regularize_bias=regularize_bias,
        one_pass=one_pass, project_after_step=project_after_step,
        implicit_readout_gradient=implicit_readout_gradient, solver=solver,
    )


# ---------------------------------------------------------------------------
# CE VarPro (classification)
# ---------------------------------------------------------------------------

def varpro_ce_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    ridge: float = 1e-2,
    inner_iter: int = 25,
    regularize_bias: bool = True,
    inner_solver: str = "lbfgs",
    cg_max_iter: int = 50,
    cg_tol: float = 1e-6,
    damping: float = 1e-4,
    newton_tol: float = 1e-6,
) -> float:
    """One VarPro step for cross-entropy classification.

    Requires a ``VarProClassifier``-like model with ``feature_net``, ``W``,
    ``b``, and ``num_classes``.

    Args:
        inner_solver: ``"lbfgs"`` or ``"newton_cg"``.

    Returns:
        Scalar cross-entropy loss.
    """
    if not _is_wrapper(model):
        raise ValueError("varpro_ce_step requires a VarProClassifier with feature_net, W, b.")

    optimizer.zero_grad(set_to_none=True)
    features = model.feature_net(X)
    W_star, b_star = solve_ce_readout(
        features.detach(), y,
        num_classes=model.num_classes, ridge=ridge, max_iter=inner_iter,
        W_init=model.W, b_init=model.b,
        regularize_bias=regularize_bias, solver=inner_solver,
        cg_max_iter=cg_max_iter, cg_tol=cg_tol,
        damping=damping, newton_tol=newton_tol,
    )
    with torch.no_grad():
        model.W.copy_(W_star)
        model.b.copy_(b_star)

    logits = features @ W_star.detach() + b_star.detach()
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Incremental / proximal stateful steps
# ---------------------------------------------------------------------------

def varpro_incremental_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    readout_state,
    loss_fn=None,
    project_after_step: bool = False,
) -> float:
    """VarPro step using an ``IncrementalRidgeReadout`` state.

    Accumulates sufficient statistics across batches; the readout W is the
    solution of the accumulated normal equations.
    """
    if Y.ndim == 1:
        raise ValueError("Y must be 2-D (N, C).")
    if _is_wrapper(model):
        return _wrapper_incremental_step(model, optimizer, X, Y, readout_state, loss_fn, project_after_step)
    return _final_linear_incremental_step(model, optimizer, X, Y, readout_state, loss_fn, project_after_step)


def varpro_proximal_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    readout_state,
    loss_fn=None,
    project_after_step: bool = False,
) -> float:
    """VarPro step using a ``ProximalRidgeReadout`` state.

    Solves a per-batch proximal problem anchored at the previous readout.
    """
    if Y.ndim == 1:
        raise ValueError("Y must be 2-D (N, C).")
    if _is_wrapper(model):
        return _wrapper_proximal_step(model, optimizer, X, Y, readout_state, loss_fn, project_after_step)
    return _final_linear_proximal_step(model, optimizer, X, Y, readout_state, loss_fn, project_after_step)


# ---------------------------------------------------------------------------
# Batch readout fitting (no gradient step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def fit_readout_from_loader(model, data_loader, ridge=1e-4, regularize_bias=False, device=None):
    """Fit the ridge readout by accumulating normal equations over a DataLoader.

    Useful for the initial readout fit before training.
    Expects batches of ``(X, Y)`` tuples or dicts with ``"X"``/``"Y"`` keys.
    """
    device = _resolve_device(model, device)
    was_training = model.training
    model.eval()
    dim = model.feature_dim + int(model.bias)
    lhs = torch.zeros(dim, dim, device=device, dtype=model.W.dtype)
    rhs = torch.zeros(dim, model.output_dim, device=device, dtype=model.W.dtype)
    n = 0

    try:
        for batch in data_loader:
            X, Y = _unpack(batch)
            X = X.to(device)
            Y = Y.to(device=device, dtype=model.W.dtype)
            design = augment_features(model.feature_net(X), bias=model.bias)
            lhs += design.T @ design
            rhs += design.T @ Y
            n += design.shape[0]
    finally:
        model.train(was_training)

    if n == 0:
        raise ValueError("Empty data_loader.")

    if ridge != 0.0:
        gamma = torch.eye(dim, device=device, dtype=model.W.dtype)
        if model.bias and not regularize_bias:
            gamma[-1, -1] = 0.0
        lhs = lhs + n * ridge * gamma

    W_full = torch.linalg.pinv(lhs) @ rhs if ridge == 0.0 else torch.linalg.solve(lhs, rhs)
    _set_wrapper_readout(model, W_full)
    return W_full


# ---------------------------------------------------------------------------
# Internal helpers — wrapper models (VarProRegressor/Classifier)
# ---------------------------------------------------------------------------

def _wrapper_mse_step(model, optimizer, X, Y, ridge, loss_fn, one_pass, regularize_bias, project_after_step, implicit_readout_gradient):
    if one_pass:
        optimizer.zero_grad(set_to_none=True)
        features = model.feature_net(X)
        W_full = solve_ridge_readout(
            features if implicit_readout_gradient else features.detach(),
            Y, ridge=ridge, bias=model.bias, regularize_bias=regularize_bias,
            detach_features=not implicit_readout_gradient,
        )
        _set_wrapper_readout(model, W_full)
        design = augment_features(features, bias=model.bias)
        pred = design @ (W_full if implicit_readout_gradient else W_full.detach())
        loss = (loss_fn or mse_loss)(pred, Y)
        loss.backward()
        optimizer.step()
        if project_after_step:
            _project_wrapper(model, X, Y, ridge, regularize_bias)
        return loss.item()

    _project_wrapper(model, X, Y, ridge, regularize_bias)
    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    optimizer.step()
    if project_after_step:
        _project_wrapper(model, X, Y, ridge, regularize_bias)
    return loss.item()


def _wrapper_sparse_step(model, optimizer, X, Y, ridge, sparsity, penalty, loss_fn, max_iter, tol, regularize_bias, one_pass, project_after_step, implicit_readout_gradient, solver):
    if one_pass:
        optimizer.zero_grad(set_to_none=True)
        features = model.feature_net(X)
        W_full = solve_sparse_readout(
            features if implicit_readout_gradient else features.detach(),
            Y, ridge=ridge, sparsity=sparsity, penalty=penalty,
            bias=model.bias, regularize_bias=regularize_bias,
            max_iter=max_iter, tol=tol,
            W_init=_get_wrapper_readout(model),
            detach_features=not implicit_readout_gradient,
            detach_result=not implicit_readout_gradient,
            solver=solver,
        )
        _set_wrapper_readout(model, W_full)
        design = augment_features(features, bias=model.bias)
        pred = design @ (W_full if implicit_readout_gradient else W_full.detach())
        loss = (loss_fn or mse_loss)(pred, Y)
        loss.backward()
        optimizer.step()
        if project_after_step:
            _project_wrapper_sparse(model, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias)
        return loss.item()

    _project_wrapper_sparse(model, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias)
    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    optimizer.step()
    if project_after_step:
        _project_wrapper_sparse(model, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias)
    return loss.item()


def _wrapper_incremental_step(model, optimizer, X, Y, state, loss_fn, project_after_step):
    optimizer.zero_grad(set_to_none=True)
    features = model.feature_net(X)
    state.update(features.detach(), Y)
    W_full = state.solve()
    _set_wrapper_readout(model, W_full)
    pred = augment_features(features, bias=model.bias) @ W_full.detach()
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    optimizer.step()
    if project_after_step:
        with torch.no_grad():
            state.update(model.feature_net(X).detach(), Y)
            _set_wrapper_readout(model, state.solve())
    return loss.item()


def _wrapper_proximal_step(model, optimizer, X, Y, state, loss_fn, project_after_step):
    optimizer.zero_grad(set_to_none=True)
    features = model.feature_net(X)
    W_full = state.update(features.detach(), Y)
    _set_wrapper_readout(model, W_full)
    pred = augment_features(features, bias=model.bias) @ W_full.detach()
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    optimizer.step()
    if project_after_step:
        with torch.no_grad():
            W_full = state.update(model.feature_net(X).detach(), Y)
            _set_wrapper_readout(model, W_full)
    return loss.item()


# ---------------------------------------------------------------------------
# Internal helpers — final-linear models (model.net = nn.Sequential[..., Linear])
# ---------------------------------------------------------------------------

def _final_linear_mse_step(model, optimizer, X, Y, ridge, loss_fn, one_pass, regularize_bias, project_after_step, implicit_readout_gradient):
    phi, readout = _split_final_linear(model)
    has_bias = readout.bias is not None

    if one_pass:
        optimizer.zero_grad(set_to_none=True)
        features = phi(X)
        W_full = solve_ridge_readout(
            features if implicit_readout_gradient else features.detach(),
            Y, ridge=ridge, bias=has_bias, regularize_bias=regularize_bias,
            detach_features=not implicit_readout_gradient,
        )
        _set_linear_readout(readout, W_full)
        pred = augment_features(features, bias=has_bias) @ (W_full if implicit_readout_gradient else W_full.detach())
        loss = (loss_fn or mse_loss)(pred, Y)
        loss.backward()
        _clear_linear_grad(readout)
        optimizer.step()
        if project_after_step:
            _project_final_linear(phi, readout, X, Y, ridge, regularize_bias)
        return loss.item()

    _project_final_linear(phi, readout, X, Y, ridge, regularize_bias)
    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    _clear_linear_grad(readout)
    optimizer.step()
    if project_after_step:
        _project_final_linear(phi, readout, X, Y, ridge, regularize_bias)
    return loss.item()


def _final_linear_sparse_step(model, optimizer, X, Y, ridge, sparsity, penalty, loss_fn, max_iter, tol, regularize_bias, one_pass, project_after_step, implicit_readout_gradient, solver):
    phi, readout = _split_final_linear(model)
    has_bias = readout.bias is not None

    if one_pass:
        optimizer.zero_grad(set_to_none=True)
        features = phi(X)
        W_full = solve_sparse_readout(
            features if implicit_readout_gradient else features.detach(),
            Y, ridge=ridge, sparsity=sparsity, penalty=penalty,
            bias=has_bias, regularize_bias=regularize_bias,
            max_iter=max_iter, tol=tol,
            W_init=_get_linear_readout(readout),
            detach_features=not implicit_readout_gradient,
            detach_result=not implicit_readout_gradient,
            solver=solver,
        )
        _set_linear_readout(readout, W_full)
        pred = augment_features(features, bias=has_bias) @ (W_full if implicit_readout_gradient else W_full.detach())
        loss = (loss_fn or mse_loss)(pred, Y)
        loss.backward()
        _clear_linear_grad(readout)
        optimizer.step()
        if project_after_step:
            _project_final_linear_sparse(phi, readout, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias)
        return loss.item()

    _project_final_linear_sparse(phi, readout, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias)
    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    _clear_linear_grad(readout)
    optimizer.step()
    if project_after_step:
        _project_final_linear_sparse(phi, readout, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias)
    return loss.item()


def _final_linear_incremental_step(model, optimizer, X, Y, state, loss_fn, project_after_step):
    phi, readout = _split_final_linear(model)
    has_bias = readout.bias is not None
    optimizer.zero_grad(set_to_none=True)
    features = phi(X)
    state.update(features.detach(), Y)
    W_full = state.solve()
    _set_linear_readout(readout, W_full)
    pred = augment_features(features, bias=has_bias) @ W_full.detach()
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    _clear_linear_grad(readout)
    optimizer.step()
    if project_after_step:
        with torch.no_grad():
            state.update(phi(X).detach(), Y)
            _set_linear_readout(readout, state.solve())
    return loss.item()


def _final_linear_proximal_step(model, optimizer, X, Y, state, loss_fn, project_after_step):
    phi, readout = _split_final_linear(model)
    has_bias = readout.bias is not None
    optimizer.zero_grad(set_to_none=True)
    features = phi(X)
    W_full = state.update(features.detach(), Y)
    _set_linear_readout(readout, W_full)
    pred = augment_features(features, bias=has_bias) @ W_full.detach()
    loss = (loss_fn or mse_loss)(pred, Y)
    loss.backward()
    _clear_linear_grad(readout)
    optimizer.step()
    if project_after_step:
        with torch.no_grad():
            W_full = state.update(phi(X).detach(), Y)
            _set_linear_readout(readout, W_full)
    return loss.item()


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _project_wrapper(model, X, Y, ridge, regularize_bias):
    W_full = solve_ridge_readout(model.feature_net(X), Y, ridge=ridge, bias=model.bias, regularize_bias=regularize_bias)
    _set_wrapper_readout(model, W_full)
    return W_full


@torch.no_grad()
def _project_wrapper_sparse(model, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias):
    W_full = solve_sparse_readout(
        model.feature_net(X), Y, ridge=ridge, sparsity=sparsity, penalty=penalty,
        bias=model.bias, regularize_bias=regularize_bias, max_iter=max_iter, tol=tol,
        W_init=_get_wrapper_readout(model),
    )
    _set_wrapper_readout(model, W_full)
    return W_full


@torch.no_grad()
def _project_final_linear(phi, readout, X, Y, ridge, regularize_bias):
    W_full = solve_ridge_readout(phi(X), Y, ridge=ridge, bias=readout.bias is not None, regularize_bias=regularize_bias)
    _set_linear_readout(readout, W_full)
    return W_full


@torch.no_grad()
def _project_final_linear_sparse(phi, readout, X, Y, ridge, sparsity, penalty, max_iter, tol, regularize_bias):
    W_full = solve_sparse_readout(
        phi(X), Y, ridge=ridge, sparsity=sparsity, penalty=penalty,
        bias=readout.bias is not None, regularize_bias=regularize_bias,
        max_iter=max_iter, tol=tol, W_init=_get_linear_readout(readout),
    )
    _set_linear_readout(readout, W_full)
    return W_full


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------

def _is_wrapper(model):
    return hasattr(model, "feature_net") and hasattr(model, "W")


def _split_final_linear(model):
    if not hasattr(model, "net") or not isinstance(model.net, nn.Sequential):
        raise ValueError("Expected model.net to be nn.Sequential.")
    layers = list(model.net.children())
    if not layers or not isinstance(layers[-1], nn.Linear):
        raise ValueError("The last layer of model.net must be nn.Linear.")
    return nn.Sequential(*layers[:-1]), layers[-1]


@torch.no_grad()
def _get_wrapper_readout(model):
    if model.bias:
        return torch.cat([model.W, model.b.unsqueeze(0)], dim=0)
    return model.W.detach().clone()


@torch.no_grad()
def _set_wrapper_readout(model, W_full):
    if model.bias:
        expected = (model.feature_dim + 1, model.output_dim)
        if W_full.shape != expected:
            raise ValueError(f"Expected {expected}, got {tuple(W_full.shape)}.")
        model.W.copy_(W_full[:-1])
        model.b.copy_(W_full[-1])
    else:
        expected = (model.feature_dim, model.output_dim)
        if W_full.shape != expected:
            raise ValueError(f"Expected {expected}, got {tuple(W_full.shape)}.")
        model.W.copy_(W_full)


@torch.no_grad()
def _get_linear_readout(readout):
    W = readout.weight.detach().T.clone()
    if readout.bias is None:
        return W
    return torch.cat([W, readout.bias.detach().unsqueeze(0)], dim=0)


@torch.no_grad()
def _set_linear_readout(readout, W_full):
    fin, fout = readout.in_features, readout.out_features
    if readout.bias is None:
        if W_full.shape != (fin, fout):
            raise ValueError(f"Expected {(fin, fout)}, got {tuple(W_full.shape)}.")
        readout.weight.copy_(W_full.T)
    else:
        if W_full.shape != (fin + 1, fout):
            raise ValueError(f"Expected {(fin + 1, fout)}, got {tuple(W_full.shape)}.")
        readout.weight.copy_(W_full[:-1].T)
        readout.bias.copy_(W_full[-1])


def _clear_linear_grad(readout):
    readout.weight.grad = None
    if readout.bias is not None:
        readout.bias.grad = None


def _resolve_device(model, device):
    if device is not None:
        return torch.device(device)
    return model.W.device


def _unpack(batch):
    if isinstance(batch, dict):
        if "X" in batch and "Y" in batch:
            return batch["X"], batch["Y"]
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        raise ValueError("Dict batches must have ('X','Y') or ('x','y') keys.")
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Batches must yield at least (X, target).")
