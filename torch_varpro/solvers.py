"""Readout solvers for Variable Projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def augment_features(features: torch.Tensor, bias: bool) -> torch.Tensor:
    """Append a constant-one column to features when a bias is used."""
    if not bias:
        return features
    ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
    return torch.cat([features, ones], dim=1)


# ---------------------------------------------------------------------------
# Ridge readout
# ---------------------------------------------------------------------------

def solve_ridge_readout(
    features: torch.Tensor,
    Y: torch.Tensor,
    ridge: float = 1e-4,
    bias: bool = True,
    regularize_bias: bool = False,
    detach_features: bool = True,
) -> torch.Tensor:
    """Solve the normalized ridge least-squares problem.

    Objective:
        1/(2N) ||Phi W - Y||_F^2  +  (ridge/2) tr(W^T Gamma W)

    Returns W of shape ``(feature_dim [+1], output_dim)``; the last row is the
    bias vector when ``bias=True``.

    Set ``detach_features=False`` to keep the gradient path through the solve
    (implicit differentiation / unrolled solve).
    """
    if Y.ndim == 1:
        raise ValueError("Y must be 2-D (N, C).")
    if detach_features:
        features = features.detach()
    Y = Y.to(device=features.device, dtype=features.dtype)
    design = augment_features(features, bias=bias)
    n, dim = design.shape

    if ridge == 0.0:
        return torch.linalg.pinv(design) @ Y

    gamma = torch.eye(dim, device=features.device, dtype=features.dtype)
    if bias and not regularize_bias:
        gamma[-1, -1] = 0.0

    lhs = design.T @ design + n * ridge * gamma
    rhs = design.T @ Y
    try:
        return torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        jitter = torch.finfo(features.dtype).eps * max(float(n), 1.0)
        return torch.linalg.pinv(lhs + jitter * torch.eye(dim, device=features.device, dtype=features.dtype)) @ rhs


# ---------------------------------------------------------------------------
# Sparse readout
# ---------------------------------------------------------------------------

def solve_sparse_readout(
    features: torch.Tensor,
    Y: torch.Tensor,
    ridge: float = 0.0,
    sparsity: float = 1e-4,
    penalty: str = "l1",
    bias: bool = True,
    regularize_bias: bool = False,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-8,
    W_init: torch.Tensor | None = None,
    accelerated: bool = True,
    detach_features: bool = True,
    detach_result: bool = True,
    solver: str = "proxgd",
    irls_eps0: float = 1.0,
    irls_eps_min: float = 1e-3,
    irls_d_max: float = 1e3,
    irls_ridge: float = 1e-3,
) -> torch.Tensor:
    """Solve a sparse squared-loss readout.

    Objective:
        1/(2N) ||Phi W - Y||_F^2
        + (ridge/2) tr(W^T Gamma W)
        + sparsity * Omega(W)

    where ``Omega`` is ``||W||_1`` (l1) or ``sum sqrt(|W| + eps)`` (l1/2).
    The bias row is never penalized for sparsity.

    Args:
        solver: ``"proxgd"`` (FISTA) for convex l1; ``"irls"`` for non-convex
            l1/2 or when you want unrolled differentiable solves.
        detach_features: detach features before the solve (standard VarPro).
        detach_result: return a detached W (no gradient through the solve).
    """
    if penalty not in {"l1", "l1/2", "lhalf", "l0.5"}:
        raise ValueError("penalty must be 'l1', 'l1/2', 'lhalf', or 'l0.5'.")
    if solver not in {"proxgd", "irls"}:
        raise ValueError("solver must be 'proxgd' or 'irls'.")
    if Y.ndim == 1:
        raise ValueError("Y must be 2-D (N, C).")

    if detach_features:
        features = features.detach()
    Y = Y.to(device=features.device, dtype=features.dtype)
    design = augment_features(features, bias=bias)
    n, dim = design.shape

    if solver == "irls":
        return _solve_sparse_irls(
            design, Y,
            ridge=ridge, sparsity=sparsity, penalty=penalty,
            bias=bias, regularize_bias=regularize_bias,
            max_iter=max_iter, eps0=irls_eps0, eps_min=irls_eps_min,
            d_max=irls_d_max, relative_ridge=irls_ridge,
            detach_result=detach_result,
        )

    # --- FISTA (proximal gradient) ---
    if W_init is None:
        W = torch.zeros(dim, Y.shape[1], device=features.device, dtype=features.dtype)
    else:
        W = W_init.detach().clone().to(device=features.device, dtype=features.dtype)
        if W.shape != (dim, Y.shape[1]):
            raise ValueError(f"W_init shape mismatch: expected {(dim, Y.shape[1])}, got {tuple(W.shape)}.")

    gamma = torch.ones(dim, 1, device=features.device, dtype=features.dtype)
    if bias and not regularize_bias:
        gamma[-1] = 0.0
    sparse_mask = torch.ones_like(gamma)
    if bias:
        sparse_mask[-1] = 0.0

    spectral = torch.linalg.matrix_norm(design, ord=2)
    L = (spectral * spectral) / n + ridge
    step = 1.0 / L.clamp_min(torch.finfo(features.dtype).eps)

    Z = W.clone()
    t = torch.ones((), device=features.device, dtype=features.dtype)

    for _ in range(max_iter):
        W_prev = W
        grad = design.T @ (design @ Z - Y) / n
        if ridge != 0.0:
            grad = grad + ridge * gamma * Z

        if penalty == "l1":
            threshold = step * sparsity * sparse_mask
        else:
            weights = 0.5 / torch.sqrt(Z.detach().abs() + eps)
            threshold = step * sparsity * sparse_mask * weights

        W = _soft_threshold(Z - step * grad, threshold)

        if not accelerated:
            Z = W
        else:
            t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
            Z = W + ((t - 1.0) / t_next) * (W - W_prev)
            t = t_next

        if (W - W_prev).norm() / W_prev.norm().clamp_min(1.0) <= tol:
            break

    return W.detach() if detach_result else W


def _solve_sparse_irls(
    design, Y, ridge, sparsity, penalty, bias, regularize_bias,
    max_iter, eps0, eps_min, d_max, relative_ridge, detach_result,
):
    """Iteratively Reweighted Least Squares for sparse readout."""
    n, dim = design.shape
    out_dim = Y.shape[1]
    device, dtype = design.device, design.dtype

    gamma_diag = torch.ones(dim, device=device, dtype=dtype)
    if bias and not regularize_bias:
        gamma_diag[-1] = 0.0
    sparse_mask = torch.ones(dim, device=device, dtype=dtype)
    if bias:
        sparse_mask[-1] = 0.0

    G = design.T @ design / n
    B = design.T @ Y / n
    I = torch.eye(dim, device=device, dtype=dtype)
    scale_ridge = relative_ridge * G.diagonal().mean().clamp_min(torch.finfo(dtype).eps)
    lhs0 = G + torch.diag(ridge * gamma_diag) + scale_ridge * I
    try:
        W = torch.linalg.solve(lhs0, B)
    except RuntimeError:
        W = torch.linalg.pinv(lhs0) @ B
    if not torch.isfinite(W).all():
        W = torch.zeros(dim, out_dim, device=device, dtype=dtype)

    eps_schedule = torch.logspace(
        torch.log10(torch.as_tensor(eps0, device=device, dtype=dtype)),
        torch.log10(torch.as_tensor(eps_min, device=device, dtype=dtype)),
        max_iter, device=device, dtype=dtype,
    )
    for eps in eps_schedule:
        weights = (1.0 / (W.abs() + eps).clamp_min(eps) if penalty == "l1"
                   else 0.5 * (W.square() + eps).pow(-0.75))
        weights = weights.clamp(max=d_max) * sparse_mask[:, None]
        lhs = G[:, :, None] + torch.diag(ridge * gamma_diag + scale_ridge).unsqueeze(-1)
        lhs = lhs + sparsity * torch.diag_embed(weights.T).permute(1, 2, 0)
        cols = []
        for j in range(out_dim):
            try:
                col = torch.linalg.solve(lhs[:, :, j], B[:, j])
            except RuntimeError:
                col = torch.linalg.pinv(lhs[:, :, j]) @ B[:, j]
            cols.append(col)
        W_next = torch.stack(cols, dim=1)
        if not torch.isfinite(W_next).all():
            break
        W = W_next

    return W.detach() if detach_result else W


# ---------------------------------------------------------------------------
# Cross-entropy readout (classification)
# ---------------------------------------------------------------------------

def solve_ce_readout(
    features: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ridge: float = 1e-2,
    max_iter: int = 25,
    W_init: torch.Tensor | None = None,
    b_init: torch.Tensor | None = None,
    regularize_bias: bool = True,
    solver: str = "lbfgs",
    cg_max_iter: int = 50,
    cg_tol: float = 1e-6,
    damping: float = 1e-4,
    newton_tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the convex softmax readout for fixed features.

    Returns ``(W, b)`` — detached tensors.

    Args:
        solver: ``"lbfgs"`` (default) or ``"newton_cg"`` (Hessian-free Newton).
    """
    if solver == "newton_cg":
        return _solve_ce_newton_cg(
            features, y, num_classes, ridge, max_iter,
            W_init, b_init, regularize_bias, cg_max_iter, cg_tol, damping, newton_tol,
        )
    if solver != "lbfgs":
        raise ValueError("solver must be 'lbfgs' or 'newton_cg'.")

    features = features.detach()
    device, dtype = features.device, features.dtype
    y = y.to(device=device)

    W = (torch.zeros(features.shape[1], num_classes, device=device, dtype=dtype)
         if W_init is None else W_init.detach().clone().to(device=device, dtype=dtype))
    b = (torch.zeros(num_classes, device=device, dtype=dtype)
         if b_init is None else b_init.detach().clone().to(device=device, dtype=dtype))
    W.requires_grad_(True)
    b.requires_grad_(True)

    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(features @ W + b, y) + 0.5 * ridge * W.square().sum()
        if regularize_bias:
            loss = loss + 0.5 * ridge * b.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _solve_ce_newton_cg(
    features, y, num_classes, ridge, max_iter,
    W_init, b_init, regularize_bias, cg_max_iter, cg_tol, damping, newton_tol,
):
    features = features.detach()
    device, dtype = features.device, features.dtype
    y = y.to(device=device)
    n_feat = features.shape[1]

    W0 = (torch.zeros(n_feat, num_classes, device=device, dtype=dtype)
          if W_init is None else W_init.detach().clone().to(device=device, dtype=dtype))
    b0 = (torch.zeros(num_classes, device=device, dtype=dtype)
          if b_init is None else b_init.detach().clone().to(device=device, dtype=dtype))

    z = torch.cat([W0.reshape(-1), b0.reshape(-1)]).detach()
    n_w = n_feat * num_classes

    def unpack(v):
        return v[:n_w].view(n_feat, num_classes), v[n_w:]

    def objective(v):
        W_, b_ = unpack(v)
        loss = F.cross_entropy(features @ W_ + b_, y) + 0.5 * ridge * W_.square().sum()
        if regularize_bias:
            loss = loss + 0.5 * ridge * b_.square().sum()
        return loss

    for _ in range(max_iter):
        z_req = z.detach().requires_grad_(True)
        grad = torch.autograd.grad(objective(z_req), z_req, create_graph=True)[0]
        grad_d = grad.detach()
        if grad_d.norm() <= newton_tol:
            break

        def hvp(v):
            hv = torch.autograd.grad((grad * v).sum(), z_req, retain_graph=True)[0]
            return (hv + damping * v).detach()

        direction = _cg(hvp, -grad_d, cg_max_iter, cg_tol)
        with torch.no_grad():
            current = objective(z).detach()
            step = 1.0
            gd = (grad_d * direction).sum()
            if gd >= 0:
                direction = -grad_d
                gd = -(grad_d * grad_d).sum()
            for _ in range(20):
                cand = z + step * direction
                if torch.isfinite(objective(cand)) and objective(cand) <= current + 1e-4 * step * gd:
                    z = cand.detach()
                    break
                step *= 0.5
            else:
                z = (z + 1e-3 * direction).detach()

    W, b = unpack(z.detach())
    return W.detach(), b.detach()


def _cg(matvec, rhs, max_iter, tol):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum()
    if rs.sqrt() <= tol:
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = (p * Ap).sum()
        if denom.abs() <= torch.finfo(rhs.dtype).eps:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() <= tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _soft_threshold(x, threshold):
    return torch.sign(x) * (x.abs() - threshold).clamp(min=0.0)
