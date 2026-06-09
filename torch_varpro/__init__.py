"""torch_varpro — Variable Projection training for PyTorch.

Public API
----------
Models:
    VarProRegressor     — MSE readout (ridge or sparse)
    VarProClassifier    — Cross-entropy readout

One-step functions:
    varpro_step         — Classic ridge VarPro (MSE)
    varpro_sparse_step  — Sparse VarPro (l1 / l1/2 penalty)
    varpro_ce_step      — Cross-entropy VarPro
    varpro_incremental_step  — Mini-batch VarPro with accumulated statistics
    varpro_proximal_step     — Mini-batch VarPro with proximal anchoring

Stateful helpers:
    IncrementalRidgeReadout
    ProximalRidgeReadout

Low-level solvers:
    solve_ridge_readout
    solve_sparse_readout
    solve_ce_readout
    augment_features
"""

from .functional import (
    fit_readout_from_loader,
    mse_loss,
    varpro_ce_step,
    varpro_incremental_step,
    varpro_proximal_step,
    varpro_sparse_step,
    varpro_step,
)
from .incremental import IncrementalRidgeReadout, ProximalRidgeReadout
from .models import VarProClassifier, VarProRegressor
from .solvers import (
    augment_features,
    solve_ce_readout,
    solve_ridge_readout,
    solve_sparse_readout,
)

__all__ = [
    # Models
    "VarProRegressor",
    "VarProClassifier",
    # One-step functions
    "varpro_step",
    "varpro_sparse_step",
    "varpro_ce_step",
    "varpro_incremental_step",
    "varpro_proximal_step",
    "fit_readout_from_loader",
    # Stateful helpers
    "IncrementalRidgeReadout",
    "ProximalRidgeReadout",
    # Low-level solvers
    "solve_ridge_readout",
    "solve_sparse_readout",
    "solve_ce_readout",
    "augment_features",
    "mse_loss",
]
