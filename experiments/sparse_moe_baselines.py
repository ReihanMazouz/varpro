"""Run the sparse-MoE baseline comparison.

* MNIST: 3,000 training and 1,000 test examples;
* dense soft-gated MoE with h=32 and k=4;
* full-batch Adam, learning rate 1e-3, 100 epochs;
* five seeds (0, ..., 4);
* sparse readout: lambda=1e-3, 100 proximal iterations;
* sparsity: fraction of readout weights with |w| < 1e-3.

Install the experiment dependencies and run:

    pip install numpy torchvision
    python experiments/sparse_moe_baselines.py

For a fast code-path check:

    python experiments/sparse_moe_baselines.py \
        --methods varpro_sparse_l1 joint_mse_prox_l1 \
        --epochs 2 --seeds 1 --output-dir /tmp/sparse_moe_smoke
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from torch_varpro.solvers import (  # noqa: E402
    solve_ce_readout,
    solve_sparse_readout,
)


N_TRAIN = 3_000
N_TEST = 1_000
N_CLASSES = 10
HIDDEN_DIM = 32
N_EXPERTS = 4
ADAM_LR = 1e-3
CE_RIDGE = 1e-2
CE_MAX_ITER = 25
SPARSE_RIDGE = 1e-4
SPARSE_LAMBDA = 1e-3
SPARSE_MAX_ITER = 100
SPARSITY_THRESHOLD = 1e-3
LHALF_EPS = 1e-8
PRUNING_TARGET = 0.875
PRUNING_EPOCH_FRACTION = 0.8


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    group: str


NO_SPARSITY = "No sparsity penalty"
L1_GROUP = "L1 sparsity and matched sparse baselines"
LHALF_GROUP = "L1/2 sparsity"

METHODS = [
    Method("varpro_mse", "VarPro dense MSE", NO_SPARSITY),
    Method("varpro_ce", "VarPro dense CE", NO_SPARSITY),
    Method("joint_mse", "Baseline MSE", NO_SPARSITY),
    Method("joint_ce", "Baseline CE", NO_SPARSITY),
    Method("varpro_sparse_l1", "VarPro sparse MSE + L1", L1_GROUP),
    Method("varpro_ce_sparse_l1", "VarPro sparse CE + L1", L1_GROUP),
    Method("joint_mse_l1", "Baseline MSE + L1", L1_GROUP),
    Method("joint_ce_l1", "Baseline CE + L1", L1_GROUP),
    Method("joint_mse_prox_l1", "Baseline MSE + prox L1", L1_GROUP),
    Method("joint_ce_prox_l1", "Baseline CE + prox L1", L1_GROUP),
    Method("periodic_lasso_k1", "Periodic Lasso re-solve (K=1)", L1_GROUP),
    Method(
        "magnitude_pruning_875",
        "Magnitude pruning 87.5% + fine-tuning",
        L1_GROUP,
    ),
    Method("varpro_sparse_lhalf", "VarPro sparse MSE + L1/2", LHALF_GROUP),
    Method("varpro_ce_sparse_lhalf", "VarPro sparse CE + L1/2", LHALF_GROUP),
    Method("joint_mse_lhalf", "Baseline MSE + L1/2", LHALF_GROUP),
    Method("joint_ce_lhalf", "Baseline CE + L1/2", LHALF_GROUP),
    Method("joint_mse_prox_lhalf", "Baseline MSE + prox L1/2", LHALF_GROUP),
    Method("joint_ce_prox_lhalf", "Baseline CE + prox L1/2", LHALF_GROUP),
]


class DenseMoEFeatureNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.gate = nn.Linear(28 * 28, N_EXPERTS)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(28 * 28, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                    nn.ReLU(),
                )
                for _ in range(N_EXPERTS)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = self.flatten(x)
        gate_weights = torch.softmax(self.gate(flattened), dim=1)
        expert_outputs = torch.stack(
            [expert(flattened) for expert in self.experts],
            dim=1,
        )
        return (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)


class MoEClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = DenseMoEFeatureNet()
        self.readout = nn.Linear(HIDDEN_DIM, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(x))


def load_mnist(data_dir: Path, seed: int) -> tuple[torch.Tensor, ...]:
    """Load MNIST through torchvision and reproduce the paper subsampling."""
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:
        raise RuntimeError(
            "This reproduction test requires torchvision: pip install torchvision"
        ) from error

    train = MNIST(root=data_dir, train=True, download=True)
    test = MNIST(root=data_dir, train=False, download=True)
    x_train_all = train.data.numpy().astype(np.float32).reshape(60_000, -1) / 255.0
    y_train_all = train.targets.numpy().astype(np.int64)
    x_test_all = test.data.numpy().astype(np.float32).reshape(10_000, -1) / 255.0
    y_test_all = test.targets.numpy().astype(np.int64)

    all_pixels = np.concatenate([x_train_all, x_test_all], axis=0)
    mean = all_pixels.mean()
    standard_deviation = all_pixels.std()
    x_train_all = (x_train_all - mean) / (standard_deviation + 1e-8)
    x_test_all = (x_test_all - mean) / (standard_deviation + 1e-8)

    generator = np.random.default_rng(seed)
    train_indices = generator.choice(60_000, N_TRAIN, replace=False)
    test_indices = generator.choice(10_000, N_TEST, replace=False)
    x_train = torch.from_numpy(
        x_train_all[train_indices].reshape(-1, 1, 28, 28)
    )
    y_train = torch.from_numpy(y_train_all[train_indices])
    x_test = torch.from_numpy(x_test_all[test_indices].reshape(-1, 1, 28, 28))
    y_test = torch.from_numpy(y_test_all[test_indices])
    y_train_one_hot = torch.zeros(N_TRAIN, N_CLASSES).scatter_(
        1,
        y_train[:, None],
        1.0,
    )
    return x_train, y_train, y_train_one_hot, x_test, y_test


def mse_loss(logits: torch.Tensor, one_hot_targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logits - one_hot_targets).square().sum(dim=1).mean()


@torch.no_grad()
def accuracy(
    model: MoEClassifier,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> float:
    return (model(x_test).argmax(dim=1) == y_test).float().mean().item()


@torch.no_grad()
def readout_sparsity(model: MoEClassifier) -> float:
    return (
        (model.readout.weight.abs() < SPARSITY_THRESHOLD)
        .float()
        .mean()
        .item()
    )


@torch.no_grad()
def get_readout(model: MoEClassifier) -> torch.Tensor:
    return torch.cat(
        [model.readout.weight.T, model.readout.bias[None, :]],
        dim=0,
    )


@torch.no_grad()
def set_readout(model: MoEClassifier, full_readout: torch.Tensor) -> None:
    model.readout.weight.copy_(full_readout[:-1].T)
    model.readout.bias.copy_(full_readout[-1])


def augment(features: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [features, torch.ones(features.shape[0], 1, dtype=features.dtype)],
        dim=1,
    )


@torch.no_grad()
def solve_mse_readout(
    model: MoEClassifier,
    x_train: torch.Tensor,
    y_train_one_hot: torch.Tensor,
    penalty: str | None,
) -> float:
    features = model.features(x_train)
    started = time.perf_counter()
    if penalty is None:
        design = augment(features)
        full_readout = torch.linalg.solve(
            design.T @ design,
            design.T @ y_train_one_hot,
        )
    else:
        full_readout = solve_sparse_readout(
            features,
            y_train_one_hot,
            ridge=SPARSE_RIDGE,
            sparsity=SPARSE_LAMBDA,
            penalty=penalty,
            bias=True,
            max_iter=SPARSE_MAX_ITER,
            W_init=get_readout(model),
        )
    set_readout(model, full_readout)
    return time.perf_counter() - started


@torch.no_grad()
def solve_sparse_ce_readout(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_readout: torch.Tensor,
    penalty: str,
) -> torch.Tensor:
    """FISTA/iteratively reweighted solve for CE plus a sparse penalty."""
    design = augment(features)
    full_readout = initial_readout.detach().clone()
    extrapolated = full_readout.clone()
    momentum = torch.ones((), dtype=features.dtype)
    spectral_norm = torch.linalg.matrix_norm(design, ord=2)
    lipschitz = (
        0.5 * spectral_norm.square() / design.shape[0] + CE_RIDGE
    )
    step = 1.0 / lipschitz.clamp_min(torch.finfo(features.dtype).eps)
    ridge_mask = torch.ones(design.shape[1], 1, dtype=features.dtype)
    sparse_mask = torch.ones_like(ridge_mask)
    sparse_mask[-1] = 0.0
    row_indices = torch.arange(design.shape[0])

    for _ in range(SPARSE_MAX_ITER):
        previous = full_readout
        probabilities = torch.softmax(design @ extrapolated, dim=1)
        probabilities[row_indices, labels] -= 1.0
        gradient = design.T @ probabilities / design.shape[0]
        gradient = gradient + CE_RIDGE * ridge_mask * extrapolated
        if penalty == "l1":
            threshold = step * SPARSE_LAMBDA * sparse_mask
        else:
            weights = 0.5 / torch.sqrt(extrapolated.abs() + LHALF_EPS)
            threshold = step * SPARSE_LAMBDA * sparse_mask * weights
        proposal = extrapolated - step * gradient
        full_readout = (
            proposal.sign() * (proposal.abs() - threshold).clamp_min(0.0)
        )
        next_momentum = 0.5 * (
            1.0 + torch.sqrt(1.0 + 4.0 * momentum.square())
        )
        extrapolated = full_readout + (
            (momentum - 1.0) / next_momentum
        ) * (full_readout - previous)
        momentum = next_momentum
        relative_change = (
            (full_readout - previous).norm()
            / previous.norm().clamp_min(1.0)
        )
        if relative_change <= 1e-6:
            break
    return full_readout


def solve_ce_and_set_readout(
    model: MoEClassifier,
    x_train: torch.Tensor,
    labels: torch.Tensor,
    penalty: str | None,
) -> float:
    features = model.features(x_train).detach()
    started = time.perf_counter()
    if penalty is None:
        weights, bias = solve_ce_readout(
            features,
            labels,
            num_classes=N_CLASSES,
            ridge=CE_RIDGE,
            max_iter=CE_MAX_ITER,
            W_init=model.readout.weight.detach().T.contiguous(),
            b_init=model.readout.bias.detach(),
            regularize_bias=True,
        )
        full_readout = torch.cat([weights, bias[None, :]], dim=0)
    else:
        full_readout = solve_sparse_ce_readout(
            features,
            labels,
            get_readout(model),
            penalty,
        )
    set_readout(model, full_readout)
    return time.perf_counter() - started


@torch.no_grad()
def apply_prox(model: MoEClassifier, penalty: str) -> None:
    weights = model.readout.weight
    effective_lambda = ADAM_LR * SPARSE_LAMBDA
    if penalty == "l1":
        weights.copy_(
            weights.sign() * (weights.abs() - effective_lambda).clamp_min(0.0)
        )
        return
    cutoff = (
        (54.0 ** (1.0 / 3.0))
        * effective_lambda ** (2.0 / 3.0)
        / 4.0
    )
    absolute = weights.abs()
    safe = absolute.clamp_min(torch.finfo(weights.dtype).tiny)
    argument = (effective_lambda / 8.0) * (safe / 3.0).pow(-1.5)
    phi = torch.arccos(argument.clamp(-1.0, 1.0))
    value = (
        (2.0 / 3.0)
        * absolute
        * (1.0 + torch.cos(2.0 * math.pi / 3.0 - 2.0 * phi / 3.0))
    )
    weights.copy_(
        torch.where(
            absolute > cutoff,
            weights.sign() * value,
            torch.zeros_like(weights),
        )
    )


@torch.no_grad()
def prune_readout(model: MoEClassifier) -> torch.Tensor:
    absolute = model.readout.weight.abs().flatten()
    count = min(
        max(int(math.floor(PRUNING_TARGET * absolute.numel())), 1),
        absolute.numel() - 1,
    )
    threshold = torch.kthvalue(absolute, count).values
    mask = (model.readout.weight.abs() > threshold).to(absolute.dtype)
    model.readout.weight.mul_(mask)
    return mask


def readout_penalty(model: MoEClassifier, penalty: str) -> torch.Tensor:
    weights = model.readout.weight
    if penalty == "l1":
        return weights.abs().sum()
    return torch.sqrt(weights.abs() + LHALF_EPS).sum()


def joint_step(
    model: MoEClassifier,
    optimizer: torch.optim.Optimizer,
    x_train: torch.Tensor,
    labels: torch.Tensor,
    one_hot_targets: torch.Tensor,
    method: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    is_ce = method.startswith("joint_ce")
    loss = (
        nn.functional.cross_entropy(model(x_train), labels)
        if is_ce
        else mse_loss(model(x_train), one_hot_targets)
    )
    if method.endswith("_l1") and "_prox_" not in method:
        loss = loss + SPARSE_LAMBDA * readout_penalty(model, "l1")
    elif method.endswith("_lhalf") and "_prox_" not in method:
        loss = loss + SPARSE_LAMBDA * readout_penalty(model, "l1/2")
    loss.backward()
    optimizer.step()
    if "_prox_l1" in method:
        apply_prox(model, "l1")
    elif "_prox_lhalf" in method:
        apply_prox(model, "l1/2")
    return loss.item()


def varpro_step(
    model: MoEClassifier,
    optimizer: torch.optim.Optimizer,
    x_train: torch.Tensor,
    labels: torch.Tensor,
    one_hot_targets: torch.Tensor,
    loss_name: str,
    penalty: str | None,
) -> tuple[float, float]:
    if loss_name == "ce":
        solve_time = solve_ce_and_set_readout(
            model,
            x_train,
            labels,
            penalty,
        )
    else:
        solve_time = solve_mse_readout(
            model,
            x_train,
            one_hot_targets,
            penalty,
        )
    optimizer.zero_grad(set_to_none=True)
    loss = (
        nn.functional.cross_entropy(model(x_train), labels)
        if loss_name == "ce"
        else mse_loss(model(x_train), one_hot_targets)
    )
    loss.backward()
    optimizer.step()
    return loss.item(), solve_time


def varpro_specification(method: str) -> tuple[str, str | None]:
    loss_name = "ce" if method.startswith("varpro_ce") else "mse"
    if method.endswith("_l1"):
        return loss_name, "l1"
    if method.endswith("_lhalf"):
        return loss_name, "l1/2"
    return loss_name, None


def run_method(
    method: Method,
    seed: int,
    epochs: int,
    data_dir: Path,
    target_accuracy: float,
    raw_writer: csv.DictWriter,
) -> dict[str, float | str | int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, labels, one_hot, x_test, y_test = load_mnist(data_dir, seed)
    model = MoEClassifier()
    is_varpro = method.key.startswith("varpro")
    parameters = model.features.parameters() if is_varpro else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=ADAM_LR)
    started = time.perf_counter()
    solve_time_total = 0.0
    time_to_target = math.nan
    pruning_mask = None

    for epoch in range(1, epochs + 1):
        if is_varpro:
            loss_name, penalty = varpro_specification(method.key)
            loss, solve_time = varpro_step(
                model,
                optimizer,
                x_train,
                labels,
                one_hot,
                loss_name,
                penalty,
            )
            solve_time_total += solve_time
        elif method.key == "periodic_lasso_k1":
            loss = joint_step(
                model,
                optimizer,
                x_train,
                labels,
                one_hot,
                "joint_mse_prox_l1",
            )
            solve_time_total += solve_mse_readout(
                model,
                x_train,
                one_hot,
                "l1",
            )
        elif method.key == "magnitude_pruning_875":
            loss = joint_step(
                model,
                optimizer,
                x_train,
                labels,
                one_hot,
                "joint_ce",
            )
            if epoch == int(PRUNING_EPOCH_FRACTION * epochs):
                pruning_mask = prune_readout(model)
            if pruning_mask is not None:
                with torch.no_grad():
                    model.readout.weight.mul_(pruning_mask)
        else:
            loss = joint_step(
                model,
                optimizer,
                x_train,
                labels,
                one_hot,
                method.key,
            )

        elapsed = time.perf_counter() - started
        test_accuracy = accuracy(model, x_test, y_test)
        sparsity = readout_sparsity(model)
        if math.isnan(time_to_target) and test_accuracy >= target_accuracy:
            time_to_target = elapsed
        raw_writer.writerow(
            {
                "method": method.key,
                "label": method.label,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "accuracy": test_accuracy,
                "readout_sparsity": sparsity,
                "time_seconds": elapsed,
                "inner_solve_seconds": solve_time_total,
            }
        )
    if test_accuracy < target_accuracy:
        time_to_target = math.nan
    return {
        "method": method.key,
        "label": method.label,
        "seed": seed,
        "accuracy": test_accuracy,
        "readout_sparsity": sparsity,
        "time_seconds": elapsed,
        "inner_solve_seconds": solve_time_total,
        "time_to_target": time_to_target,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if np.isnan(array).all():
        return math.nan, math.nan
    return float(np.nanmean(array)), float(np.nanstd(array))


def display(mean: float, standard_deviation: float, scale: float = 1.0) -> str:
    if math.isnan(mean):
        return "not reached"
    return f"{scale * mean:.1f} +/- {scale * standard_deviation:.1f}"


def write_summary(
    results: list[dict[str, float | str | int]],
    methods: list[Method],
    output_dir: Path,
    target_accuracy: float,
) -> None:
    summary_rows = []
    with (output_dir / "sparse_moe_summary.csv").open(
        "w",
        newline="",
    ) as handle:
        fieldnames = [
            "method",
            "label",
            "n_seeds",
            "accuracy_mean",
            "accuracy_std",
            "sparsity_mean",
            "sparsity_std",
            "time_mean",
            "time_std",
            "inner_solve_time_mean",
            "inner_solve_fraction_mean",
            "time_to_target_mean",
            "time_to_target_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            selected = [row for row in results if row["method"] == method.key]
            accuracy_values = [
                float(row["accuracy"]) for row in selected
            ]
            sparsity_values = [
                float(row["readout_sparsity"]) for row in selected
            ]
            time_values = [
                float(row["time_seconds"]) for row in selected
            ]
            inner_values = [
                float(row["inner_solve_seconds"]) for row in selected
            ]
            target_values = [
                float(row["time_to_target"]) for row in selected
            ]
            accuracy_mean, accuracy_std = mean_std(accuracy_values)
            sparsity_mean, sparsity_std = mean_std(sparsity_values)
            time_mean, time_std = mean_std(time_values)
            inner_mean, _ = mean_std(inner_values)
            target_mean, target_std = mean_std(target_values)
            summary = {
                "method": method.key,
                "label": method.label,
                "n_seeds": len(selected),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "sparsity_mean": sparsity_mean,
                "sparsity_std": sparsity_std,
                "time_mean": time_mean,
                "time_std": time_std,
                "inner_solve_time_mean": inner_mean,
                "inner_solve_fraction_mean": inner_mean / time_mean,
                "time_to_target_mean": target_mean,
                "time_to_target_std": target_std,
            }
            writer.writerow(summary)
            summary_rows.append(summary)

    lines = [
        "# Sparse MoE baseline comparison",
        "",
        "MNIST, n_train=3,000, n_test=1,000, h=32, k=4, "
        "Adam, 100 epochs, five seeds.",
        "",
        "| Method | Test accuracy (%) | Readout sparsity (%) | "
        "Total time (s) | Time to 90% acc. (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    current_group = None
    for method, row in zip(methods, summary_rows):
        if method.group != current_group:
            lines.append(f"| **{method.group}** | | | | |")
            current_group = method.group
        lines.append(
            "| {label} | {accuracy} | {sparsity} | {runtime} | {target} |".format(
                label=method.label,
                accuracy=display(
                    float(row["accuracy_mean"]),
                    float(row["accuracy_std"]),
                    scale=100.0,
                ),
                sparsity=display(
                    float(row["sparsity_mean"]),
                    float(row["sparsity_std"]),
                    scale=100.0,
                ),
                runtime=display(
                    float(row["time_mean"]),
                    float(row["time_std"]),
                ),
                target=display(
                    float(row["time_to_target_mean"]),
                    float(row["time_to_target_std"]),
                ),
            )
        )
    (output_dir / "sparse_moe_table.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sparse_moe_baselines"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method.key for method in METHODS],
    )
    args = parser.parse_args()
    methods = (
        [method for method in METHODS if method.key in args.methods]
        if args.methods
        else METHODS
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    results = []
    raw_path = args.output_dir / "sparse_moe_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "label",
                "seed",
                "epoch",
                "loss",
                "accuracy",
                "readout_sparsity",
                "time_seconds",
                "inner_solve_seconds",
            ],
        )
        writer.writeheader()
        for method in methods:
            print(f"\n== {method.label} ==", flush=True)
            for seed in range(args.seeds):
                result = run_method(
                    method,
                    seed,
                    args.epochs,
                    args.data_dir,
                    args.target_accuracy,
                    writer,
                )
                results.append(result)
                handle.flush()
                print(
                    f"seed={seed} "
                    f"accuracy={100 * float(result['accuracy']):.1f}% "
                    f"sparsity={100 * float(result['readout_sparsity']):.1f}% "
                    f"time={float(result['time_seconds']):.2f}s",
                    flush=True,
                )
    write_summary(results, methods, args.output_dir, args.target_accuracy)
    print(f"Wrote {raw_path}")
    print(f"Wrote {args.output_dir / 'sparse_moe_summary.csv'}")
    print(f"Wrote {args.output_dir / 'sparse_moe_table.md'}")


if __name__ == "__main__":
    main()
