from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np


DEFAULT_SAMPLE_SIZES = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]
DEFAULT_COEF_NAMES = ["intercept", "x1", "x2", "x3"]
DEFAULT_DESIGNS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

DEFAULT_TRUE_BETA = {
    "eq1": np.array([0.5772, -0.8, 1.3, -0.5], dtype=float),
    "eq2": np.array([0.5772, 0.8, -1.5, 0.7], dtype=float),
}


def make_beta_regime(intercept_shift: float = 0.0) -> dict[str, np.ndarray]:
    beta = {k: v.copy() for k, v in DEFAULT_TRUE_BETA.items()}
    beta["eq1"][0] += intercept_shift
    beta["eq2"][0] += intercept_shift
    return beta


DEFAULT_TRUE_BETA_BY_REGIME = {
    "baseline": make_beta_regime(0.0),
    "more_reference": make_beta_regime(-0.75),
    "rare_reference": make_beta_regime(0.75),
    "very_rare_reference": make_beta_regime(1.50),
}


@dataclass(slots=True)
class OptimizerConfig:
    max_iter: int = 200
    tol: float = 1e-6
    ridge: float = 1e-8
    max_step_halving: int = 12


@dataclass(slots=True)
class SimulationConfig:
    output_dir: Path = Path("outputs")
    sample_sizes: List[int] = field(default_factory=lambda: DEFAULT_SAMPLE_SIZES.copy())
    designs: List[str] = field(default_factory=lambda: DEFAULT_DESIGNS.copy())
    replications: int = 1000
    base_seed: int = 20260327
    coefficient_names: List[str] = field(default_factory=lambda: DEFAULT_COEF_NAMES.copy())

    true_beta: dict[str, np.ndarray] = field(default_factory=lambda: {k: v.copy() for k, v in DEFAULT_TRUE_BETA.items()})
    coefficient_regimes: List[str] = field(default_factory=lambda: ["baseline"])
    true_beta_by_regime: dict[str, dict[str, np.ndarray]] = field(
        default_factory=lambda: {
            regime: {k: v.copy() for k, v in beta.items()}
            for regime, beta in DEFAULT_TRUE_BETA_BY_REGIME.items()
        }
    )

    alpha: float = 0.05
    design_modes: List[str] = field(default_factory=lambda: ["fixed", "random"])
    cat3_values: Sequence[int] = (0, 1, 2)
    cat3_probabilities: Sequence[float] = (1 / 3, 1 / 3, 1 / 3)
    nr: OptimizerConfig = field(default_factory=OptimizerConfig)
    save_replication_estimates: bool = False
    make_plots: bool = True

    def beta_for_regime(self, regime: str) -> dict[str, np.ndarray]:
        if regime == "baseline" and "baseline" not in self.true_beta_by_regime:
            return {k: v.copy() for k, v in self.true_beta.items()}
        if regime not in self.true_beta_by_regime:
            raise ValueError(f"Unknown coefficient regime: {regime}")
        return {k: v.copy() for k, v in self.true_beta_by_regime[regime].items()}

    def validate(self) -> None:
        if self.replications <= 0:
            raise ValueError("replications must be positive")
        if any(n <= 0 for n in self.sample_sizes):
            raise ValueError("all sample sizes must be positive")
        if set(self.design_modes) - {"fixed", "random"}:
            raise ValueError("design_modes must be a subset of {'fixed', 'random'}")
        if len(self.cat3_values) != 3 or len(self.cat3_probabilities) != 3:
            raise ValueError("Cat(3) must have exactly three support points and three probabilities")
        if not np.isclose(sum(self.cat3_probabilities), 1.0):
            raise ValueError("cat3_probabilities must sum to 1")

        invalid_designs = set(self.designs) - set(DEFAULT_DESIGNS)
        if invalid_designs:
            raise ValueError(f"Unexpected design labels: {sorted(invalid_designs)}")

        invalid_regimes = set(self.coefficient_regimes) - set(self.true_beta_by_regime)
        if invalid_regimes:
            raise ValueError(f"Unexpected coefficient regimes: {sorted(invalid_regimes)}")


def quick_config(output_dir: str | Path = "outputs_quick") -> SimulationConfig:
    cfg = SimulationConfig(
        output_dir=Path(output_dir),
        sample_sizes=[200, 500, 1000],
        replications=100,
        design_modes=["fixed", "random"],
        coefficient_regimes=["baseline"],
        make_plots=False,
    )
    cfg.validate()
    return cfg


def core_paper_config(output_dir: str | Path = "outputs_core") -> SimulationConfig:
    cfg = SimulationConfig(
        output_dir=Path(output_dir),
        design_modes=["fixed"],
        coefficient_regimes=["baseline"],
        replications=1000,
        make_plots=True,
    )
    cfg.validate()
    return cfg


def upgraded_config(output_dir: str | Path = "outputs_upgraded") -> SimulationConfig:
    cfg = SimulationConfig(
        output_dir=Path(output_dir),
        design_modes=["fixed", "random"],
        coefficient_regimes=["baseline"],
        replications=1000,
        make_plots=True,
    )
    cfg.validate()
    return cfg


def novelty_retention_config(output_dir: str | Path = "outputs_novelty_retention") -> SimulationConfig:
    cfg = SimulationConfig(
        output_dir=Path(output_dir),
        sample_sizes=[200, 500, 1000, 2000, 5000],
        designs=["D1", "D2", "D5", "D7"],
        design_modes=["fixed", "random"],
        coefficient_regimes=["more_reference", "baseline", "rare_reference", "very_rare_reference"],
        replications=1000,
        make_plots=True,
    )
    cfg.validate()
    return cfg
