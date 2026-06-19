from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .designs import generate_design_matrix
from .dgp import compute_utilities, draw_outcomes, softmax_probabilities
from .estimators import fit_binary_logit, fit_multinomial_logit
from .metrics import summarize_estimates


@dataclass(slots=True)
class ScenarioResult:
    summary_df: pd.DataFrame
    convergence_records: list[dict[str, Any]]
    category_records: list[dict[str, Any]]
    replication_estimates: pd.DataFrame | None


def _scenario_seed(base_seed: int, coefficient_regime: str, design_mode: str, design: str, n: int) -> int:
    regime_part = sum(ord(c) for c in coefficient_regime) * 1_000
    mode_part = 0 if design_mode == "fixed" else 10_000_000
    design_part = int(design[1:]) * 100_000
    return base_seed + regime_part + mode_part + design_part + n


def run_scenario(
    cfg: SimulationConfig,
    coefficient_regime: str,
    design_mode: str,
    design: str,
    sample_size: int,
) -> ScenarioResult:
    true_beta = cfg.beta_for_regime(coefficient_regime)
    scenario_seed = _scenario_seed(cfg.base_seed, coefficient_regime, design_mode, design, sample_size)
    master_rng = np.random.default_rng(scenario_seed)

    X_fixed = None
    if design_mode == "fixed":
        X_fixed = generate_design_matrix(
            design_name=design,
            n=sample_size,
            rng=master_rng,
            cat3_values=cfg.cat3_values,
            cat3_probabilities=cfg.cat3_probabilities,
        )

    eq1_estimates, eq2_estimates = [], []
    bl13_estimates, bl23_estimates = [], []
    convergence_records, category_records, replication_rows = [], [], []

    for rep in range(cfg.replications):
        rep_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1, dtype=np.uint32))

        X = X_fixed
        if design_mode == "random":
            X = generate_design_matrix(
                design_name=design,
                n=sample_size,
                rng=rep_rng,
                cat3_values=cfg.cat3_values,
                cat3_probabilities=cfg.cat3_probabilities,
            )
        assert X is not None

        utilities = compute_utilities(X, true_beta["eq1"], true_beta["eq2"])
        probabilities = softmax_probabilities(utilities)
        y = draw_outcomes(probabilities, rep_rng)

        n_y1 = int(np.sum(y == 1))
        n_y2 = int(np.sum(y == 2))
        n_y3 = int(np.sum(y == 3))
        n_13 = n_y1 + n_y3
        n_23 = n_y2 + n_y3

        expected_prop_y1 = float(np.mean(probabilities[:, 0]))
        expected_prop_y2 = float(np.mean(probabilities[:, 1]))
        expected_prop_y3 = float(np.mean(probabilities[:, 2]))

        mnl_fit = fit_multinomial_logit(X, y, cfg.nr)

        mask_13 = np.isin(y, [1, 3])
        mask_23 = np.isin(y, [2, 3])

        y13 = (y[mask_13] == 1).astype(float)
        y23 = (y[mask_23] == 2).astype(float)

        bl13_fit = fit_binary_logit(X[mask_13], y13, cfg.nr)
        bl23_fit = fit_binary_logit(X[mask_23], y23, cfg.nr)

        eq1_est = mnl_fit.params[:, 0] if mnl_fit.converged else np.full(len(cfg.coefficient_names), np.nan)
        eq2_est = mnl_fit.params[:, 1] if mnl_fit.converged else np.full(len(cfg.coefficient_names), np.nan)
        bl13_est = bl13_fit.params if bl13_fit.converged else np.full(len(cfg.coefficient_names), np.nan)
        bl23_est = bl23_fit.params if bl23_fit.converged else np.full(len(cfg.coefficient_names), np.nan)

        eq1_estimates.append(eq1_est)
        eq2_estimates.append(eq2_est)
        bl13_estimates.append(bl13_est)
        bl23_estimates.append(bl23_est)

        base_record = {
            "coefficient_regime": coefficient_regime,
            "design_mode": design_mode,
            "design": design,
            "sample_size": sample_size,
            "replication": rep + 1,
        }

        convergence_records.append(
            {
                **base_record,
                "mnl_converged": int(mnl_fit.converged),
                "bl13_converged": int(bl13_fit.converged),
                "bl23_converged": int(bl23_fit.converged),
                "mnl_iterations": mnl_fit.iterations,
                "bl13_iterations": bl13_fit.iterations,
                "bl23_iterations": bl23_fit.iterations,
            }
        )

        category_records.append(
            {
                **base_record,
                "n_y1": n_y1,
                "n_y2": n_y2,
                "n_y3": n_y3,
                "n_13": n_13,
                "n_23": n_23,
                "prop_y1": float(n_y1 / sample_size),
                "prop_y2": float(n_y2 / sample_size),
                "prop_y3": float(n_y3 / sample_size),
                "retention_13": float(n_13 / sample_size),
                "retention_23": float(n_23 / sample_size),
                "expected_prop_y1": expected_prop_y1,
                "expected_prop_y2": expected_prop_y2,
                "expected_prop_y3": expected_prop_y3,
                "expected_retention_13": float(expected_prop_y1 + expected_prop_y3),
                "expected_retention_23": float(expected_prop_y2 + expected_prop_y3),
            }
        )

        if cfg.save_replication_estimates:
            for estimator, equation, est in (
                ("MNL", "eq1", eq1_est),
                ("MNL", "eq2", eq2_est),
                ("BL13", "eq1", bl13_est),
                ("BL23", "eq2", bl23_est),
            ):
                row = {
                    **base_record,
                    "estimator": estimator,
                    "equation": equation,
                    "n_y1": n_y1,
                    "n_y2": n_y2,
                    "n_y3": n_y3,
                    "n_13": n_13,
                    "n_23": n_23,
                    "retention_13": float(n_13 / sample_size),
                    "retention_23": float(n_23 / sample_size),
                }
                for coef_name, value in zip(cfg.coefficient_names, est):
                    row[coef_name] = value
                replication_rows.append(row)

    summary_frames = [
        summarize_estimates(np.vstack(eq1_estimates), true_beta["eq1"], cfg.alpha, "MNL", "eq1", cfg.coefficient_names, design, sample_size, design_mode, coefficient_regime),
        summarize_estimates(np.vstack(eq2_estimates), true_beta["eq2"], cfg.alpha, "MNL", "eq2", cfg.coefficient_names, design, sample_size, design_mode, coefficient_regime),
        summarize_estimates(np.vstack(bl13_estimates), true_beta["eq1"], cfg.alpha, "BL13", "eq1", cfg.coefficient_names, design, sample_size, design_mode, coefficient_regime),
        summarize_estimates(np.vstack(bl23_estimates), true_beta["eq2"], cfg.alpha, "BL23", "eq2", cfg.coefficient_names, design, sample_size, design_mode, coefficient_regime),
    ]

    return ScenarioResult(
        summary_df=pd.concat(summary_frames, ignore_index=True),
        convergence_records=convergence_records,
        category_records=category_records,
        replication_estimates=pd.DataFrame(replication_rows) if cfg.save_replication_estimates else None,
    )
