from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import shapiro


GROUP_KEYS = ["coefficient_regime", "design_mode", "design", "sample_size"]


def summarize_estimates(
    estimates: np.ndarray,
    true_values: np.ndarray,
    alpha: float,
    estimator_name: str,
    equation_name: str,
    coefficient_names: list[str],
    design: str,
    sample_size: int,
    design_mode: str,
    coefficient_regime: str = "baseline",
) -> pd.DataFrame:
    valid_mask = np.isfinite(estimates).all(axis=1)
    valid_estimates = estimates[valid_mask]
    rows = []

    for j, coef_name in enumerate(coefficient_names):
        coef_est = valid_estimates[:, j] if len(valid_estimates) else np.array([], dtype=float)
        true_val = float(true_values[j])

        if coef_est.size == 0:
            mean_est = bias = variance = mse = rmse = shapiro_p = shapiro_reject = np.nan
        else:
            mean_est = float(np.mean(coef_est))
            bias = mean_est - true_val
            variance = float(np.var(coef_est, ddof=1)) if coef_est.size > 1 else 0.0
            mse = float(np.mean((coef_est - true_val) ** 2))
            rmse = float(np.sqrt(mse))
            try:
                shapiro_p = float(shapiro(coef_est)[1]) if coef_est.size >= 3 else np.nan
            except Exception:
                shapiro_p = np.nan
            shapiro_reject = float(shapiro_p < alpha) if np.isfinite(shapiro_p) else np.nan

        rows.append(
            {
                "coefficient_regime": coefficient_regime,
                "design_mode": design_mode,
                "design": design,
                "sample_size": sample_size,
                "estimator": estimator_name,
                "equation": equation_name,
                "coefficient": coef_name,
                "true_value": true_val,
                "n_replications_total": int(estimates.shape[0]),
                "n_replications_valid": int(valid_estimates.shape[0]),
                "mean_estimate": mean_est,
                "bias": bias,
                "variance": variance,
                "smse": mse,
                "rmse": rmse,
                "shapiro_pvalue": shapiro_p,
                "shapiro_reject_5pct": shapiro_reject,
            }
        )

    return pd.DataFrame(rows)


def summarize_shapiro_counts(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df.groupby(GROUP_KEYS + ["estimator", "equation"], dropna=False)
        .agg(
            rejected_coefficients=("shapiro_reject_5pct", lambda s: int(np.nansum(s))),
            total_coefficients=("coefficient", "count"),
        )
        .reset_index()
        .assign(rejected_proportion=lambda d: d["rejected_coefficients"] / d["total_coefficients"])
    )


def compute_re_ratio(smse_df: pd.DataFrame) -> pd.DataFrame:
    base_keys = GROUP_KEYS + ["coefficient"]

    eq1_mnl = smse_df[(smse_df["estimator"] == "MNL") & (smse_df["equation"] == "eq1")]
    eq2_mnl = smse_df[(smse_df["estimator"] == "MNL") & (smse_df["equation"] == "eq2")]
    bl13 = smse_df[(smse_df["estimator"] == "BL13") & (smse_df["equation"] == "eq1")]
    bl23 = smse_df[(smse_df["estimator"] == "BL23") & (smse_df["equation"] == "eq2")]

    out = []

    for mnl, bl, comparison in (
        (eq1_mnl, bl13, "BL13_vs_MNL_eq1"),
        (eq2_mnl, bl23, "BL23_vs_MNL_eq2"),
    ):
        merged = mnl.merge(bl, on=base_keys, suffixes=("_mnl", "_bl"))
        if merged.empty:
            continue
        tmp = merged[base_keys].copy()
        tmp["comparison"] = comparison
        tmp["smse_mnl"] = merged["smse_mnl"]
        tmp["smse_binary"] = merged["smse_bl"]
        tmp["smse_ratio"] = merged["smse_bl"] / merged["smse_mnl"]
        out.append(tmp)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize_convergence(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    grouped = (
        df.groupby(GROUP_KEYS, dropna=False)
        .agg(
            mnl_converged=("mnl_converged", "sum"),
            bl13_converged=("bl13_converged", "sum"),
            bl23_converged=("bl23_converged", "sum"),
            replications=("replication", "count"),
            mean_iter_mnl=("mnl_iterations", "mean"),
            mean_iter_bl13=("bl13_iterations", "mean"),
            mean_iter_bl23=("bl23_iterations", "mean"),
        )
        .reset_index()
    )

    grouped["mnl_failure_rate"] = 1 - grouped["mnl_converged"] / grouped["replications"]
    grouped["bl13_failure_rate"] = 1 - grouped["bl13_converged"] / grouped["replications"]
    grouped["bl23_failure_rate"] = 1 - grouped["bl23_converged"] / grouped["replications"]
    return grouped


def summarize_category_proportions(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    cols = [
        "n_y1", "n_y2", "n_y3", "n_13", "n_23",
        "prop_y1", "prop_y2", "prop_y3",
        "retention_13", "retention_23",
        "expected_prop_y1", "expected_prop_y2", "expected_prop_y3",
        "expected_retention_13", "expected_retention_23",
    ]

    return df.groupby(GROUP_KEYS, dropna=False)[cols].mean().reset_index()


def attach_retention_to_ratio(re_ratio_df: pd.DataFrame, category_df: pd.DataFrame) -> pd.DataFrame:
    if re_ratio_df.empty or category_df.empty:
        return re_ratio_df.copy()

    out = re_ratio_df.merge(category_df, on=GROUP_KEYS, how="left")
    out["retained_sample_fraction"] = np.where(
        out["comparison"].eq("BL13_vs_MNL_eq1"),
        out["retention_13"],
        out["retention_23"],
    )
    out["retained_sample_size"] = np.where(
        out["comparison"].eq("BL13_vs_MNL_eq1"),
        out["n_13"],
        out["n_23"],
    )
    out["expected_retained_fraction"] = np.where(
        out["comparison"].eq("BL13_vs_MNL_eq1"),
        out["expected_retention_13"],
        out["expected_retention_23"],
    )
    return out


def summarize_retention_effects(ratio_with_retention_df: pd.DataFrame) -> pd.DataFrame:
    if ratio_with_retention_df.empty:
        return ratio_with_retention_df.copy()

    return (
        ratio_with_retention_df.groupby(GROUP_KEYS + ["comparison"], dropna=False)
        .agg(
            mean_smse_ratio=("smse_ratio", "mean"),
            median_smse_ratio=("smse_ratio", "median"),
            max_smse_ratio=("smse_ratio", "max"),
            retained_sample_fraction=("retained_sample_fraction", "mean"),
            retained_sample_size=("retained_sample_size", "mean"),
            expected_retained_fraction=("expected_retained_fraction", "mean"),
            prop_y1=("prop_y1", "mean"),
            prop_y2=("prop_y2", "mean"),
            prop_y3=("prop_y3", "mean"),
        )
        .reset_index()
    )
