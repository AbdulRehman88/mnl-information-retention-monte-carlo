from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "base": base_dir,
        "tables": base_dir / "tables",
        "plots": base_dir / "plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_tables(
    base_dir: Path,
    summary_df: pd.DataFrame,
    shapiro_counts: pd.DataFrame,
    re_ratio_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    category_df: pd.DataFrame,
    ratio_with_retention_df: pd.DataFrame | None = None,
    retention_effects_df: pd.DataFrame | None = None,
    replication_df: pd.DataFrame | None = None,
) -> None:
    dirs = ensure_dirs(base_dir)

    summary_df.to_csv(dirs["tables"] / "coefficient_summary.csv", index=False)
    shapiro_counts.to_csv(dirs["tables"] / "shapiro_counts.csv", index=False)
    re_ratio_df.to_csv(dirs["tables"] / "smse_ratio.csv", index=False)
    convergence_df.to_csv(dirs["tables"] / "convergence_summary.csv", index=False)
    category_df.to_csv(dirs["tables"] / "category_proportions.csv", index=False)

    if ratio_with_retention_df is not None and not ratio_with_retention_df.empty:
        ratio_with_retention_df.to_csv(dirs["tables"] / "smse_ratio_with_retention.csv", index=False)

    if retention_effects_df is not None and not retention_effects_df.empty:
        retention_effects_df.to_csv(dirs["tables"] / "retention_effect_summary.csv", index=False)

    if replication_df is not None and not replication_df.empty:
        replication_df.to_csv(dirs["tables"] / "replication_estimates.csv", index=False)


def write_markdown_report(
    base_dir: Path,
    cfg_text: str,
    summary_df: pd.DataFrame,
    shapiro_counts: pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> None:
    dirs = ensure_dirs(base_dir)

    lines = [
        "# Multinomial vs Binary Logit Monte Carlo Report",
        "",
        "## Run configuration",
        "",
        "```text",
        cfg_text,
        "```",
        "",
        "## High-level summary",
        "",
    ]

    summary_group_keys = ["coefficient_regime", "design_mode", "estimator"]
    core = (
        summary_df.groupby(summary_group_keys, dropna=False)
        .agg(
            mean_smse=("smse", "mean"),
            mean_abs_bias=("bias", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    lines.append(core.to_markdown(index=False))

    lines.extend(["", "## Shapiro-Wilk rejection counts", ""])
    lines.append(
        shapiro_counts.groupby(["coefficient_regime", "design_mode", "estimator"], dropna=False)
        .agg(mean_rejected=("rejected_coefficients", "mean"))
        .reset_index()
        .to_markdown(index=False)
    )

    lines.extend(["", "## Convergence summary", ""])
    lines.append(
        convergence_df.groupby(["coefficient_regime", "design_mode"], dropna=False)
        .agg(
            mnl_failure_rate=("mnl_failure_rate", "mean"),
            bl13_failure_rate=("bl13_failure_rate", "mean"),
            bl23_failure_rate=("bl23_failure_rate", "mean"),
        )
        .reset_index()
        .to_markdown(index=False)
    )

    (dirs["base"] / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def _safe_name(text: str) -> str:
    return str(text).replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "")


def make_basic_plots(
    base_dir: Path,
    summary_df: pd.DataFrame,
    re_ratio_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    ratio_with_retention_df: pd.DataFrame | None = None,
) -> None:
    dirs = ensure_dirs(base_dir)

    regimes = summary_df["coefficient_regime"].dropna().unique()

    for regime in regimes:
        regime_summary = summary_df[summary_df["coefficient_regime"] == regime]

        for design_mode in regime_summary["design_mode"].dropna().unique():
            mnl = regime_summary[
                (regime_summary["design_mode"] == design_mode)
                & (regime_summary["estimator"] == "MNL")
            ]

            plot_df = (
                mnl.groupby(["design", "sample_size"], dropna=False)["smse"]
                .mean()
                .reset_index()
            )

            plt.figure(figsize=(10, 6))
            for design, group in plot_df.groupby("design"):
                group = group.sort_values("sample_size")
                plt.plot(group["sample_size"], group["smse"], marker="o", label=design)

            plt.xlabel("Sample size")
            plt.ylabel("Mean SMSE across coefficients")
            plt.title(f"MNL SMSE trends | {regime} | {design_mode}")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(dirs["plots"] / f"mnl_smse_trends_{_safe_name(regime)}_{design_mode}.png", dpi=300)
            plt.close()

    if not re_ratio_df.empty:
        for regime in re_ratio_df["coefficient_regime"].dropna().unique():
            regime_ratio = re_ratio_df[re_ratio_df["coefficient_regime"] == regime]

            for comparison in regime_ratio["comparison"].dropna().unique():
                comp = regime_ratio[regime_ratio["comparison"] == comparison]
                comp_plot = (
                    comp.groupby(["design_mode", "sample_size"], dropna=False)["smse_ratio"]
                    .median()
                    .reset_index()
                )

                plt.figure(figsize=(10, 6))
                for design_mode, group in comp_plot.groupby("design_mode"):
                    group = group.sort_values("sample_size")
                    plt.plot(group["sample_size"], group["smse_ratio"], marker="o", label=design_mode)

                plt.axhline(1.0, linestyle="--")
                plt.xlabel("Sample size")
                plt.ylabel("Median binary / MNL SMSE ratio")
                plt.title(f"{comparison} | {regime}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(dirs["plots"] / f"median_ratio_{_safe_name(regime)}_{comparison}.png", dpi=300)
                plt.close()

    if ratio_with_retention_df is not None and not ratio_with_retention_df.empty:
        for regime in ratio_with_retention_df["coefficient_regime"].dropna().unique():
            regime_ret = ratio_with_retention_df[ratio_with_retention_df["coefficient_regime"] == regime]

            for comparison in regime_ret["comparison"].dropna().unique():
                comp = regime_ret[regime_ret["comparison"] == comparison].copy()

                plt.figure(figsize=(10, 6))
                for design_mode, group in comp.groupby("design_mode"):
                    plt.scatter(
                        group["retained_sample_fraction"],
                        group["smse_ratio"],
                        alpha=0.60,
                        label=design_mode,
                    )

                plt.axhline(1.0, linestyle="--")
                plt.xlabel("Retained binary sample fraction")
                plt.ylabel("Binary / MNL SMSE ratio")
                plt.title(f"SMSE ratio vs retention | {comparison} | {regime}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    dirs["plots"] / f"ratio_vs_retention_{_safe_name(regime)}_{comparison}.png",
                    dpi=300,
                )
                plt.close()

                positive = comp[comp["smse_ratio"] > 0]
                if not positive.empty:
                    plt.figure(figsize=(10, 6))
                    for design_mode, group in positive.groupby("design_mode"):
                        plt.scatter(
                            group["retained_sample_fraction"],
                            group["smse_ratio"],
                            alpha=0.60,
                            label=design_mode,
                        )

                    plt.axhline(1.0, linestyle="--")
                    plt.yscale("log")
                    plt.xlabel("Retained binary sample fraction")
                    plt.ylabel("Binary / MNL SMSE ratio, log scale")
                    plt.title(f"SMSE ratio vs retention, log scale | {comparison} | {regime}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        dirs["plots"] / f"ratio_vs_retention_log_{_safe_name(regime)}_{comparison}.png",
                        dpi=300,
                    )
                    plt.close()

    if not convergence_df.empty:
        for regime in convergence_df["coefficient_regime"].dropna().unique():
            regime_conv = convergence_df[convergence_df["coefficient_regime"] == regime]

            plot_df = (
                regime_conv.groupby(["design_mode", "sample_size"], dropna=False)[
                    ["mnl_failure_rate", "bl13_failure_rate", "bl23_failure_rate"]
                ]
                .mean()
                .reset_index()
            )

            plt.figure(figsize=(10, 6))
            for metric in ["mnl_failure_rate", "bl13_failure_rate", "bl23_failure_rate"]:
                for design_mode, group in plot_df.groupby("design_mode"):
                    group = group.sort_values("sample_size")
                    plt.plot(group["sample_size"], group[metric], marker="o", label=f"{metric}-{design_mode}")

            plt.xlabel("Sample size")
            plt.ylabel("Failure rate")
            plt.title(f"Average convergence failure rates | {regime}")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(dirs["plots"] / f"convergence_failure_rates_{_safe_name(regime)}.png", dpi=300)
            plt.close()
