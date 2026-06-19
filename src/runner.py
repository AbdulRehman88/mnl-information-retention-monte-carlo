from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from .config import SimulationConfig
from .metrics import (
    attach_retention_to_ratio,
    compute_re_ratio,
    summarize_category_proportions,
    summarize_convergence,
    summarize_retention_effects,
    summarize_shapiro_counts,
)
from .reporting import make_basic_plots, write_markdown_report, write_tables
from .scenario_runner import run_scenario


def _config_to_text(cfg: SimulationConfig) -> str:
    return "\n".join(
        [
            f"sample_sizes={cfg.sample_sizes}",
            f"designs={cfg.designs}",
            f"replications={cfg.replications}",
            f"design_modes={cfg.design_modes}",
            f"coefficient_regimes={cfg.coefficient_regimes}",
            f"base_seed={cfg.base_seed}",
            f"alpha={cfg.alpha}",
            f"optimizer_max_iter={cfg.nr.max_iter}",
            f"optimizer_tol={cfg.nr.tol}",
        ]
    )


def run_all(cfg: SimulationConfig, max_workers: int = 1) -> dict[str, pd.DataFrame]:
    cfg.validate()

    scenarios = [
        (regime, mode, design, n)
        for regime in cfg.coefficient_regimes
        for mode in cfg.design_modes
        for design in cfg.designs
        for n in cfg.sample_sizes
    ]

    results = []

    if max_workers <= 1:
        for regime, mode, design, n in scenarios:
            print(f"[RUN] regime={regime} mode={mode} design={design} n={n}")
            results.append(run_scenario(cfg, regime, mode, design, n))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(run_scenario, cfg, regime, mode, design, n): (regime, mode, design, n)
                for regime, mode, design, n in scenarios
            }
            for future in as_completed(futures):
                regime, mode, design, n = futures[future]
                print(f"[DONE] regime={regime} mode={mode} design={design} n={n}")
                results.append(future.result())

    summary_df = pd.concat([r.summary_df for r in results], ignore_index=True)

    convergence_records = [row for r in results for row in r.convergence_records]
    category_records = [row for r in results for row in r.category_records]

    replication_df = None
    if cfg.save_replication_estimates:
        frames = [
            r.replication_estimates
            for r in results
            if r.replication_estimates is not None and not r.replication_estimates.empty
        ]
        replication_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    shapiro_counts = summarize_shapiro_counts(summary_df)
    re_ratio_df = compute_re_ratio(summary_df)
    convergence_df = summarize_convergence(convergence_records)
    category_df = summarize_category_proportions(category_records)
    ratio_with_retention_df = attach_retention_to_ratio(re_ratio_df, category_df)
    retention_effects_df = summarize_retention_effects(ratio_with_retention_df)

    write_tables(
        base_dir=cfg.output_dir,
        summary_df=summary_df,
        shapiro_counts=shapiro_counts,
        re_ratio_df=re_ratio_df,
        convergence_df=convergence_df,
        category_df=category_df,
        ratio_with_retention_df=ratio_with_retention_df,
        retention_effects_df=retention_effects_df,
        replication_df=replication_df,
    )

    write_markdown_report(
        base_dir=cfg.output_dir,
        cfg_text=_config_to_text(cfg),
        summary_df=summary_df,
        shapiro_counts=shapiro_counts,
        convergence_df=convergence_df,
    )

    if cfg.make_plots:
        make_basic_plots(
            cfg.output_dir,
            summary_df,
            re_ratio_df,
            convergence_df,
            ratio_with_retention_df=ratio_with_retention_df,
        )

    return {
        "coefficient_summary": summary_df,
        "shapiro_counts": shapiro_counts,
        "smse_ratio": re_ratio_df,
        "convergence_summary": convergence_df,
        "category_proportions": category_df,
        "smse_ratio_with_retention": ratio_with_retention_df,
        "retention_effect_summary": retention_effects_df,
        "replication_estimates": replication_df if replication_df is not None else pd.DataFrame(),
    }
