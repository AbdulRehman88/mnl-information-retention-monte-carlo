from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    SimulationConfig,
    core_paper_config,
    novelty_retention_config,
    quick_config,
    upgraded_config,
)
from src.runner import run_all


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo comparison of multinomial and binary logit estimators")
    parser.add_argument("--mode", choices=["quick", "core", "upgraded", "novelty"], default="quick")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-raw", action="store_true", help="save all replication-level estimates")
    return parser


def build_config(args: argparse.Namespace) -> SimulationConfig:
    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.mode == "quick":
        cfg = quick_config(out_dir or "outputs_quick")
    elif args.mode == "core":
        cfg = core_paper_config(out_dir or "outputs_core")
    elif args.mode == "upgraded":
        cfg = upgraded_config(out_dir or "outputs_upgraded")
    else:
        cfg = novelty_retention_config(out_dir or "outputs_novelty_retention")

    if args.replications is not None:
        cfg.replications = int(args.replications)
    if args.skip_plots:
        cfg.make_plots = False
    if args.save_raw:
        cfg.save_replication_estimates = True

    cfg.validate()
    return cfg


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = build_config(args)
    print(f"Running mode={args.mode} with output_dir={cfg.output_dir}")
    run_all(cfg, max_workers=args.workers)
    print("Finished.")
    print(f"Results saved under: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
