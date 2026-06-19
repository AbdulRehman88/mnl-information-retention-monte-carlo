from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate polished manuscript figures from Monte Carlo result tables."
    )
    parser.add_argument(
        "--primary-input",
        type=str,
        default="results/primary_newton_tables.zip",
        help="Path to primary result ZIP file or extracted primary tables directory.",
    )
    parser.add_argument(
        "--novelty-input",
        type=str,
        default="results/novelty_retention_tables.zip",
        help="Path to novelty/retention result ZIP file or extracted novelty tables directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory where polished figures will be saved.",
    )
    return parser


def read_table(input_path: Path, filename: str) -> pd.DataFrame:
    """Read a CSV table from either an extracted directory or a ZIP archive."""
    input_path = Path(input_path)

    if input_path.is_dir():
        table_path = input_path / filename
        if not table_path.exists():
            raise FileNotFoundError(f"Missing table: {table_path}")
        return pd.read_csv(table_path)

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with ZipFile(input_path, "r") as zf:
            names = zf.namelist()
            matches = [name for name in names if name.endswith(filename)]
            if not matches:
                raise FileNotFoundError(f"Missing {filename} inside {input_path}")
            with zf.open(matches[0]) as handle:
                return pd.read_csv(handle)

    raise FileNotFoundError(
        f"Input path must be an extracted directory or ZIP archive: {input_path}"
    )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def make_legend_bold(ax, **kwargs):
    leg = ax.legend(**kwargs)
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontweight("bold")
    return leg


def create_figure2(coef: pd.DataFrame, ratio: pd.DataFrame, output_dir: Path) -> None:
    design_order = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

    if "coefficient_regime" in coef.columns:
        coef = coef[coef["coefficient_regime"].eq("baseline")].copy()

    if "coefficient_regime" in ratio.columns:
        ratio = ratio[ratio["coefficient_regime"].eq("baseline")].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    axes = axes.ravel()

    ax = axes[0]
    tmp = (
        coef[(coef["design_mode"].eq("fixed")) & (coef["estimator"].eq("MNL"))]
        .groupby(["design", "sample_size"], as_index=False)["smse"]
        .mean()
    )

    for d in design_order:
        sub = tmp[tmp["design"].eq(d)]
        ax.plot(
            sub["sample_size"],
            sub["smse"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=d,
        )

    ax.set_title("(a) Fixed design: MNL SMSE", fontweight="bold", pad=12)
    ax.set_xlabel("Sample size", fontweight="bold")
    ax.set_ylabel("Mean SMSE", fontweight="bold")
    ax.grid(True, alpha=0.25)
    make_legend_bold(ax, ncol=2, frameon=True)

    ax = axes[1]
    tmp = (
        coef[(coef["design_mode"].eq("random")) & (coef["estimator"].eq("MNL"))]
        .groupby(["design", "sample_size"], as_index=False)["smse"]
        .mean()
    )

    for d in design_order:
        sub = tmp[tmp["design"].eq(d)]
        ax.plot(
            sub["sample_size"],
            sub["smse"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=d,
        )

    ax.set_title("(b) Random design: MNL SMSE", fontweight="bold", pad=12)
    ax.set_xlabel("Sample size", fontweight="bold")
    ax.set_ylabel("Mean SMSE", fontweight="bold")
    ax.grid(True, alpha=0.25)
    make_legend_bold(ax, ncol=2, frameon=True)

    ax = axes[2]
    tmp = (
        ratio[ratio["comparison"].eq("BL13_vs_MNL_eq1")]
        .groupby(["design_mode", "sample_size"], as_index=False)["smse_ratio"]
        .median()
    )

    for mode, label in [("fixed", "Fixed"), ("random", "Random")]:
        sub = tmp[tmp["design_mode"].eq(mode)]
        ax.plot(
            sub["sample_size"],
            sub["smse_ratio"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=label,
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.4, color="black", alpha=0.70)
    ax.set_title("(c) BL(1,3) versus MNL equation 1", fontweight="bold", pad=12)
    ax.set_xlabel("Sample size", fontweight="bold")
    ax.set_ylabel("Median SMSE ratio", fontweight="bold")
    ax.grid(True, alpha=0.25)
    make_legend_bold(ax, frameon=True)

    ax = axes[3]
    tmp = (
        ratio[ratio["comparison"].eq("BL23_vs_MNL_eq2")]
        .groupby(["design_mode", "sample_size"], as_index=False)["smse_ratio"]
        .median()
    )

    for mode, label in [("fixed", "Fixed"), ("random", "Random")]:
        sub = tmp[tmp["design_mode"].eq(mode)]
        ax.plot(
            sub["sample_size"],
            sub["smse_ratio"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=label,
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.4, color="black", alpha=0.70)
    ax.set_title("(d) BL(2,3) versus MNL equation 2", fontweight="bold", pad=12)
    ax.set_xlabel("Sample size", fontweight="bold")
    ax.set_ylabel("Median SMSE ratio", fontweight="bold")
    ax.grid(True, alpha=0.25)
    make_legend_bold(ax, frameon=True)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_dir / "figure2_main_results_newton_polished.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure2_main_results_newton_polished.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figure3(ret: pd.DataFrame, output_dir: Path) -> None:
    ret = ret[ret["comparison"].eq("BL23_vs_MNL_eq2")].copy()

    summary = (
        ret.groupby("coefficient_regime", as_index=False)
        .agg(
            prop_y3=("prop_y3", "mean"),
            retention_23=("retained_sample_fraction", "mean"),
            median_ratio=("median_smse_ratio", "mean"),
        )
    )

    order = ["more_reference", "baseline", "rare_reference", "very_rare_reference"]
    labels = ["More\nreference", "Baseline", "Rare\nreference", "Very rare\nreference"]

    summary["order"] = summary["coefficient_regime"].map({v: i for i, v in enumerate(order)})
    summary = summary.sort_values("order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    x = np.arange(len(summary))

    ax.bar(x, summary["median_ratio"], width=0.56)
    ax.axhline(1.0, linestyle="--", linewidth=1.4, color="black", alpha=0.70)
    ax.set_yscale("log")
    ax.set_ylim(0.85, summary["median_ratio"].max() * 3.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Median BL(2,3) / MNL SMSE ratio", fontweight="bold")
    ax.set_xlabel("Coefficient regime", fontweight="bold")
    ax.set_title(
        "Reference-category rarity and binary-sample information retention",
        fontweight="bold",
        fontsize=19,
        pad=26,
    )
    ax.grid(True, axis="y", alpha=0.25)

    for i, (_, row) in enumerate(summary.iterrows()):
        annotation = f"$R_{{23}}$={row['retention_23']:.3f}\n$y_3$={row['prop_y3']:.3f}"
        ax.text(
            i,
            row["median_ratio"] * 1.28,
            annotation,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            clip_on=True,
        )

    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.12, right=0.97)
    fig.savefig(output_dir / "figure3_retention_sensitivity_polished.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure3_retention_sensitivity_polished.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    primary_input = Path(args.primary_input)
    novelty_input = Path(args.novelty_input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    coef = read_table(primary_input, "coefficient_summary.csv")
    ratio = read_table(primary_input, "smse_ratio.csv")
    ret = read_table(novelty_input, "retention_effect_summary.csv")

    create_figure2(coef, ratio, output_dir)
    create_figure3(ret, output_dir)

    print(f"Saved polished figures to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
