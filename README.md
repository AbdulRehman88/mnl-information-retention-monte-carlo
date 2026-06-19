# Finite-Sample Information Retention in Multinomial Logit Estimation

This repository contains the source code, simulation outputs, reproducibility scripts, and manuscript figures for the study:

**Finite-Sample Information Retention in Multinomial Logit Estimation: A Monte Carlo Study of Full and Matched Binary Maximum Likelihood Estimators**

## Overview

This project implements a Monte Carlo simulation framework for comparing the finite-sample behavior of the full multinomial logit maximum likelihood estimator with matched binary logit estimators. The study evaluates estimator behavior across seven covariate structures, fixed-design and random-design simulation protocols, and a targeted reference-category rarity sensitivity experiment.

The main contribution is an information-retention perspective: matched binary reductions are conditionally valid under the maintained multinomial logit data-generating process, but they may lose finite-sample efficiency because they discard observations outside the retained binary comparison.

## Repository contents

- `main.py`: command-line entry point for running the Monte Carlo simulations.
- `src/`: source modules for configuration, data generation, estimation, metrics, reporting, and scenario execution.
- `scripts/`: reproducibility scripts for generating polished manuscript figures.
- `results/`: final archived numerical result tables used in the manuscript.
- `figures/`: regenerated polished manuscript figures.
- `docs/`: reproducibility notes.
- `requirements.txt`: Python package requirements.
- `CITATION.cff`: citation metadata.
- `LICENSE`: MIT license.

## Installation

Create and activate a Python environment, then install the required packages:

    pip install -r requirements.txt

The code was developed for Python 3.10+.

## Quick test run

To verify that the code runs correctly:

    python .\main.py --mode quick --output-dir outputs_quick_test --replications 10 --workers 1

## Reproduce the main simulations

Run the final primary Newton-CG simulation:

    python .\main.py --mode upgraded --output-dir outputs_final_primary_newton --replications 1000 --workers 4

Run the targeted reference-category rarity sensitivity simulation:

    python .\main.py --mode novelty --output-dir outputs_final_novelty_retention_1000_newton --replications 1000 --workers 4

The full simulations may take substantial time depending on hardware.

## Regenerate manuscript figures

The final result archives are already provided in `results/`, so the polished manuscript figures can be regenerated without rerunning the full simulation:

    python .\scripts\polish_final_manuscript_figures.py

This command reads:

- `results/primary_newton_tables.zip`
- `results/novelty_retention_tables.zip`

and writes figures to:

- `figures/`

## Result archives

The repository includes two final result archives:

- `results/primary_newton_tables.zip`
- `results/novelty_retention_tables.zip`

These archives contain coefficient-level and scenario-level CSV tables used to generate the manuscript tables and figures.

## License

This repository is released under the MIT License.

## Citation

If you use this code, simulation design, or result archive, please cite the associated manuscript and this repository.
