# Reproducibility Notes

This repository accompanies the manuscript:

Finite-Sample Information Retention in Multinomial Logit Estimation: A Monte Carlo Study of Full and Matched Binary Maximum Likelihood Estimators

The study uses simulated data only. No external dataset is required.

## Main simulation modes

The repository provides four execution modes:

- quick: small test run for checking installation and code execution.
- core: original core simulation configuration.
- upgraded: final primary Monte Carlo configuration used for the main Newton-CG results.
- novelty: targeted reference-category rarity sensitivity configuration.

## Primary reproduction commands

Install dependencies:

    pip install -r requirements.txt

Run the final primary simulation:

    python .\main.py --mode upgraded --output-dir outputs_final_primary_newton --replications 1000 --workers 4

Run the targeted reference-category rarity sensitivity simulation:

    python .\main.py --mode novelty --output-dir outputs_final_novelty_retention_1000_newton --replications 1000 --workers 4

Regenerate the polished manuscript figures from the archived result ZIP files:

    python .\scripts\polish_final_manuscript_figures.py

The default figure-generation command reads:

- results/primary_newton_tables.zip
- results/novelty_retention_tables.zip

and writes the polished figures to:

- figures/

## Result archives

The repository includes two final numerical result archives:

- results/primary_newton_tables.zip
- results/novelty_retention_tables.zip

These files contain the coefficient-level and scenario-level tables used to construct the manuscript tables and figures.

## Environment

The code was developed for Python 3.10+ and requires the packages listed in requirements.txt.

## Runtime note

The full 1000-replication simulations may take substantial time depending on hardware and the number of workers used. The archived result ZIP files are provided so that all manuscript tables and figures can be inspected and regenerated without rerunning the full simulation.
