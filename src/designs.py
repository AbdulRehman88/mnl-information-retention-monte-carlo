from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(slots=True)
class DesignSpec:
    name: str
    generator: Callable[[np.random.Generator, int, Sequence[int], Sequence[float]], np.ndarray]
    description: str


def _cat3(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return rng.choice(np.asarray(values), size=n, replace=True, p=np.asarray(probs, dtype=float)).astype(float)



def _design_d1(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return np.column_stack([
        rng.normal(0.0, 1.0, size=n),
        rng.normal(0.0, 1.0, size=n),
        rng.normal(0.0, 1.0, size=n),
    ])



def _design_d2(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return np.column_stack([
        rng.uniform(0.0, 1.0, size=n),
        rng.uniform(0.0, 1.0, size=n),
        rng.uniform(0.0, 1.0, size=n),
    ])



def _design_d3(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return np.column_stack([
        rng.binomial(1, 0.4, size=n),
        rng.binomial(1, 0.5, size=n),
        rng.binomial(1, 0.3, size=n),
    ]).astype(float)



def _design_d4(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return np.column_stack([
        _cat3(rng, n, values, probs),
        _cat3(rng, n, values, probs),
        _cat3(rng, n, values, probs),
    ])



def _design_d5(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return np.column_stack([
        rng.normal(2.0, 1.0, size=n),
        rng.normal(2.0, np.sqrt(2.0), size=n),
        rng.normal(3.0, 2.0, size=n),
    ])



def _design_d6(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return np.column_stack([
        rng.normal(0.0, 1.0, size=n),
        rng.uniform(0.0, 1.0, size=n),
        rng.binomial(1, 0.4, size=n),
    ]).astype(float)



def _design_d7(rng: np.random.Generator, n: int, values: Sequence[int], probs: Sequence[float]) -> np.ndarray:
    return np.column_stack([
        rng.uniform(0.0, 1.0, size=n),
        rng.normal(2.0, np.sqrt(2.0), size=n),
        _cat3(rng, n, values, probs),
    ])


DESIGN_SPECS = {
    "D1": DesignSpec("D1", _design_d1, "N(0,1), N(0,1), N(0,1)"),
    "D2": DesignSpec("D2", _design_d2, "U(0,1), U(0,1), U(0,1)"),
    "D3": DesignSpec("D3", _design_d3, "Bern(0.4), Bern(0.5), Bern(0.3)"),
    "D4": DesignSpec("D4", _design_d4, "Cat(3), Cat(3), Cat(3) as single coded regressors"),
    "D5": DesignSpec("D5", _design_d5, "N(2,1), N(2,2), N(3,4) with second argument interpreted as variance"),
    "D6": DesignSpec("D6", _design_d6, "N(0,1), U(0,1), Bern(0.4)"),
    "D7": DesignSpec("D7", _design_d7, "U(0,1), N(2,2), Cat(3)"),
}



def generate_design_matrix(
    design_name: str,
    n: int,
    rng: np.random.Generator,
    cat3_values: Sequence[int],
    cat3_probabilities: Sequence[float],
) -> np.ndarray:
    if design_name not in DESIGN_SPECS:
        raise KeyError(f"Unknown design: {design_name}")
    X_nonconstant = DESIGN_SPECS[design_name].generator(rng, n, cat3_values, cat3_probabilities)
    intercept = np.ones((n, 1), dtype=float)
    return np.hstack([intercept, X_nonconstant.astype(float)])
