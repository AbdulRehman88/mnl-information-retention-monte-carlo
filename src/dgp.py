from __future__ import annotations

import numpy as np



def compute_utilities(X: np.ndarray, beta1: np.ndarray, beta2: np.ndarray) -> np.ndarray:
    v1 = X @ beta1
    v2 = X @ beta2
    v3 = np.zeros(X.shape[0], dtype=float)
    return np.column_stack([v1, v2, v3])



def softmax_probabilities(utilities: np.ndarray) -> np.ndarray:
    shifted = utilities - utilities.max(axis=1, keepdims=True)
    expu = np.exp(np.clip(shifted, -700, 700))
    return expu / expu.sum(axis=1, keepdims=True)



def draw_outcomes(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    u = rng.random(probabilities.shape[0])[:, None]
    return (u > cumulative[:, :2]).sum(axis=1) + 1
