from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from .config import OptimizerConfig
from .dgp import softmax_probabilities


@dataclass(slots=True)
class FitResult:
    params: np.ndarray
    converged: bool
    iterations: int
    loglik: float
    message: str = ""



def _binary_loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = np.clip(X @ beta, -35.0, 35.0)
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))



def _binary_negloglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return -_binary_loglik(X, y, beta)



def _binary_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    eta = np.clip(X @ beta, -35.0, 35.0)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    return -(X.T @ (y - p_hat))



def _binary_hessian(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    eta = np.clip(X @ beta, -35.0, 35.0)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    weights = p_hat * (1.0 - p_hat)
    hess = X.T @ (weights[:, None] * X)
    return hess + np.eye(X.shape[1]) * 1e-8



def fit_binary_logit(
    X: np.ndarray,
    y: np.ndarray,
    cfg: OptimizerConfig,
    start: np.ndarray | None = None,
) -> FitResult:
    n, p = X.shape
    if len(np.unique(y)) < 2:
        return FitResult(np.full(p, np.nan), False, 0, np.nan, "binary outcome has one class only")

    beta0 = np.zeros(p, dtype=float) if start is None else start.astype(float).copy()
    result = minimize(
        fun=_binary_negloglik,
        x0=beta0,
        args=(X, y),
        method="Newton-CG",
        jac=_binary_gradient,
        hess=_binary_hessian,
        options={"maxiter": cfg.max_iter, "xtol": cfg.tol, "disp": False},
    )
    grad_norm = float(np.max(np.abs(_binary_gradient(result.x, X, y)))) if np.all(np.isfinite(result.x)) else np.inf
    converged = bool(result.success or grad_norm < max(cfg.tol, 1e-5))
    return FitResult(
        params=result.x if converged else np.full(p, np.nan),
        converged=converged,
        iterations=int(getattr(result, "nit", 0) or 0),
        loglik=-float(result.fun) if np.isfinite(result.fun) else np.nan,
        message=str(result.message),
    )



def _unpack_multinomial_params(beta_flat: np.ndarray, p: int) -> np.ndarray:
    return np.column_stack([beta_flat[:p], beta_flat[p:]])



def _pack_multinomial_params(beta: np.ndarray) -> np.ndarray:
    return np.concatenate([beta[:, 0], beta[:, 1]])



def _multinomial_loglik(X: np.ndarray, y: np.ndarray, beta_flat: np.ndarray) -> float:
    p = X.shape[1]
    beta = _unpack_multinomial_params(beta_flat, p)
    utilities = np.column_stack([X @ beta[:, 0], X @ beta[:, 1], np.zeros(X.shape[0], dtype=float)])
    probs = softmax_probabilities(utilities)
    idx = y.astype(int) - 1
    return float(np.sum(np.log(np.clip(probs[np.arange(len(y)), idx], 1e-300, 1.0))))



def _multinomial_negloglik(beta_flat: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return -_multinomial_loglik(X, y, beta_flat)



def _multinomial_gradient(beta_flat: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = X.shape
    beta = _unpack_multinomial_params(beta_flat, p)
    utilities = np.column_stack([X @ beta[:, 0], X @ beta[:, 1], np.zeros(n, dtype=float)])
    probs = softmax_probabilities(utilities)
    y1 = (y == 1).astype(float)
    y2 = (y == 2).astype(float)
    p1 = probs[:, 0]
    p2 = probs[:, 1]
    g1 = -(X.T @ (y1 - p1))
    g2 = -(X.T @ (y2 - p2))
    return np.concatenate([g1, g2])



def _multinomial_hessian(beta_flat: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = X.shape
    beta = _unpack_multinomial_params(beta_flat, p)
    utilities = np.column_stack([X @ beta[:, 0], X @ beta[:, 1], np.zeros(n, dtype=float)])
    probs = softmax_probabilities(utilities)
    p1 = probs[:, 0]
    p2 = probs[:, 1]

    w11 = p1 * (1.0 - p1)
    w22 = p2 * (1.0 - p2)
    w12 = p1 * p2

    h11 = X.T @ (w11[:, None] * X)
    h22 = X.T @ (w22[:, None] * X)
    h12 = -(X.T @ (w12[:, None] * X))

    top = np.hstack([h11, h12])
    bottom = np.hstack([h12, h22])
    hess = np.vstack([top, bottom])
    return hess + np.eye(2 * p) * 1e-8



def fit_multinomial_logit(
    X: np.ndarray,
    y: np.ndarray,
    cfg: OptimizerConfig,
    start: np.ndarray | None = None,
) -> FitResult:
    p = X.shape[1]
    beta0 = np.zeros(p * 2, dtype=float) if start is None else _pack_multinomial_params(start.astype(float).copy())
    if len(np.unique(y)) < 3:
        return FitResult(np.full((p, 2), np.nan), False, 0, np.nan, "multinomial outcome missing at least one class")

    result = minimize(
        fun=_multinomial_negloglik,
        x0=beta0,
        args=(X, y),
        method="Newton-CG",
        jac=_multinomial_gradient,
        hess=_multinomial_hessian,
        options={"maxiter": cfg.max_iter, "xtol": cfg.tol, "disp": False},
    )
    grad_norm = float(np.max(np.abs(_multinomial_gradient(result.x, X, y)))) if np.all(np.isfinite(result.x)) else np.inf
    converged = bool(result.success or grad_norm < max(cfg.tol, 1e-5))
    params = _unpack_multinomial_params(result.x, p) if converged else np.full((p, 2), np.nan)
    return FitResult(
        params=params,
        converged=converged,
        iterations=int(getattr(result, "nit", 0) or 0),
        loglik=-float(result.fun) if np.isfinite(result.fun) else np.nan,
        message=str(result.message),
    )
