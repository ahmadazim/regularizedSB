from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

PenaltyEval = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class TerminalPenaltyBundle:
    name: str
    evaluate: PenaltyEval
    metadata: Dict[str, Any]


def terminal_penalty(X: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
    """
    Quadratic penalty centered at target_mean:
        g(x) = 0.5 * ||x - μ||^2
    """
    diff = X - target_mean[None, :]
    return 0.5 * np.sum(diff * diff, axis=1)


def terminal_penalty_grad(X: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
    """
    Gradient of quadratic penalty wrt X: ∇g(x) = x - μ
    """
    return X - target_mean[None, :]


def huber_gaussian_terminal(
    X: np.ndarray,
    mu: np.ndarray,
    R: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Huberized quadratic terminal cost:
        r = ||x - mu||
        g(x) = 0.5 r^2             if r <= R
             = 0.5 R^2 + R (r - R) if r > R
    """
    diff = X - mu                    # (N, d)
    r2 = np.sum(diff**2, axis=1)     # (N,)
    r = np.sqrt(r2 + 1e-8)           # (N,)

    inside = r <= R
    g_vec = np.empty_like(r)
    g_vec[inside] = 0.5 * r2[inside]
    g_vec[~inside] = 0.5 * R**2 + R * (r[~inside] - R)

    phi_prime = np.where(inside, r, R)    # (N,)
    scale = (phi_prime / (r + 1e-8))[:, None]
    grad = scale * diff

    return g_vec.astype(X.dtype), grad.astype(X.dtype)


def gaussian_terminal(
    X: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian negative-log-likelihood terminal cost with covariance cov.
    Returns per-sample cost and gradient.
    """
    cov = np.asarray(cov, dtype=X.dtype)
    cov_inv = np.linalg.inv(cov)
    diff = X - mu
    quad = 0.5 * np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
    grad = diff @ cov_inv.T
    return quad.astype(X.dtype), grad.astype(X.dtype)


def centroid_terminal(
    X: np.ndarray,
    target_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Penalize deviation of empirical centroid from target_mean.
    All samples share the same scalar penalty; gradients distribute uniformly.
    """
    centroid = X.mean(axis=0)
    diff = centroid - target_mean
    g_scalar = 0.5 * float(np.dot(diff, diff))
    grad = np.broadcast_to(diff / X.shape[0], X.shape)
    g_vec = np.full(X.shape[0], g_scalar, dtype=X.dtype)
    return g_vec, grad.astype(X.dtype)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    diff = X[:, None, :] - Y[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.exp(-dist2 / (2.0 * bandwidth**2))


def mmd_terminal(
    X: np.ndarray,
    target_samples: np.ndarray,
    bandwidth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maximum mean discrepancy penalty between transported samples X and reference samples.
    """
    N = X.shape[0]
    M = target_samples.shape[0]
    K_xx = _rbf_kernel(X, X, bandwidth)
    K_xy = _rbf_kernel(X, target_samples, bandwidth)
    term_xx = K_xx.sum() / (N * N + 1e-8)
    term_yy = _rbf_kernel(target_samples, target_samples, bandwidth).sum() / (M * M + 1e-8)
    term_xy = K_xy.sum() * 2.0 / (N * M + 1e-8)
    mmd2 = term_xx + term_yy - term_xy

    # Gradient
    diff_xx = X[:, None, :] - X[None, :, :]
    grad_xx = (K_xx[..., None] * diff_xx).sum(axis=1) * (-2.0 / (bandwidth**2 * N * N + 1e-8))
    diff_xy = X[:, None, :] - target_samples[None, :, :]
    grad_xy = (K_xy[..., None] * diff_xy).sum(axis=1) * (2.0 / (bandwidth**2 * N * M + 1e-8))
    grad = grad_xx + grad_xy

    g_vec = np.full(N, mmd2, dtype=X.dtype)
    return g_vec, grad.astype(X.dtype)


def build_terminal_penalty(
    name: str = "quadratic",
    *,
    target_mean: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> TerminalPenaltyBundle:
    """
    Factory returning a callable g(X) -> (penalty_vec, grad) along with metadata.
    """
    params = params or {}
    name = name.lower()

    def require_target_mean() -> np.ndarray:
        if target_mean is None:
            raise ValueError(f"Penalty '{name}' requires target_mean.")
        return np.asarray(target_mean, dtype=np.float64)

    if name in ("quadratic", "mean_shift"):
        mu = require_target_mean()

        def evaluate(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            g = terminal_penalty(X, mu)
            grad = terminal_penalty_grad(X, mu)
            return g.astype(X.dtype), grad.astype(X.dtype)

        return TerminalPenaltyBundle(name="quadratic", evaluate=evaluate, metadata={"target_mean": mu})

    if name == "huber":
        mu = require_target_mean()
        radius = float(params.get("radius", 1.0))

        def evaluate(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return huber_gaussian_terminal(X, mu, radius)

        return TerminalPenaltyBundle(
            name="huber", evaluate=evaluate, metadata={"target_mean": mu, "radius": radius}
        )

    if name == "gaussian":
        mu = require_target_mean()
        if "covariance" in params:
            cov = np.asarray(params["covariance"], dtype=np.float64)
        else:
            variance = float(params.get("variance", 1.0))
            d = mu.shape[0]
            cov = np.eye(d, dtype=np.float64) * variance

        def evaluate(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return gaussian_terminal(X, mu, cov)

        return TerminalPenaltyBundle(
            name="gaussian", evaluate=evaluate, metadata={"target_mean": mu, "covariance": cov}
        )

    if name == "centroid":
        mu = require_target_mean()

        def evaluate(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return centroid_terminal(X, mu)

        return TerminalPenaltyBundle(name="centroid", evaluate=evaluate, metadata={"target_mean": mu})

    if name == "mmd":
        target_samples = params.get("target_samples")
        path = params.get("target_samples_path")
        if target_samples is None and path:
            target_samples = np.load(os.path.expanduser(path))
        if target_samples is None:
            raise ValueError("MMD penalty requires target_samples or target_samples_path.")
        target_samples = np.asarray(target_samples, dtype=np.float64)
        bandwidth = float(params.get("bandwidth", 1.0))

        def evaluate(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return mmd_terminal(X, target_samples, bandwidth)

        metadata = {
            "target_samples": target_samples,
            "target_mean": target_samples.mean(axis=0),
            "bandwidth": bandwidth,
        }
        return TerminalPenaltyBundle(name="mmd", evaluate=evaluate, metadata=metadata)

    raise ValueError(f"Unknown terminal penalty '{name}'")
