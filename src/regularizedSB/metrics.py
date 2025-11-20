import math
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

def u_nom_fn(
    policy_net: nn.Module,
    T: int,
    u_max: float,
):
    """
    Construct a nominal control function from a trained policy network.
    policy_net: PolicyNet mapping (t, x) -> u(t,x) in R^d
    T:         number of time steps
    u_max:     box constraint used during training (for consistency)
    """
    device = next(policy_net.parameters()).device
    def u_nom(x: np.ndarray, i: int) -> np.ndarray:
        """
        x: (N, d) numpy array
        i: time index in {0, ..., T-1}
        """
        t_scalar = i / float(T)
        x_t = torch.from_numpy(x.astype(np.float32)).to(device)
        t_t = torch.tensor(t_scalar, dtype=torch.float32, device=device)
        with torch.no_grad():
            u = policy_net(t_t, x_t).cpu().numpy()
        return np.clip(u, -u_max, u_max)
    return u_nom



def rollout_bridge(
    X0: np.ndarray,
    T: int,
    eps: float,
    u_nom,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the learned SB from X0 under the nominal control with noise."""
    if rng is None:
        rng = np.random.default_rng()
    dt = 1.0 / T
    X = X0.copy()
    X_paths = [X0.copy()]
    for j in range(T):
        dW = rng.normal(size=X.shape) * math.sqrt(dt)
        X = X + u_nom(X, j) * dt + np.sqrt(eps) * dW
        X_paths.append(X.copy())
    X_paths = np.array(X_paths)
    return X, X_paths


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between corresponding rows of x and y."""
    num = np.sum(x * y, axis=1)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    denom = np.maximum(denom, 1e-8)
    return num / denom


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance between centroids of point clouds `a` and `b`.
    """
    return float(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)))


def rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    diff = X[:, None, :] - Y[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.exp(-dist2 / (2.0 * bandwidth**2))


def mmd_squared(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    K_xx = rbf_kernel(X, X, bandwidth)
    K_yy = rbf_kernel(Y, Y, bandwidth)
    K_xy = rbf_kernel(X, Y, bandwidth)

    n = X.shape[0]
    m = Y.shape[0]
    term_xx = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1) + 1e-8)
    term_yy = (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1) + 1e-8)
    term_xy = 2.0 * K_xy.sum() / (n * m + 1e-8)
    return float(term_xx + term_yy - term_xy)


def calculate_metrics(
    X0: np.ndarray,
    Yt: Optional[np.ndarray],
    policy_net: nn.Module,
    T: int,
    eps: float = 0.33,
    u_max: float = 3.0,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
    target_mean: Optional[np.ndarray] = None,
    bandwidth: float = 1.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Roll out the learned policy and compute centroid, cosine, and MMD metrics.
    Returns (metrics_dict, X_end, X_paths).
    """
    u_nom = u_nom_fn(policy_net, T, u_max)
    X_end, X_paths = rollout_bridge(X0, T, eps, u_nom, rng=rng)

    metrics: Dict[str, float] = {}

    if Yt is not None:
        centroid_target = np.mean(Yt, axis=0)
        metrics["centroid_distance_source"] = centroid_distance(X0, Yt)
        metrics["centroid_distance_transported"] = centroid_distance(X_end, Yt)
        metrics["cosine_similarity_source"] = float(cosine_similarity(X0, Yt).mean())
        metrics["cosine_similarity_transported"] = float(cosine_similarity(X_end, Yt).mean())
        metrics["mmd_squared"] = mmd_squared(X_end, Yt, bandwidth)
    elif target_mean is not None:
        centroid_target = target_mean
        metrics["centroid_distance_source"] = float(
            np.linalg.norm(np.mean(X0, axis=0) - centroid_target)
        )
        metrics["centroid_distance_transported"] = float(
            np.linalg.norm(np.mean(X_end, axis=0) - centroid_target)
        )
    else:
        centroid_target = np.zeros(X0.shape[1], dtype=X0.dtype)

    metrics["mean_shift"] = float(
        np.linalg.norm(np.mean(X_end, axis=0) - centroid_target)
    )
    metrics["transport_variance"] = float(np.trace(np.cov(X_end.T)))

    if verbose:
        for k, v in metrics.items():
            print(f"[metrics] {k}: {v:.4f}")

    return metrics, X_end, X_paths


# === Notebook-inspired analysis helpers (toyExample_dGauss.ipynb 1-158) ===

def extract_mse_stats(
    res: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    d_sel: int,
    lambdas: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a fixed dimension d_sel, compute active & nuisance MSE
    for each lambda in `lambdas`.
    """
    lam_list = []
    mse_active = []
    mse_nuisance = []

    for lam in lambdas:
        key = f"d={d_sel}_lam={lam}"
        entry = res[key]
        XT = entry["XT_eval"]
        tgt = entry["target_mean"]

        err_active = float(((XT[:, 0] - tgt[0]) ** 2).mean())
        if d_sel > 1:
            nuisance = XT[:, 1:]
            err_nuisance = float(((nuisance - tgt[1:]) ** 2).mean())
        else:
            err_nuisance = 0.0

        lam_list.append(lam)
        mse_active.append(err_active)
        mse_nuisance.append(err_nuisance)

    return np.array(lam_list), np.array(mse_active), np.array(mse_nuisance)


def plot_active_and_nuisance_densities(
    res: Dict[str, Dict[str, np.ndarray]],
    d_sel: int,
    lambdas: Sequence[float],
    bins: int = 80,
) -> None:
    """
    Replicates notebook plotting utilities for active (dim 0) and nuisance dims.
    """
    if not lambdas:
        raise ValueError("`lambdas` must contain at least one value.")
    key0 = f"d={d_sel}_lam={lambdas[0]}"
    base_entry = res[key0]
    X0 = base_entry["X0"]
    target_mean = base_entry["target_mean"]
    target_active = float(target_mean[0])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax_active, ax_nuis = ax

    x0_active = X0[:, 0]
    ax_active.hist(
        x0_active,
        bins=bins,
        density=True,
        alpha=0.4,
        label="source X0 (dim 0)",
    )

    for lam in lambdas:
        key = f"d={d_sel}_lam={lam}"
        XT = res[key]["XT_eval"]
        xt_active = XT[:, 0]
        ax_active.hist(
            xt_active,
            bins=bins,
            density=True,
            alpha=0.4,
            histtype="step",
            label=f"XT (dim 0, λ={lam})",
        )

    ax_active.axvline(
        x=target_active,
        linestyle="--",
        linewidth=1.5,
        label=f"target mean={target_active}",
    )
    ax_active.set_title(f"Active dimension (dim 0), d={d_sel}")
    ax_active.set_xlabel("x_0")
    ax_active.set_ylabel("density")
    ax_active.legend()

    if d_sel > 1:
        x0_nuis = X0[:, 1:].ravel()
        ax_nuis.hist(
            x0_nuis,
            bins=bins,
            density=True,
            alpha=0.4,
            label="source X0 (dims 1..d-1)",
        )
        for lam in lambdas:
            key = f"d={d_sel}_lam={lam}"
            XT = res[key]["XT_eval"]
            xt_nuis = XT[:, 1:].ravel()
            ax_nuis.hist(
                xt_nuis,
                bins=bins,
                density=True,
                alpha=0.4,
                histtype="step",
                label=f"XT (dims 1..d-1, λ={lam})",
            )
        ax_nuis.axvline(
            x=0.0,
            linestyle="--",
            linewidth=1.5,
            label="target mean (nuisance) = 0",
        )
        ax_nuis.set_title(f"Nuisance dimensions (1..{d_sel-1}), stacked")
        ax_nuis.set_xlabel("x_j, j=1..d-1")
        ax_nuis.set_ylabel("density")
        ax_nuis.legend()
    else:
        ax_nuis.set_visible(False)

    plt.tight_layout()


def plot_mse_vs_lambda(
    res: Dict[str, Dict[str, np.ndarray]],
    d_sel: int,
    lambdas: Sequence[float],
) -> None:
    lam_arr, mse_active, mse_nuis = extract_mse_stats(res, d_sel, lambdas)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(lam_arr, mse_active, marker="o", label="Active dim (dim 0) MSE")
    if d_sel > 1:
        ax.plot(lam_arr, mse_nuis, marker="s", label="Nuisance dims MSE")
    ax.set_xlabel("lambda")
    ax.set_ylabel("MSE to target")
    ax.set_title(f"MSE vs lambda (d={d_sel})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
