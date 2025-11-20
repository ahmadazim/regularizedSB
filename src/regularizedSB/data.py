from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DTYPE = np.float64


def make_sparse_mean(
    d: int,
    k: int,
    shift_scale: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Construct a k-sparse mean vector in R^d by choosing k coordinates at random.
    """
    if rng is None:
        rng = np.random.default_rng()
    m = np.zeros(d, dtype=DTYPE)
    idx = rng.choice(d, size=min(k, d), replace=False)
    signs = rng.choice([-1.0, 1.0], size=idx.shape[0])
    m[idx] = shift_scale * signs
    return m


def sample_gaussian_shift_data(
    d: int,
    k: int,
    n_samples: int,
    shift_scale: float = 2.0,
    var: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Toy problem:
        source ~ N(0, var I_d)
        target ~ N(m, var I_d) with sparse mean m.
    """
    if rng is None:
        rng = np.random.default_rng()
    m = make_sparse_mean(d, k, shift_scale=shift_scale, rng=rng)
    source = rng.normal(loc=0.0, scale=np.sqrt(var), size=(n_samples, d)).astype(DTYPE)
    target = rng.normal(loc=m, scale=np.sqrt(var), size=(n_samples, d)).astype(DTYPE)
    return source, target, m

