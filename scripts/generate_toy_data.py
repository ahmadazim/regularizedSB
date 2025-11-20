#!/usr/bin/env python
import argparse
import os

import numpy as np

from src.regularizedSB.data import sample_gaussian_shift_data


def parse_args():
    ap = argparse.ArgumentParser(description="Generate toy Gaussian source/target datasets.")
    ap.add_argument("--dim", type=int, default=4, help="Ambient dimension d.")
    ap.add_argument("--sparsity", type=int, default=1, help="Number of shifted coordinates.")
    ap.add_argument("--shift", type=float, default=2.0, help="Shift magnitude for active dims.")
    ap.add_argument("--var", type=float, default=1.0, help="Variance of the Gaussians.")
    ap.add_argument("--n_samples", type=int, default=4096, help="Number of particles to sample.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")
    ap.add_argument(
        "--output",
        type=str,
        default="data/toy_dataset.npz",
        help="Path to save the dataset (.npz with source/target).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    source, target, mean = sample_gaussian_shift_data(
        d=args.dim,
        k=args.sparsity,
        n_samples=args.n_samples,
        shift_scale=args.shift,
        var=args.var,
        rng=rng,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        source=source,
        target=target,
        target_mean=mean,
    )
    print(f"Saved dataset to {args.output} (source/target shape {source.shape}).")


if __name__ == "__main__":
    main()

