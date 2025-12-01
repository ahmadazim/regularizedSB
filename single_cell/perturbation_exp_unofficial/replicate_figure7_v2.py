"""
Figure 7 Replication: Regularized Schrödinger Bridge for Drug Perturbation

This script replicates Figure 7 from the paper using real sciPlex3 drug perturbation data.

Steps:
1. Load scRNA-seq data (Vehicle vs Drug-treated cells)
2. Select top N highly variable genes (HVGs)
3. Train Regularized SB with varying gamma (L1 regularization) values
4. Evaluate transport quality via Sinkhorn divergence
5. Evaluate sparsity of learned transport map
6. Compare against standard Sinkhorn OT baseline
7. Plot results matching Figure 7 format

Outputs:
- results.csv / results.json: Per-fold metrics for each gamma
- summary_stats.csv: Mean ± std across folds
- policies/gamma_{γ}/fold_{k}_policy.pt: Saved model weights
- experiment_config.json: Full reproducibility config
- figure7_replication.png/pdf: Main figure
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.multiprocessing as mp
import geomloss
import ot  # POT library for Sinkhorn OT baseline
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.regularizedSB.solver import RegularizedSBSolver
from src.regularizedSB.config import (
    ExperimentConfig, SolverConfig, NetworkConfig,
    LoggingConfig, DatasetConfig, PenaltyConfig, MetricsConfig
)
from src.regularizedSB.metrics import rollout_bridge, u_nom_fn

# ============================================================================
# Configuration
# ============================================================================
# Drug and cell line to analyze
DRUG = "Hesperadin"
CELL_LINE = "MCF7"

# Data path (update this to your local path)
DATA_PATH = Path("/home/tig687/regularizedSB/sc_data/sciPlex3_subset_10k.h5ad")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "PERTURB_HESPERADIN"

# Regularization values to sweep
GAMMAS = [1.0, 32.0, 64.0, 1024.0]

# Experiment parameters
N_TOP_GENES = 1024  # Number of HVGs to select
N_FOLDS = 2  # Number of cross-validation folds
SPARSITY_THRESHOLD = 1e-5  # Threshold for counting non-zero displacement

# Solver parameters
T_STEPS = 30  # Number of time discretization steps
OUTER_LOOPS = 60  # Training iterations
EPS = 0.1  # Diffusion coefficient
U_MAX = 5.0  # Box constraint on control
HIDDEN_DIM = 256  # Hidden layer size for networks
SEED = 42

# Baseline computation
COMPUTE_BASELINE = True  # Whether to compute Sinkhorn OT baseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def load_and_preprocess_data(
    data_path: Path,
    drug: str,
    n_top_genes: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load scRNA-seq data and preprocess for transport experiment.
    
    Args:
        data_path: Path to .h5ad file
        drug: Name of drug treatment to use as target
        n_top_genes: Number of HVGs to select
        
    Returns:
        source_data: (n_source, n_genes) Vehicle cells
        target_data: (n_target, n_genes) Drug-treated cells
        gene_names: List of gene names
    """
    print(f"Loading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    print(f"  Raw data: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Standard preprocessing
    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes
    if n_top_genes is not None:
        print(f"Selecting top {n_top_genes} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"  After HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Identify Source (Vehicle) and Target (Drug)
    vehicle_mask = (
        (adata.obs['vehicle'] == True) |
        (adata.obs['product_name'].str.contains('Vehicle', case=False, na=False))
    )
    drug_mask = adata.obs['product_name'].str.contains(drug, case=False, na=False)
    
    source_adata = adata[vehicle_mask].copy()
    target_adata = adata[drug_mask].copy()
    
    print(f"  Source (Vehicle) cells: {source_adata.n_obs}")
    print(f"  Target ({drug}) cells: {target_adata.n_obs}")
    
    # Convert to dense arrays
    source_data = source_adata.X
    if hasattr(source_data, 'toarray'):
        source_data = source_data.toarray()
    source_data = np.asarray(source_data, dtype=np.float64)
    
    target_data = target_adata.X
    if hasattr(target_data, 'toarray'):
        target_data = target_data.toarray()
    target_data = np.asarray(target_data, dtype=np.float64)
    
    # Extract gene names
    if 'gene_short_name' in source_adata.var.columns:
        gene_names = source_adata.var['gene_short_name'].astype(str).tolist()
    else:
        gene_names = source_adata.var_names.tolist()
    
    return source_data, target_data, gene_names


# ============================================================================
# Metrics
# ============================================================================
def compute_sinkhorn_divergence(
    X_transported: np.ndarray,
    X_target: np.ndarray,
    blur: float = 0.005,
) -> float:
    """
    Compute Sinkhorn divergence between transported and target distributions.
    
    Args:
        X_transported: (N, D) transported samples
        X_target: (M, D) target samples
        blur: Entropic regularization (lower = sharper but slower)
        
    Returns:
        Sinkhorn divergence value
    """
    xt = torch.from_numpy(X_transported.astype(np.float32)).to(DEVICE)
    yt = torch.from_numpy(X_target.astype(np.float32)).to(DEVICE)
    
    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True)
    
    with torch.no_grad():
        divergence = loss_fn(xt, yt)
    
    return float(divergence.item())


def compute_sparsity_metrics(
    X_transported: np.ndarray,
    X_source: np.ndarray,
    threshold: float = 1e-2,
) -> Dict[str, float]:
    """
    Compute sparsity-related metrics for the transport map.
    
    Uses mean displacement across cells to identify which genes are perturbed,
    then computes what fraction of genes have significant displacement.
    
    Args:
        X_transported: (N, D) transported samples
        X_source: (N, D) original source samples
        threshold: Threshold for considering displacement as "non-zero"
        
    Returns:
        Dictionary of sparsity metrics
    """
    n_cells, n_genes = X_transported.shape
    displacement = X_transported - X_source
    
    # Mean displacement per gene (average across cells)
    mean_displacement = np.mean(displacement, axis=0)  # (n_genes,)
    
    # Gene-level sparsity: % of genes with mean displacement above threshold
    n_nonzero_genes = np.sum(np.abs(mean_displacement) > threshold)
    sparsity_pct = 100.0 * n_nonzero_genes / n_genes
    
    # Per-cell sparsity (original metric): average % of genes moved per cell
    per_cell_nonzero = np.sum(np.abs(displacement) > threshold, axis=1)
    mean_sample_sparsity = 100.0 * np.mean(per_cell_nonzero) / n_genes
    
    # L1 norm of mean displacement
    l1_total = float(np.sum(np.abs(mean_displacement)))
    
    # L2 norm of mean displacement
    l2_total = float(np.linalg.norm(mean_displacement))
    
    return {
        "sparsity_pct": sparsity_pct,  # Gene-level: % of genes that moved
        "mean_sample_sparsity": mean_sample_sparsity,  # Cell-level: avg % genes per cell
        "l1_total": l1_total,
        "l2_total": l2_total,
        "n_nonzero_genes": int(n_nonzero_genes),
    }


def get_top_genes(
    X_transported: np.ndarray,
    X_source: np.ndarray,
    gene_names: List[str],
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Identify top genes by mean absolute displacement.
    
    Returns:
        Dictionary with gene names and displacement values for top K genes
    """
    displacement = X_transported - X_source
    mean_abs_disp = np.mean(np.abs(displacement), axis=0)
    top_idx = np.argsort(mean_abs_disp)[-top_k:][::-1]
    
    result = {}
    for rank, idx in enumerate(top_idx):
        result[f'gene_rank_{rank+1}'] = gene_names[idx]
        result[f'disp_rank_{rank+1}'] = float(mean_abs_disp[idx])
    
    return result


# ============================================================================
# Sinkhorn OT Baseline
# ============================================================================
def compute_sinkhorn_ot_baseline(
    source: np.ndarray,
    target: np.ndarray,
    reg: float = 0.1,
) -> np.ndarray:
    """
    Compute standard entropic Sinkhorn OT transport.
    
    Args:
        source: (N, D) source samples
        target: (M, D) target samples
        reg: Entropic regularization parameter
        
    Returns:
        X_transported: (N, D) transported source samples via barycentric projection
    """
    n_source = source.shape[0]
    n_target = target.shape[0]
    
    # Uniform weights
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target
    
    # Cost matrix (squared Euclidean)
    C = ot.dist(source, target, metric='sqeuclidean')
    
    # Sinkhorn transport plan
    P = ot.sinkhorn(a, b, C, reg=reg, numItermax=100)
    
    # Barycentric projection: X_transported = n_source * P @ target
    X_transported = n_source * P @ target
    
    return X_transported


# ============================================================================
# Single Experiment Runner
# ============================================================================
def run_single_experiment(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_test: np.ndarray,
    target_test: np.ndarray,
    gamma: float,
    fold_idx: int,
    gene_names: List[str],
    save_policy: bool = True,
) -> Dict[str, Any]:
    """
    Train Regularized SB solver and evaluate on test data.
    
    Args:
        source_train: Training source samples
        target_train: Training target samples
        source_test: Test source samples
        target_test: Test target samples
        gamma: L1 regularization strength
        fold_idx: Fold index for logging/saving
        gene_names: List of gene names
        save_policy: Whether to save model weights
        
    Returns:
        Dictionary of results
    """
    n_genes = source_train.shape[1]
    n_particles = source_train.shape[0]
    
    # Configure solver
    solver_cfg = SolverConfig(
        d=n_genes,
        n_particles=n_particles,
        lam=gamma,
        eps=EPS,
        u_max=U_MAX,
        T=T_STEPS,
        outer_loops=OUTER_LOOPS,
        value_epochs=5,
        policy_epochs=5,
        seed=SEED,
        target_type="dataset",
    )
    
    exp_cfg = ExperimentConfig(
        solver=solver_cfg,
        value_net=NetworkConfig(name="ValueNet", hidden=HIDDEN_DIM),
        policy_net=NetworkConfig(name="PolicyNet", hidden=HIDDEN_DIM),
        logging=LoggingConfig(
            output_dir=str(OUTPUT_DIR),
            experiment_name=f"gamma_{gamma}",
            save_final=False,
            save_samples=False,
        ),
        dataset=DatasetConfig(),
        penalty=PenaltyConfig(name="mmd", params={"bandwidth": 10.0}),
        metrics=MetricsConfig(enabled=False),
    )
    
    # Train solver
    print(f"    Training Regularized SB (gamma={gamma}, fold={fold_idx})...")
    solver = RegularizedSBSolver(exp_cfg, source_data=source_train, target_data=target_train)
    train_result = solver.run()
    
    # Save policy network weights (atomic write to prevent corruption)
    if save_policy:
        import os
        policy_dir = OUTPUT_DIR / "policies" / f"gamma_{gamma}"
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy_path = policy_dir / f"fold_{fold_idx}_policy.pt"
        temp_path = policy_dir / f"fold_{fold_idx}_policy.pt.tmp"
        
        # Save to temp file first, then flush to disk
        torch.save({
            "policy_state_dict": solver.policy_net.state_dict(),
            "value_state_dict": solver.value_net.state_dict(),
            "config": {
                "d": n_genes,
                "hidden": HIDDEN_DIM,
                "T": T_STEPS,
                "eps": EPS,
                "gamma": gamma,
                "u_max": U_MAX,
            }
        }, temp_path)
        
        # Ensure data is flushed to disk (important for Lustre/network filesystems)
        with open(temp_path, 'rb') as f:
            os.fsync(f.fileno())
        
        # Atomic rename (prevents corruption if job is killed mid-write)
        temp_path.rename(policy_path)
        
        # Verify the saved file is valid
        try:
            _ = torch.load(policy_path, map_location="cpu", weights_only=False)
            print(f"    Saved and verified policy to {policy_path}")
        except Exception as e:
            print(f"    WARNING: Policy save verification failed: {e}")
    
    # Apply learned policy to test data
    u_nom = u_nom_fn(solver.policy_net, solver_cfg.T, solver_cfg.u_max)
    
    # Deterministic rollout (eps=0) for sparsity measurement
    X_transported_det, _ = rollout_bridge(source_test, solver_cfg.T, 0.0, u_nom)
    
    # Stochastic rollout for distribution matching
    X_transported, _ = rollout_bridge(source_test, solver_cfg.T, solver_cfg.eps, u_nom)
    
    # Compute metrics
    sinkhorn = compute_sinkhorn_divergence(X_transported, target_test)
    sparsity_metrics = compute_sparsity_metrics(X_transported_det, source_test, SPARSITY_THRESHOLD)
    top_genes = get_top_genes(X_transported_det, source_test, gene_names, top_k=10)
    
    # Build result entry
    result = {
        "gamma": gamma,
        "fold": fold_idx,
        "sinkhorn": sinkhorn,
        **sparsity_metrics,
        **top_genes,
    }
    
    # Compute Sinkhorn OT baseline if requested
    if COMPUTE_BASELINE:
        print(f"    Computing Sinkhorn OT baseline...")
        X_baseline = compute_sinkhorn_ot_baseline(source_test, target_test, reg=0.1)
        
        baseline_sinkhorn = compute_sinkhorn_divergence(X_baseline, target_test)
        baseline_sparsity = compute_sparsity_metrics(X_baseline, source_test, SPARSITY_THRESHOLD)
        
        result["baseline_sinkhorn"] = baseline_sinkhorn
        result["baseline_sparsity_pct"] = baseline_sparsity["sparsity_pct"]
        result["baseline_l1_total"] = baseline_sparsity["l1_total"]
    
    print(f"    Fold {fold_idx}: Sinkhorn={sinkhorn:.4f}, Sparsity={sparsity_metrics['sparsity_pct']:.1f}%")
    
    return result


# ============================================================================
# Main Experiment
# ============================================================================
def run_gamma_worker(
    rank: int,
    gamma: float,
    source_data: np.ndarray,
    target_data: np.ndarray,
    source_folds: List[np.ndarray],
    target_folds: List[np.ndarray],
    gene_names: np.ndarray,
    output_dir: Path,
    n_folds: int,
    compute_baseline: bool,
    result_queue: mp.Queue,
):
    """
    Worker function for parallel training of a single gamma value.
    Each worker runs all folds for its assigned gamma on a specific GPU.
    """
    # Assign GPU based on rank
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 0:
        gpu_id = rank % n_gpus
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"
    
    print(f"[Worker {rank}] Starting gamma={gamma} on {device}")
    
    gamma_results = []
    
    for fold_idx in range(n_folds):
        print(f"[Worker {rank}] Gamma={gamma}, Fold {fold_idx + 1}/{n_folds}")
        
        # Create train/test split
        test_idx_s = source_folds[fold_idx]
        train_idx_s = np.concatenate([source_folds[k] for k in range(n_folds) if k != fold_idx])
        
        test_idx_t = target_folds[fold_idx]
        train_idx_t = np.concatenate([target_folds[k] for k in range(n_folds) if k != fold_idx])
        
        source_train = source_data[train_idx_s]
        source_test = source_data[test_idx_s]
        target_train = target_data[train_idx_t]
        target_test = target_data[test_idx_t]
        
        print(f"[Worker {rank}]   Train: {len(train_idx_s)} source, {len(train_idx_t)} target")
        print(f"[Worker {rank}]   Test: {len(test_idx_s)} source, {len(test_idx_t)} target")
        
        # Run experiment
        result = run_single_experiment(
            source_train, target_train,
            source_test, target_test,
            gamma, fold_idx, gene_names,
            save_policy=True,
        )
        
        gamma_results.append(result)
    
    # Save intermediate results for this gamma
    gamma_df = pd.DataFrame(gamma_results)
    gamma_df.to_csv(output_dir / f"results_gamma_{gamma}.csv", index=False)
    print(f"[Worker {rank}] Completed gamma={gamma}, saved results")
    
    # Put results in queue
    result_queue.put(gamma_results)


def run_experiment_parallel(
    n_top_genes: int = N_TOP_GENES,
    compute_baseline: bool = COMPUTE_BASELINE,
):
    """
    Main experiment function with k-fold CV and parallel training across gammas.
    Uses multiprocessing to train different gammas on different GPUs.
    """
    global COMPUTE_BASELINE
    COMPUTE_BASELINE = compute_baseline
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    
    # Load and preprocess data
    source_data, target_data, gene_names = load_and_preprocess_data(
        DATA_PATH, DRUG, n_top_genes
    )
    
    n_source = source_data.shape[0]
    n_target = target_data.shape[0]
    n_genes = source_data.shape[1]
    
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"\nExperiment Configuration (Parallel Mode):")
    print(f"  Drug: {DRUG}, Cell Line: {CELL_LINE}")
    print(f"  Source cells: {n_source}, Target cells: {n_target}")
    print(f"  Genes: {n_genes}")
    print(f"  Gammas: {GAMMAS}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Available GPUs: {n_gpus}")
    print(f"  Parallel workers: {len(GAMMAS)}")
    
    # Create k-fold splits using simple permutation
    source_indices = rng.permutation(n_source)
    target_indices = rng.permutation(n_target)
    
    source_fold_size = n_source // N_FOLDS
    target_fold_size = n_target // N_FOLDS
    
    source_folds = []
    target_folds = []
    for k in range(N_FOLDS):
        s_start = k * source_fold_size
        s_end = s_start + source_fold_size if k < N_FOLDS - 1 else n_source
        source_folds.append(source_indices[s_start:s_end])
        
        t_start = k * target_fold_size
        t_end = t_start + target_fold_size if k < N_FOLDS - 1 else n_target
        target_folds.append(target_indices[t_start:t_end])
    
    print(f"\nFold sizes:")
    for k in range(N_FOLDS):
        print(f"  Fold {k}: {len(source_folds[k])} source, {len(target_folds[k])} target")
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    
    # Spawn workers for each gamma
    processes = []
    for rank, gamma in enumerate(GAMMAS):
        p = mp.Process(
            target=run_gamma_worker,
            args=(
                rank, gamma, source_data, target_data,
                source_folds, target_folds, gene_names,
                OUTPUT_DIR, N_FOLDS, compute_baseline, result_queue
            )
        )
        p.start()
        processes.append(p)
        print(f"Started worker {rank} for gamma={gamma}")
    
    # Collect results
    all_results = []
    for _ in range(len(GAMMAS)):
        gamma_results = result_queue.get()
        all_results.extend(gamma_results)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("\nAll workers completed!")
    
    # Save all results
    df = pd.DataFrame(all_results)
    df = df.sort_values(['gamma', 'fold']).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.csv'}")
    
    # Save as JSON too
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary statistics
    summary_cols = ['sinkhorn', 'sparsity_pct', 'l1_total']
    if COMPUTE_BASELINE:
        summary_cols += ['baseline_sinkhorn', 'baseline_sparsity_pct']
    
    summary = df.groupby('gamma')[summary_cols].agg(['mean', 'std'])
    summary.to_csv(OUTPUT_DIR / "summary_stats.csv")
    print(f"Summary saved to {OUTPUT_DIR / 'summary_stats.csv'}")
    
    # Save experiment configuration
    experiment_config = {
        "drug": DRUG,
        "cell_line": CELL_LINE,
        "gammas": GAMMAS,
        "n_top_genes": n_top_genes,
        "n_folds": N_FOLDS,
        "sparsity_threshold": SPARSITY_THRESHOLD,
        "t_steps": T_STEPS,
        "outer_loops": OUTER_LOOPS,
        "eps": EPS,
        "u_max": U_MAX,
        "hidden_dim": HIDDEN_DIM,
        "seed": SEED,
        "compute_baseline": COMPUTE_BASELINE,
        "n_source_cells": n_source,
        "n_target_cells": n_target,
        "n_genes": n_genes,
        "parallel": True,
        "n_gpus": n_gpus,
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    print(f"Config saved to {OUTPUT_DIR / 'experiment_config.json'}")
    
    # Generate plots
    plot_results(df)
    
    return df


def run_experiment(
    n_top_genes: int = N_TOP_GENES,
    compute_baseline: bool = COMPUTE_BASELINE,
):
    """
    Main experiment function with k-fold cross-validation.
    """
    global COMPUTE_BASELINE
    COMPUTE_BASELINE = compute_baseline
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    
    # Load and preprocess data
    source_data, target_data, gene_names = load_and_preprocess_data(
        DATA_PATH, DRUG, n_top_genes
    )
    
    n_source = source_data.shape[0]
    n_target = target_data.shape[0]
    n_genes = source_data.shape[1]
    
    print(f"\nExperiment Configuration:")
    print(f"  Drug: {DRUG}, Cell Line: {CELL_LINE}")
    print(f"  Source cells: {n_source}, Target cells: {n_target}")
    print(f"  Genes: {n_genes}")
    print(f"  Gammas: {GAMMAS}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Device: {DEVICE}")
    
    # Create k-fold splits using simple permutation
    source_indices = rng.permutation(n_source)
    target_indices = rng.permutation(n_target)
    
    source_fold_size = n_source // N_FOLDS
    target_fold_size = n_target // N_FOLDS
    
    source_folds = []
    target_folds = []
    for k in range(N_FOLDS):
        s_start = k * source_fold_size
        s_end = s_start + source_fold_size if k < N_FOLDS - 1 else n_source
        source_folds.append(source_indices[s_start:s_end])
        
        t_start = k * target_fold_size
        t_end = t_start + target_fold_size if k < N_FOLDS - 1 else n_target
        target_folds.append(target_indices[t_start:t_end])
    
    print(f"\nFold sizes:")
    for k in range(N_FOLDS):
        print(f"  Fold {k}: {len(source_folds[k])} source, {len(target_folds[k])} target")
    
    # Run experiments
    all_results = []
    
    for gamma in GAMMAS:
        print(f"\n{'='*60}")
        print(f"Gamma = {gamma}")
        print(f"{'='*60}")
        
        for fold_idx in range(N_FOLDS):
            print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS} ---")
            
            # Create train/test split
            test_idx_s = source_folds[fold_idx]
            train_idx_s = np.concatenate([source_folds[k] for k in range(N_FOLDS) if k != fold_idx])
            
            test_idx_t = target_folds[fold_idx]
            train_idx_t = np.concatenate([target_folds[k] for k in range(N_FOLDS) if k != fold_idx])
            
            source_train = source_data[train_idx_s]
            source_test = source_data[test_idx_s]
            target_train = target_data[train_idx_t]
            target_test = target_data[test_idx_t]
            
            print(f"    Train: {len(train_idx_s)} source, {len(train_idx_t)} target")
            print(f"    Test: {len(test_idx_s)} source, {len(test_idx_t)} target")
            
            # Run experiment
            result = run_single_experiment(
                source_train, target_train,
                source_test, target_test,
                gamma, fold_idx, gene_names,
                save_policy=True,
            )
            
            all_results.append(result)
        
        # Save intermediate results for this gamma
        gamma_df = pd.DataFrame([r for r in all_results if r['gamma'] == gamma])
        gamma_df.to_csv(OUTPUT_DIR / f"results_gamma_{gamma}.csv", index=False)
    
    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.csv'}")
    
    # Save as JSON too
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary statistics
    summary_cols = ['sinkhorn', 'sparsity_pct', 'l1_total']
    if COMPUTE_BASELINE:
        summary_cols += ['baseline_sinkhorn', 'baseline_sparsity_pct']
    
    summary = df.groupby('gamma')[summary_cols].agg(['mean', 'std'])
    summary.to_csv(OUTPUT_DIR / "summary_stats.csv")
    print(f"Summary saved to {OUTPUT_DIR / 'summary_stats.csv'}")
    
    # Save experiment configuration
    experiment_config = {
        "drug": DRUG,
        "cell_line": CELL_LINE,
        "gammas": GAMMAS,
        "n_top_genes": n_top_genes,
        "n_folds": N_FOLDS,
        "sparsity_threshold": SPARSITY_THRESHOLD,
        "t_steps": T_STEPS,
        "outer_loops": OUTER_LOOPS,
        "eps": EPS,
        "u_max": U_MAX,
        "hidden_dim": HIDDEN_DIM,
        "seed": SEED,
        "compute_baseline": COMPUTE_BASELINE,
        "n_source_cells": n_source,
        "n_target_cells": n_target,
        "n_genes": n_genes,
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    print(f"Config saved to {OUTPUT_DIR / 'experiment_config.json'}")
    
    # Generate plots
    plot_results(df)
    
    return df


# ============================================================================
# Plotting
# ============================================================================
def plot_results(df: pd.DataFrame):
    """Generate Figure 7-style plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get unique gammas
    gammas = sorted(df['gamma'].unique())
    
    # ---- Plot 1: Sinkhorn Divergence ----
    ax = axes[0]
    
    # Scatter individual fold results
    for gamma in gammas:
        subset = df[df['gamma'] == gamma]
        ax.scatter([gamma] * len(subset), subset['sinkhorn'], 
                   alpha=0.5, color='blue', s=50)
    
    # Plot mean line
    means = df.groupby('gamma')['sinkhorn'].mean()
    ax.plot(means.index, means.values, 'x-', color='navy', 
            markersize=12, linewidth=2, label='Regularized SB')
    
    # Plot baseline if available
    if 'baseline_sinkhorn' in df.columns:
        baseline_means = df.groupby('gamma')['baseline_sinkhorn'].mean()
        ax.axhline(baseline_means.mean(), linestyle='--', color='red', 
                   linewidth=2, label='Sinkhorn OT Baseline')
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Regularization (γ)', fontsize=12)
    ax.set_ylabel('Sinkhorn Divergence', fontsize=12)
    ax.set_title(f'{DRUG}, {CELL_LINE}', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    
    # ---- Plot 2: Sparsity ----
    ax = axes[1]
    
    # Scatter individual fold results
    for gamma in gammas:
        subset = df[df['gamma'] == gamma]
        ax.scatter([gamma] * len(subset), subset['sparsity_pct'], 
                   alpha=0.5, color='blue', s=50)
    
    # Plot mean line
    means_sp = df.groupby('gamma')['sparsity_pct'].mean()
    ax.plot(means_sp.index, means_sp.values, 'x-', color='navy', 
            markersize=12, linewidth=2, label='Regularized SB')
    
    # Plot baseline if available
    if 'baseline_sparsity_pct' in df.columns:
        baseline_sp = df.groupby('gamma')['baseline_sparsity_pct'].mean()
        ax.axhline(baseline_sp.mean(), linestyle='--', color='red', 
                   linewidth=2, label='Sinkhorn OT Baseline')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Regularization (γ)', fontsize=12)
    ax.set_ylabel('% Non-zero Genes', fontsize=12)
    ax.set_title(f'{DRUG}, {CELL_LINE}', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(OUTPUT_DIR / "figure7_replication.png", dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure7_replication.pdf", bbox_inches='tight')
    print(f"Plots saved to {OUTPUT_DIR / 'figure7_replication.png'}")
    
    plt.close()
    
    # ---- Additional Plot: L1 Displacement ----
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for gamma in gammas:
        subset = df[df['gamma'] == gamma]
        ax.scatter([gamma] * len(subset), subset['l1_total'], 
                   alpha=0.5, color='blue', s=50)
    
    means_l1 = df.groupby('gamma')['l1_total'].mean()
    ax.plot(means_l1.index, means_l1.values, 'x-', color='navy', 
            markersize=12, linewidth=2, label='Regularized SB')
    
    if 'baseline_l1_total' in df.columns:
        baseline_l1 = df.groupby('gamma')['baseline_l1_total'].mean()
        ax.axhline(baseline_l1.mean(), linestyle='--', color='red', 
                   linewidth=2, label='Sinkhorn OT Baseline')
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Regularization (γ)', fontsize=12)
    ax.set_ylabel('L1 Displacement', fontsize=12)
    ax.set_title(f'Total Displacement - {DRUG}, {CELL_LINE}', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "l1_displacement.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Replicate Figure 7 with Regularized SB')
    parser.add_argument('--sequential', action='store_true', 
                        help='Run sequentially instead of parallel')
    parser.add_argument('--n-genes', type=int, default=N_TOP_GENES,
                        help='Number of top HVGs to use')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip computing Sinkhorn OT baseline')
    args = parser.parse_args()
    
    if args.sequential:
        print("Running in SEQUENTIAL mode")
        run_experiment(n_top_genes=args.n_genes, compute_baseline=not args.no_baseline)
    else:
        print("Running in PARALLEL mode")
        run_experiment_parallel(n_top_genes=args.n_genes, compute_baseline=not args.no_baseline)
