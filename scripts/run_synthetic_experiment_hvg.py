"""
Synthetic Perturbation Experiment for Regularized Schrödinger Bridge

This script:
1. Loads real scRNA-seq data (sciPlex3 dataset)
2. Selects top 2000 highly variable genes (HVGs)
3. Creates a synthetic perturbation by shifting a sparse subset of genes
4. Trains Regularized SB with varying lambda (L1 regularization) values
5. Evaluates transport quality via Sinkhorn divergence
6. Plots regularization vs. Sinkhorn divergence trade-off

The goal is to demonstrate that with appropriate regularization (lam), the SB solver
can recover sparse transport maps that match the ground-truth sparse perturbation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from geomloss import SamplesLoss

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.regularizedSB.solver import RegularizedSBSolver
from src.regularizedSB.config import (
    ExperimentConfig, SolverConfig, NetworkConfig, 
    LoggingConfig, DatasetConfig, PenaltyConfig, MetricsConfig
)
from src.regularizedSB.metrics import u_nom_fn, rollout_bridge

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = PROJECT_ROOT / "sc_data" / "sciPlex3_subset_10k.h5ad"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "synthetic_experiment_givino"

# Regularization values to sweep (powers of 2 for log-scale plotting)
LAMBDAS = [1.0, 16.0, 64.0, 256.0, 1024.0] #keep small for test run

N_TOP_GENES = 1024  #Number of highly variable genes to select
N_PERTURBED_GENES = 40  #approx 40 genes have >1 LFC per rudimentary DE analysis
PERTURBATION_SCALE = 1.5 # Magnitude of the synthetic perturbation

# Solver parameters
N_FOLDS = 3  # Number of cross-validation folds
OUTER_LOOPS = 30  # Training iterations (increase for better convergence)
T_STEPS = 25  # Number of time discretization steps
EPS = 0.1 #was 0.1  # Diffusion coefficient
U_MAX = 5.0  # Box constraint on control
HIDDEN_DIM = 256  # Hidden layer size for networks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def load_and_preprocess_data(
    data_path: Path,
    n_top_genes: int = 2000,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load scRNA-seq data and select top highly variable genes.
    
    Args:
        data_path: Path to .h5ad file
        n_top_genes: Number of HVGs to select
        
    Returns:
        data: (n_cells, n_top_genes) numpy array of normalized expression
        gene_names: List of selected gene names
    """
    print(f"Loading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    print(f"  Raw data: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Standard preprocessing
    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes
    print(f"Selecting top {n_top_genes} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    adata = adata[:, adata.var['highly_variable']].copy()
    print(f"  After HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Extract vehicle (control) cells as our source distribution
    vehicle_mask = (
        (adata.obs['vehicle'] == True) | 
        (adata.obs['product_name'].str.contains('Vehicle', case=False, na=False))
    )
    source_adata = adata[vehicle_mask].copy()
    print(f"  Vehicle cells (source): {source_adata.n_obs}")
    
    # Convert to dense array
    data = source_adata.X
    if hasattr(data, 'toarray'):
        data = data.toarray()
    data = np.asarray(data, dtype=np.float64)
    
    gene_names = list(source_adata.var_names)
    
    return data, gene_names


def create_synthetic_perturbation(
    source_data: np.ndarray,
    n_perturbed: int,
    perturbation_scale: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Create a synthetic perturbation by shifting a sparse subset of genes.
    
    This simulates a sparse drug effect where only a few genes are affected.
    
    Args:
        source_data: (n_cells, n_genes) source expression data
        n_perturbed: Number of genes to perturb
        perturbation_scale: Magnitude of perturbation (in log-space)
        rng: Random number generator
        
    Returns:
        target_data: Perturbed expression data
        perturbation_vector: (n_genes,) vector of perturbation magnitudes
        perturbed_indices: List of perturbed gene indices
    """
    n_cells, n_genes = source_data.shape
    
    # Select random genes to perturb
    perturbed_indices = rng.choice(n_genes, size=n_perturbed, replace=False).tolist()
    
    # Create perturbation vector (sparse)
    perturbation_vector = np.zeros(n_genes, dtype=np.float64)
    
    # Assign random signed perturbations
    signs = rng.choice([-1.0, 1.0], size=n_perturbed)
    perturbation_vector[perturbed_indices] = perturbation_scale * signs
    
    # Apply perturbation (additive in log-space)
    target_data = source_data + perturbation_vector[None, :]
    
    # Add small noise to make cells distinct
    noise = rng.normal(0, 0.1, size=target_data.shape)
    target_data = target_data + noise
    
    print(f"Created synthetic perturbation:")
    print(f"  Perturbed genes: {n_perturbed} / {n_genes}")
    print(f"  Perturbation scale: {perturbation_scale}")
    print(f"  Perturbed indices: {sorted(perturbed_indices)[:10]}...")
    
    return target_data, perturbation_vector, perturbed_indices


# ============================================================================
# Metrics
# ============================================================================
def compute_sinkhorn_divergence(
    X_transported: np.ndarray,
    X_target: np.ndarray,
    blur: float = 0.1,
) -> float:
    xt = torch.from_numpy(X_transported.astype(np.float32)).to(DEVICE)
    yt = torch.from_numpy(X_target.astype(np.float32)).to(DEVICE)
    
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True)
    
    with torch.no_grad():
        divergence = loss_fn(xt, yt)
    
    return float(divergence.item())


def compute_sparsity_metrics(
    X_transported: np.ndarray,
    X_source: np.ndarray,
    perturbed_indices: List[int],
    threshold: float = 1e-2,
) -> Dict[str, float]:
    """
    Compute sparsity-related metrics for the transport map.
    
    Args:
        X_transported: Transported samples
        X_source: Original source samples  
        perturbed_indices: Ground-truth perturbed gene indices
        threshold: Threshold for considering displacement as "non-zero"
        
    Returns:
        Dictionary of sparsity metrics
    """
    n_genes = X_transported.shape[1]
    displacement = X_transported - X_source
    mean_displacement = np.mean(displacement, axis=0)
    
    # All dimensions
    l1_total = float(np.sum(np.abs(mean_displacement)))
    l2_total = float(np.linalg.norm(mean_displacement))
    
    # Nuisance dimensions (should ideally be zero)
    nuisance_indices = [i for i in range(n_genes) if i not in perturbed_indices]
    l1_nuisance = float(np.sum(np.abs(mean_displacement[nuisance_indices])))
    
    # Active dimensions
    l1_active = float(np.sum(np.abs(mean_displacement[perturbed_indices])))
    
    # Sparsity: % of dimensions with displacement above threshold
    n_nonzero = np.sum(np.abs(mean_displacement) > threshold)
    sparsity_pct = 100.0 * n_nonzero / n_genes
    
    # Per-sample sparsity
    sample_nonzero = np.sum(np.abs(displacement) > threshold, axis=1)
    mean_sample_sparsity = 100.0 * np.mean(sample_nonzero) / n_genes
    
    return {
        "l1_total": l1_total,
        "l2_total": l2_total,
        "l1_nuisance": l1_nuisance,
        "l1_active": l1_active,
        "sparsity_pct": sparsity_pct,
        "mean_sample_sparsity": mean_sample_sparsity,
        "nuisance_ratio": l1_nuisance / (l1_total + 1e-8),
    }


def compute_perturbation_recovery(
    X_transported: np.ndarray,
    X_source: np.ndarray,
    ground_truth_perturbation: np.ndarray,
) -> Dict[str, float]:
    """
    Measure how well the transport map recovers the ground-truth perturbation.
    """
    mean_displacement = np.mean(X_transported - X_source, axis=0)
    
    # Cosine similarity between learned displacement and ground truth
    cos_sim = np.dot(mean_displacement, ground_truth_perturbation) / (
        np.linalg.norm(mean_displacement) * np.linalg.norm(ground_truth_perturbation) + 1e-8
    )
    
    # L2 error
    l2_error = np.linalg.norm(mean_displacement - ground_truth_perturbation)
    
    # Relative L2 error
    rel_l2_error = l2_error / (np.linalg.norm(ground_truth_perturbation) + 1e-8)
    
    return {
        "cosine_similarity": float(cos_sim),
        "l2_error": float(l2_error),
        "relative_l2_error": float(rel_l2_error),
    }


# ============================================================================
# Experiment Runner
# ============================================================================
def run_single_experiment(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_test: np.ndarray,
    target_test: np.ndarray,
    lam: float,
    fold_idx: int,
    perturbed_indices: List[int],
    ground_truth_perturbation: np.ndarray,
    save_policy: bool = True,
) -> Dict[str, Any]:
    n_genes = source_train.shape[1]
    n_particles = source_train.shape[0]
    
    # Configure solver
    solver_cfg = SolverConfig(
        d=n_genes,
        n_particles=n_particles,
        eps=EPS,
        lam=lam,
        u_max=U_MAX,
        T=T_STEPS,
        outer_loops=OUTER_LOOPS,
        beta_eff=600.0,
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
            experiment_name=f"lam_{lam}",
            save_final=False,
            save_samples=False,
        ),
        dataset=DatasetConfig(),
        penalty=PenaltyConfig(name="quadratic"),
        metrics=MetricsConfig(enabled=False),
    )
    
    # Train solver
    solver = RegularizedSBSolver(exp_cfg, source_data=source_train, target_data=target_train)
    train_result = solver.run()
    
    # Extract training logs (list of (total, quad, l1, term) per outer loop)
    training_logs = train_result.get("training_logs", [])
    
    # Save policy network weights
    if save_policy:
        policy_dir = OUTPUT_DIR / "policies" / f"lam_{lam}"
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy_path = policy_dir / f"fold_{fold_idx}_policy.pt"
        torch.save({
            "policy_state_dict": solver.policy_net.state_dict(),
            "value_state_dict": solver.value_net.state_dict(),
            "config": {
                "d": n_genes,
                "hidden": HIDDEN_DIM,
                "T": T_STEPS,
                "eps": EPS,
                "lam": lam,
                "u_max": U_MAX,
            }
        }, policy_path)
        print(f"    Saved policy to {policy_path}")
    
    # Apply learned policy to test data
    u_nom = u_nom_fn(solver.policy_net, solver_cfg.T, solver_cfg.u_max)
    X_transported, _ = rollout_bridge(source_test, solver_cfg.T, solver_cfg.eps, u_nom)
    
    # Compute metrics
    sinkhorn = compute_sinkhorn_divergence(X_transported, target_test)
    sparsity = compute_sparsity_metrics(X_transported, source_test, perturbed_indices)
    recovery = compute_perturbation_recovery(X_transported, source_test, ground_truth_perturbation)
    
    return {
        "lambda": lam,
        "sinkhorn": sinkhorn,
        "training_logs": training_logs,
        **sparsity,
        **recovery,
    }


def run_experiment():
    """
    Main experiment function with k-fold cross-validation.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    
    # Load and preprocess data
    source_data, gene_names = load_and_preprocess_data(DATA_PATH, N_TOP_GENES)
    n_cells, n_genes = source_data.shape
    print(f"\nWorking with {n_cells} cells x {n_genes} genes")
    
    # Create synthetic perturbation
    target_data, perturbation_vector, perturbed_indices = create_synthetic_perturbation(
        source_data, N_PERTURBED_GENES, PERTURBATION_SCALE, rng
    )
    
    # Save perturbation info
    perturbation_info = {
        "n_genes": n_genes,
        "n_perturbed": N_PERTURBED_GENES,
        "perturbation_scale": PERTURBATION_SCALE,
        "perturbed_indices": perturbed_indices,
        "perturbed_genes": [gene_names[i] for i in perturbed_indices],
        "perturbation_vector": perturbation_vector.tolist(),
        "n_folds": N_FOLDS,
    }
    with open(OUTPUT_DIR / "perturbation_info.json", "w") as f:
        json.dump(perturbation_info, f, indent=2)
    
    # Create k-fold indices
    indices = rng.permutation(n_cells)
    fold_size = n_cells // N_FOLDS
    folds = []
    for k in range(N_FOLDS):
        start = k * fold_size
        end = start + fold_size if k < N_FOLDS - 1 else n_cells
        folds.append(indices[start:end])
    
    print(f"Using {N_FOLDS}-fold cross-validation")
    for k, fold in enumerate(folds):
        print(f"  Fold {k+1}: {len(fold)} cells")
    
    # Run experiments with k-fold CV
    results = []
    all_training_logs = {}  # Store training logs for each lambda
    
    for lam in LAMBDAS:
        print(f"\n{'='*60}")
        print(f"Running experiment for lambda = {lam}")
        print(f"{'='*60}")
        
        fold_results = []
        fold_training_logs = []
        
        for fold_idx in range(N_FOLDS):
            print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS} ---")
            
            # Create train/test split for this fold
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[k] for k in range(N_FOLDS) if k != fold_idx])
            
            source_train = source_data[train_idx]
            source_test = source_data[test_idx]
            target_train = target_data[train_idx]
            target_test = target_data[test_idx]
            
            print(f"  Train: {len(train_idx)} cells, Test: {len(test_idx)} cells")
            
            # Run experiment for this fold
            result = run_single_experiment(
                source_train, target_train,
                source_test, target_test,
                lam, fold_idx, perturbed_indices, perturbation_vector,
            )
            
            # Extract training logs for this fold
            training_logs = result.pop("training_logs", [])
            fold_training_logs.append(training_logs)
            fold_results.append(result)
            
            print(f"  Fold {fold_idx + 1} - Sinkhorn: {result['sinkhorn']:.4f}, "
                  f"Sparsity: {result['sparsity_pct']:.1f}%, "
                  f"Nuisance L1: {result['l1_nuisance']:.4f}")
        
        # Average results across folds
        avg_result = {"lambda": lam}
        metric_keys = [k for k in fold_results[0].keys() if k != "lambda"]
        
        for key in metric_keys:
            values = [r[key] for r in fold_results]
            avg_result[key] = float(np.mean(values))
            avg_result[f"{key}_std"] = float(np.std(values))
        
        results.append(avg_result)
        
        # Average training logs across folds (element-wise average of tuples)
        if fold_training_logs and all(len(logs) > 0 for logs in fold_training_logs):
            n_iters = min(len(logs) for logs in fold_training_logs)
            avg_logs = []
            for i in range(n_iters):
                avg_tuple = tuple(
                    np.mean([fold_training_logs[f][i][j] for f in range(N_FOLDS)])
                    for j in range(4)  # (total, quad, l1, term)
                )
                avg_logs.append(avg_tuple)
            all_training_logs[lam] = avg_logs
        else:
            all_training_logs[lam] = []
        
        print(f"\n  λ={lam} Average: Sinkhorn={avg_result['sinkhorn']:.4f} ± {avg_result['sinkhorn_std']:.4f}, "
              f"Sparsity={avg_result['sparsity_pct']:.1f}% ± {avg_result['sparsity_pct_std']:.1f}%")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.csv'}")
    
    # Save full results as JSON
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save training logs
    training_logs_serializable = {str(k): v for k, v in all_training_logs.items()}
    with open(OUTPUT_DIR / "training_logs.json", "w") as f:
        json.dump(training_logs_serializable, f, indent=2)
    print(f"Training logs saved to {OUTPUT_DIR / 'training_logs.json'}")
    
    # Save experiment configuration for reproducibility
    experiment_config = {
        "lambdas": LAMBDAS,
        "n_top_genes": N_TOP_GENES,
        "n_perturbed_genes": N_PERTURBED_GENES,
        "perturbation_scale": PERTURBATION_SCALE,
        "n_folds": N_FOLDS,
        "outer_loops": OUTER_LOOPS,
        "t_steps": T_STEPS,
        "eps": EPS,
        "u_max": U_MAX,
        "hidden_dim": HIDDEN_DIM,
        "seed": SEED,
        "beta_eff": 600.0,
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    print(f"Experiment config saved to {OUTPUT_DIR / 'experiment_config.json'}")
    
    # Save source and target data for plotting/analysis
    np.savez(
        OUTPUT_DIR / "data.npz",
        source_data=source_data,
        target_data=target_data,
        perturbation_vector=perturbation_vector,
        perturbed_indices=np.array(perturbed_indices),
        gene_names=np.array(gene_names),
    )
    print(f"Data saved to {OUTPUT_DIR / 'data.npz'}")
    
    # Generate plots
    plot_results(df, perturbation_info, all_training_logs)
    
    return df


'''
def plot_training_curves(all_training_logs: Dict[float, List[Tuple]], output_dir: Path):
    """
    Plot training curves for each lambda value, similar to toyExample.ipynb.
    
    Each training log entry is a tuple: (total, quad, l1, term)
    - total: Total cost = quad + l1 + term
    - quad: Quadratic control cost (||u||²/2ε integrated over time)
    - l1: L1 sparsity penalty (λ||u||₁ integrated over time)
    - term: Terminal cost (β × g)
    """
    n_lambdas = len(all_training_logs)
    if n_lambdas == 0:
        print("No training logs to plot.")
        return
    
    # Create subplots: one column per lambda
    n_cols = min(n_lambdas, 4)
    n_rows = (n_lambdas + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, (lam, logs) in enumerate(sorted(all_training_logs.items())):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        if len(logs) == 0:
            ax.text(0.5, 0.5, "No training data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"λ = {lam}")
            continue
        
        # Extract components from logs
        iterations = range(1, len(logs) + 1)
        total_costs = [v[0] for v in logs]
        quad_costs = [v[1] for v in logs]
        l1_costs = [v[2] for v in logs]
        term_costs = [v[3] for v in logs]
        
        # Plot each component
        ax.plot(iterations, total_costs, label="total", linewidth=2, color='black')
        ax.plot(iterations, quad_costs, label="quad (||u||²/2ε)", linewidth=1.5, color='blue')
        ax.plot(iterations, l1_costs, label=f"L1 (λ||u||₁)", linewidth=1.5, color='orange')
        ax.plot(iterations, term_costs, label="terminal (βg)", linewidth=1.5, color='green')
        
        ax.set_xlabel("Outer Iteration", fontsize=10)
        ax.set_ylabel("Cost", fontsize=10)
        ax.set_title(f"Training Curves (λ = {lam})", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc='best', fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(all_training_logs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "training_curves.pdf", bbox_inches="tight")
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()
    
    # Also create a combined plot showing all lambdas together
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_training_logs)))
    
    for idx, (lam, logs) in enumerate(sorted(all_training_logs.items())):
        if len(logs) == 0:
            continue
        
        iterations = range(1, len(logs) + 1)
        total_costs = [v[0] for v in logs]
        quad_costs = [v[1] for v in logs]
        l1_costs = [v[2] for v in logs]
        term_costs = [v[3] for v in logs]
        
        color = colors[idx]
        
        axes[0, 0].plot(iterations, total_costs, label=f"λ={lam}", linewidth=2, color=color)
        axes[0, 1].plot(iterations, quad_costs, label=f"λ={lam}", linewidth=2, color=color)
        axes[1, 0].plot(iterations, l1_costs, label=f"λ={lam}", linewidth=2, color=color)
        axes[1, 1].plot(iterations, term_costs, label=f"λ={lam}", linewidth=2, color=color)
    
    axes[0, 0].set_title("Total Cost", fontsize=14)
    axes[0, 0].set_xlabel("Outer Iteration")
    axes[0, 0].set_ylabel("Cost")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend()
    
    axes[0, 1].set_title("Quadratic Control Cost (||u||²/2ε)", fontsize=14)
    axes[0, 1].set_xlabel("Outer Iteration")
    axes[0, 1].set_ylabel("Cost")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend()
    
    axes[1, 0].set_title("L1 Sparsity Penalty (λ||u||₁)", fontsize=14)
    axes[1, 0].set_xlabel("Outer Iteration")
    axes[1, 0].set_ylabel("Cost")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend()
    
    axes[1, 1].set_title("Terminal Cost (βg)", fontsize=14)
    axes[1, 1].set_xlabel("Outer Iteration")
    axes[1, 1].set_ylabel("Cost")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_combined.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "training_curves_combined.pdf", bbox_inches="tight")
    print(f"Combined training curves saved to {output_dir / 'training_curves_combined.png'}")
    plt.close()
    
    # ---- LOG SCALE training curves ----
    # Per-lambda log-scale plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, (lam, logs) in enumerate(sorted(all_training_logs.items())):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        
        if len(logs) == 0:
            ax.text(0.5, 0.5, "No training data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"λ = {lam} (log scale)")
            continue
        
        iterations = range(1, len(logs) + 1)
        total_costs = [max(v[0], 1e-10) for v in logs]  # Avoid log(0)
        quad_costs = [max(v[1], 1e-10) for v in logs]
        l1_costs = [max(v[2], 1e-10) for v in logs]
        term_costs = [max(v[3], 1e-10) for v in logs]
        
        ax.semilogy(iterations, total_costs, label="total", linewidth=2, color='black')
        ax.semilogy(iterations, quad_costs, label="quad", linewidth=1.5, color='blue')
        ax.semilogy(iterations, l1_costs, label="L1", linewidth=1.5, color='orange')
        ax.semilogy(iterations, term_costs, label="terminal", linewidth=1.5, color='green')
        
        ax.set_xlabel("Outer Iteration", fontsize=10)
        ax.set_ylabel("Cost (log scale)", fontsize=10)
        ax.set_title(f"Training Curves (λ = {lam}) - Log Scale", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4, which='both')
        ax.legend(loc='best', fontsize=8)
    
    for idx in range(len(all_training_logs), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_log.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "training_curves_log.pdf", bbox_inches="tight")
    print(f"Log-scale training curves saved to {output_dir / 'training_curves_log.png'}")
    plt.close()
    
    # Combined log-scale plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (lam, logs) in enumerate(sorted(all_training_logs.items())):
        if len(logs) == 0:
            continue
        
        iterations = range(1, len(logs) + 1)
        total_costs = [max(v[0], 1e-10) for v in logs]
        quad_costs = [max(v[1], 1e-10) for v in logs]
        l1_costs = [max(v[2], 1e-10) for v in logs]
        term_costs = [max(v[3], 1e-10) for v in logs]
        
        color = colors[idx]
        
        axes[0, 0].semilogy(iterations, total_costs, label=f"λ={lam}", linewidth=2, color=color)
        axes[0, 1].semilogy(iterations, quad_costs, label=f"λ={lam}", linewidth=2, color=color)
        axes[1, 0].semilogy(iterations, l1_costs, label=f"λ={lam}", linewidth=2, color=color)
        axes[1, 1].semilogy(iterations, term_costs, label=f"λ={lam}", linewidth=2, color=color)
    
    axes[0, 0].set_title("Total Cost (log)", fontsize=14)
    axes[0, 0].set_xlabel("Outer Iteration")
    axes[0, 0].set_ylabel("Cost")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4, which='both')
    axes[0, 0].legend()
    
    axes[0, 1].set_title("Quadratic Control Cost (log)", fontsize=14)
    axes[0, 1].set_xlabel("Outer Iteration")
    axes[0, 1].set_ylabel("Cost")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4, which='both')
    axes[0, 1].legend()
    
    axes[1, 0].set_title("L1 Sparsity Penalty (log)", fontsize=14)
    axes[1, 0].set_xlabel("Outer Iteration")
    axes[1, 0].set_ylabel("Cost")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4, which='both')
    axes[1, 0].legend()
    
    axes[1, 1].set_title("Terminal Cost (log)", fontsize=14)
    axes[1, 1].set_xlabel("Outer Iteration")
    axes[1, 1].set_ylabel("Cost")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4, which='both')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_combined_log.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "training_curves_combined_log.pdf", bbox_inches="tight")
    print(f"Combined log-scale training curves saved to {output_dir / 'training_curves_combined_log.png'}")
    plt.close()

'''
def plot_results(df: pd.DataFrame, perturbation_info: Dict[str, Any], 
                 all_training_logs: Optional[Dict[float, List]] = None):
    # First, plot training curves if available
    #if all_training_logs:
    #    plot_training_curves(all_training_logs, OUTPUT_DIR)
    
    lambdas = df["lambda"].values
    sinkhorn_vals = df["sinkhorn"].values
    sparsity_vals = df["sparsity_pct"].values
    l1_nuisance_vals = df["l1_nuisance"].values
    l1_active_vals = df["l1_active"].values
    cosine_vals = df["cosine_similarity"].values
    rel_l2_vals = df["relative_l2_error"].values
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ---- Plot 1: Sinkhorn Divergence vs Lambda ----
    ax = axes[0, 0]
    ax.plot(lambdas, sinkhorn_vals, "o-", color="navy", linewidth=2, markersize=8)
    
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Regularization (λ)", fontsize=12)
    ax.set_ylabel("Sinkhorn Divergence", fontsize=12)
    ax.set_title(f"Transport Fidelity vs Regularization ({N_FOLDS}-fold CV)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # ---- Plot 2: Sparsity vs Lambda ----
    ax = axes[0, 1]
    ax.plot(lambdas, sparsity_vals, "s-", color="darkorange", linewidth=2, markersize=8)
    
    # Reference line for ground-truth sparsity
    gt_sparsity = 100.0 * perturbation_info["n_perturbed"] / perturbation_info["n_genes"]
    ax.axhline(gt_sparsity, linestyle="--", color="green", linewidth=2, 
               label=f"Ground Truth ({gt_sparsity:.1f}%)")
    
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Regularization (λ)", fontsize=12)
    ax.set_ylabel("% Non-zero Displacement", fontsize=12)
    ax.set_title(f"Transport Sparsity vs Regularization ({N_FOLDS}-fold CV)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    # ---- Plot 3: Nuisance L1 vs Lambda ----
    ax = axes[1, 0]
    ax.plot(lambdas, l1_nuisance_vals, "^-", color="darkred", linewidth=2, markersize=8, label="Nuisance Dims")
    ax.plot(lambdas, l1_active_vals, "v-", color="darkgreen", linewidth=2, markersize=8, label="Active Dims")
    
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("log")
    ax.set_xlabel("Regularization (λ)", fontsize=12)
    ax.set_ylabel("L1 Norm of Mean Displacement", fontsize=12)
    ax.set_title(f"Active vs Nuisance Gene Displacement ({N_FOLDS}-fold CV)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    # ---- Plot 4: Trade-off (Sinkhorn vs Sparsity) ----
    ax = axes[1, 1]
    
    sc = ax.scatter(
        sinkhorn_vals, sparsity_vals,
        c=np.log10(np.array(lambdas) + 1e-3), cmap="viridis", s=100, edgecolors="black"
    )
    
    # Add lambda annotations
    for i, lam in enumerate(lambdas):
        ax.annotate(
            f"λ={lam}", (sinkhorn_vals[i], sparsity_vals[i]),
            xytext=(8, 5), textcoords="offset points", fontsize=9
        )
    
    # Connect points with a line to show the Pareto frontier
    ax.plot(sinkhorn_vals, sparsity_vals, "--", color="gray", alpha=0.5)
    
    ax.axhline(gt_sparsity, linestyle="--", color="green", linewidth=2, alpha=0.7,
               label=f"Ground Truth Sparsity ({gt_sparsity:.1f}%)")
    
    ax.set_xlabel("Sinkhorn Divergence (Fidelity ↓)", fontsize=12)
    ax.set_ylabel("% Non-zero Displacement (Sparsity ↓)", fontsize=12)
    ax.set_title(f"Fidelity-Sparsity Trade-off ({N_FOLDS}-fold CV)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log₁₀(λ)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "synthetic_experiment_results.png", dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "synthetic_experiment_results.pdf", bbox_inches="tight")
    print(f"Plots saved to {OUTPUT_DIR}")
    plt.close()
    
    # ---- Additional Plot: Perturbation Recovery ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(lambdas, cosine_vals, "o-", color="purple", linewidth=2, markersize=8)
    ax.axhline(1.0, linestyle="--", color="green", alpha=0.7, label="Perfect Recovery")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Regularization (λ)", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(f"Perturbation Direction Recovery ({N_FOLDS}-fold CV)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    ax = axes[1]
    ax.plot(lambdas, rel_l2_vals, "s-", color="teal", linewidth=2, markersize=8)
    ax.axhline(0.0, linestyle="--", color="green", alpha=0.7, label="Perfect Recovery")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Regularization (λ)", fontsize=12)
    ax.set_ylabel("Relative L2 Error", fontsize=12)
    ax.set_title(f"Perturbation Magnitude Recovery ({N_FOLDS}-fold CV)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "perturbation_recovery.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment()
