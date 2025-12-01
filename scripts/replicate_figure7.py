
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import geomloss
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from regularizedSB.solver import run_regularized_sb
from regularizedSB.config import ExperimentConfig, SolverConfig, NetworkConfig, LoggingConfig, DatasetConfig, PenaltyConfig, MetricsConfig
from regularizedSB.solver import RegularizedSBSolver
from regularizedSB.metrics import rollout_bridge, u_nom_fn

# Configuration
DATA_PATH = '/n/holylfs06/LABS/mzitnik_lab/Users/rzhu/regularizedSB/sc_data/raw_data/sciPlex3_K562_Givinostat.h5ad'
OUTPUT_DIR = '/n/holylfs06/LABS/mzitnik_lab/Users/rzhu/regularizedSB/outputs/figure7_replication_givinostat_T40_loops30_n2000_1e-5cutoff_mmd_3folds_heuristicbandwidth'
GAMMAS = [0.25, 1, 500, 500000]
N_FOLDS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DRUG = "Givinostat"
CELL_LINE = "K562"
def setup_data(n_top_genes=None):
    print(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    
    # Preprocessing
    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if n_top_genes is not None:
        print(f"Selecting top {n_top_genes} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"  After HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Identify source (Vehicle) and target (DRUG)
    vehicle_mask = (adata.obs['vehicle'] == True) | \
                   (adata.obs['product_name'].str.contains('Vehicle', case=False, na=False))
    drug_mask = adata.obs['product_name'].str.contains(DRUG, case=False, na=False)
    
    source_adata = adata[vehicle_mask].copy()
    target_adata = adata[drug_mask].copy()
    
    print(f"Source cells: {source_adata.n_obs}")
    print(f"Target cells: {target_adata.n_obs}")
    
    return source_adata, target_adata

def compute_sinkhorn_div(X_transported, X_target):
    """
    Computes sinkhorn divergence between two point clouds in gene space.
    X_transported: (N, D) numpy array
    X_target: (M, D) numpy array
    """
    # Convert to torch tensors
    xt = torch.from_numpy(X_transported).float().to(DEVICE)
    yt = torch.from_numpy(X_target).float().to(DEVICE)
    
    # Define sinkhorn loss
    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05, debias=True)
    
    with torch.no_grad():
        loss = loss_fn(xt, yt)
    
    return loss.item()

def compute_sparsity(X_transported, X_source, threshold=1e-5): 
    """
    Computes % non-zero elements in displacement in gene space and prints summary stats
    to help choose a good threshold.
    """
    displacement = X_transported - X_source
    abs_disp = np.abs(displacement)
    
    # Sparsity: % of dimensions with displacement above threshold
    # We compute this per cell, then average, to capture varying sparsity patterns.
    is_nonzero = abs_disp > threshold
    sparsity_per_cell = np.mean(is_nonzero, axis=1)  # fraction per cell
    sparsity_mean = np.mean(sparsity_per_cell) * 100.0  # percent
    
    # Percentiles of per-cell sparsity (0..100 step 10)
    sparsity_percentiles = np.percentile(sparsity_per_cell * 100.0, np.arange(0, 101, 10))
    
    # Percentiles of absolute displacements across all entries (useful to judge threshold)
    abs_flat = abs_disp.ravel()
    abs_percentiles = np.percentile(abs_flat, np.arange(0, 101, 10))
    
    # Print summaries to help tune threshold
    print(f"  Sparsity summary (threshold={threshold}): mean={sparsity_mean:.4f}% ", flush=True)
    print("  Sparsity percentiles (0..100 step10) [%]: " + ", ".join(f"{p:.3f}" for p in sparsity_percentiles), flush=True)
    print("  Displacement abs-value percentiles (0..100 step10): " + ", ".join(f"{p:.6e}" for p in abs_percentiles), flush=True)
    
    return sparsity_mean

def run_single_gamma(gamma, gpu_id, S_full, T_full, source_splits, target_splits, n_genes, output_dir, gene_names, bandwidth):
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    
    # Update global DEVICE for helper functions in this process
    global DEVICE
    DEVICE = device
    
    print(f"Process for Gamma={gamma} started on {device}", flush=True)
    
    results = []
    n_folds = len(source_splits)
    
    for fold_idx in range(n_folds):
        # Get train/test indices
        train_idx_s, test_idx_s = source_splits[fold_idx]
        train_idx_t, test_idx_t = target_splits[fold_idx]
        
        # Train data
        S_train = S_full[train_idx_s]
        T_train = T_full[train_idx_t]
        
        # Test data
        S_test = S_full[test_idx_s]
        T_test = T_full[test_idx_t]
        
        # Config
        solver_cfg = SolverConfig(
            d=n_genes,
            n_particles=S_train.shape[0],
            lam=gamma,
            eps=0.1, 
            u_max=10.0, 
            T=40,
            outer_loops=30,
            value_epochs=5,
            policy_epochs=5,
            target_type="dataset"
        )
        
        # Hidden dimension
        hidden_dim = 256
        
        exp_cfg = ExperimentConfig(
            solver=solver_cfg,
            value_net=NetworkConfig(name="ValueNet", hidden=hidden_dim),
            policy_net=NetworkConfig(name="PolicyNet", hidden=hidden_dim),
            logging=LoggingConfig(output_dir=output_dir, save_final=False, save_samples=False),
            dataset=DatasetConfig(),
            penalty=PenaltyConfig(name="mmd", params={"bandwidth": bandwidth}), # Use MMD to match target distribution, not just mean (quadratic)
            # penalty=PenaltyConfig(name="quadratic"),
            metrics=MetricsConfig(enabled=False)
        )
        
        solver = RegularizedSBSolver(exp_cfg, source_data=S_train, target_data=T_train)
        solver.run()
        
        # Apply learned policy to S_test
        u_nom = u_nom_fn(solver.policy_net, solver_cfg.T, solver_cfg.u_max)
        
        # 1. Sparsity (in gene space) - use deterministic rollout (eps=0) to measure drift sparsity
        X_transported_det, _ = rollout_bridge(S_test, solver_cfg.T, 0.0, u_nom)
        sparsity = compute_sparsity(X_transported_det, S_test)

        # Identify top 10 genes by mean absolute displacement
        displacement = X_transported_det - S_test
        mean_abs_disp = np.mean(np.abs(displacement), axis=0)
        top_10_idx = np.argsort(mean_abs_disp)[-10:][::-1]
        
        top_genes_data = {}
        for rank, idx in enumerate(top_10_idx):
            top_genes_data[f'gene_rank_{rank+1}'] = gene_names[idx]
            top_genes_data[f'disp_rank_{rank+1}'] = mean_abs_disp[idx]
        
        # 2. Sinkhorn div (in gene space) - use stochastic rollout (eps=solver_cfg.eps) for distribution match
        X_transported, _ = rollout_bridge(S_test, solver_cfg.T, solver_cfg.eps, u_nom)
        sinkhorn = compute_sinkhorn_div(X_transported, T_test)
        
        res_entry = {
            'gamma': gamma,
            'fold': fold_idx,
            'sparsity': sparsity,
            'sinkhorn': sinkhorn
        }
        res_entry.update(top_genes_data)
        results.append(res_entry)
        print(f"  Gamma={gamma} Fold {fold_idx+1}/{n_folds} Done.", flush=True)
        
    # Save results for this gamma
    gamma_results_path = os.path.join(output_dir, f'results_gamma_{gamma}.csv')
    pd.DataFrame(results).to_csv(gamma_results_path, index=False)
    print(f"Saved results for Gamma={gamma} to {gamma_results_path}", flush=True)
        
    return results

def run_experiment(n=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    source_adata, target_adata = setup_data(n_top_genes=n)
    
    # Extract gene names if available
    if 'id gene_short_name' in source_adata.var.columns:
        gene_names = source_adata.var['id gene_short_name'].astype(str).apply(lambda x: x.split(' ')[-1] if ' ' in x else x).tolist()
    elif 'gene_short_name' in source_adata.var.columns:
        gene_names = source_adata.var['gene_short_name'].astype(str).tolist()
    else:
        gene_names = source_adata.var_names.tolist()
    
    # Convert to dense arrays for the solver
    S_full = source_adata.X.toarray() if hasattr(source_adata.X, 'toarray') else source_adata.X
    T_full = target_adata.X.toarray() if hasattr(target_adata.X, 'toarray') else target_adata.X
    
    n_genes = S_full.shape[1]
    print(f"Running in full gene space with {n_genes} genes.")

    # Compute heuristic bandwidth
    idx_s = np.random.choice(S_full.shape[0], min(1000, S_full.shape[0]), replace=False)
    idx_t = np.random.choice(T_full.shape[0], min(1000, T_full.shape[0]), replace=False)
    dists = cdist(S_full[idx_s], T_full[idx_t], metric='euclidean')
    median_dist = np.median(dists)
    bandwidth = median_dist / 2.0
    print(f"Computed heuristic bandwidth: {bandwidth:.4f} (median dist: {median_dist:.4f})")
    
    # Prepare cross-validation splits, handling edge-cases where N_FOLDS < 2 or sample counts are small.
    max_possible = min(S_full.shape[0], T_full.shape[0])
    if N_FOLDS is None or N_FOLDS < 2 or max_possible < 2:
        # Single fold: use all data as both train and test to avoid KFold errors
        # Modified to 80/20 split of the full dataset
        s_indices = np.arange(S_full.shape[0])
        np.random.shuffle(s_indices)
        n_s = int(0.8 * len(s_indices))
        source_splits = [(s_indices[:n_s], s_indices[n_s:])]

        t_indices = np.arange(T_full.shape[0])
        np.random.shuffle(t_indices)
        n_t = int(0.8 * len(t_indices))
        target_splits = [(t_indices[:n_t], t_indices[n_t:])]
    else:
        n_splits = min(N_FOLDS, S_full.shape[0], T_full.shape[0])
        if n_splits < 2:
            # Fallback to single-fold if computed n_splits is less than 2
            s_indices = np.arange(S_full.shape[0])
            np.random.shuffle(s_indices)
            n_s = int(0.8 * len(s_indices))
            source_splits = [(s_indices[:n_s], s_indices[n_s:])]

            t_indices = np.arange(T_full.shape[0])
            np.random.shuffle(t_indices)
            n_t = int(0.8 * len(t_indices))
            target_splits = [(t_indices[:n_t], t_indices[n_t:])]
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Custom split: Use the fold subset as the dataset, then split 80/20
            source_splits = []
            for _, fold_indices in kf.split(S_full):
                # fold_indices is the "test" part of KFold, which is 1/N of data
                # We treat this as our dataset for this fold
                np.random.shuffle(fold_indices)
                n_train = int(0.8 * len(fold_indices))
                source_splits.append((fold_indices[:n_train], fold_indices[n_train:]))
                
            target_splits = []
            for _, fold_indices in kf.split(T_full):
                np.random.shuffle(fold_indices)
                n_train = int(0.8 * len(fold_indices))
                target_splits.append((fold_indices[:n_train], fold_indices[n_train:]))
    
    # parallel execution
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
    if n_gpus > 0:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass 
            
        pool_size = min(len(GAMMAS), n_gpus)
        print(f"Parallelizing across {pool_size} GPUs...")
        
        args_list = []
        for i, gamma in enumerate(GAMMAS):
            gpu_id = i % n_gpus
            args_list.append((gamma, gpu_id, S_full, T_full, source_splits, target_splits, n_genes, OUTPUT_DIR, gene_names, bandwidth))
            
        with mp.Pool(pool_size) as pool:
            nested_results = pool.starmap(run_single_gamma, args_list)
            
        results = [item for sublist in nested_results for item in sublist]
        
    else:
        print("No GPUs found. Running sequentially on CPU (this will be slow)...")
        results = []
        for gamma in GAMMAS:
            results.extend(run_single_gamma(gamma, 0, S_full, T_full, source_splits, target_splits, n_genes, OUTPUT_DIR, gene_names, bandwidth))
            
    # Save results
    df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, 'results.csv')
    df.to_csv(results_path, index=False)
    print(f"\nRaw results (per fold) saved to {results_path}")

    # Save summary stats (means and stds for plotting)
    summary = df.groupby('gamma')[['sparsity', 'sinkhorn']].agg(['mean', 'std'])
    summary_path = os.path.join(OUTPUT_DIR, 'summary_stats.csv')
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to {summary_path}")
    
    plot_results(df)

def plot_results(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: sinkhorn div
    ax = axes[0]
    means = df.groupby('gamma')['sinkhorn'].mean()
    
    # Scatter points
    for gamma in GAMMAS:
        subset = df[df['gamma'] == gamma]
        ax.scatter([gamma] * len(subset), subset['sinkhorn'], alpha=0.5, color='blue')
    
    # Plot mean line
    ax.plot(means.index, means.values, 'x-', color='navy', markersize=10, label='Regularized SB')
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Regularization (gamma)')
    ax.set_ylabel('l_2^2 Sinkhorn div., gene space')
    ax.set_title(f'{DRUG}, {CELL_LINE}')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Plot 2: sparsity
    ax = axes[1]
    means_sp = df.groupby('gamma')['sparsity'].mean()
    
    # Scatter points
    for gamma in GAMMAS:
        subset = df[df['gamma'] == gamma]
        ax.scatter([gamma] * len(subset), subset['sparsity'], alpha=0.5, color='blue')
        
    ax.plot(means_sp.index, means_sp.values, 'x-', color='navy', markersize=10, label='Regularized SB')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') 
    ax.set_xlabel('Regularization (gamma)')
    ax.set_ylabel('% non-zero in disp.')
    ax.set_title(f'{DRUG}, {CELL_LINE}')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure7_replication.png'))
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'figure7_replication.png')}")

if __name__ == "__main__":
    run_experiment(n=2000)
