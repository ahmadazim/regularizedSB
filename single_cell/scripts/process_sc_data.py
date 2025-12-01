import gzip
import csv
import random
import numpy as np
from scipy import sparse
import anndata
import pandas as pd
import os
import scanpy as sc

# Paths
DATA_DIR = '/home/tig687/regularizedSB/sc_data'
MATRIX_FILE = os.path.join(DATA_DIR, 'GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz')
CELL_FILE = os.path.join(DATA_DIR, 'GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz')
PDATA_FILE = os.path.join(DATA_DIR, 'GSM4150378_sciPlex3_pData.txt.gz')
GENE_FILE = os.path.join(DATA_DIR, 'GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz')
OUTPUT_FILE = os.path.join(DATA_DIR, 'givinostat_subset_10k.h5ad')

TARGET_DRUGS = ['Givinostat']
TARGET_CELL_LINES = ['MCF7']
TARGET_CELLS = 8000



def main():
    # 1. Load cell annotations (to get original indices)
    print("Loading cell annotations...")
    # Assuming no header based on inspection
    cell_df = pd.read_csv(CELL_FILE, sep='\t', header=None, names=['barcode', 'sample_id'])
    # Store original 0-based index
    cell_df['orig_index'] = range(len(cell_df))
    print(f"Total cells in annotation file: {len(cell_df)}")

    # 2. Load metadata (pData)
    print("Loading metadata...")
    # pData is space-separated with quotes
    pdata_df = pd.read_csv(PDATA_FILE, sep=' ', quotechar='"', low_memory=False)
    # The 'cell' column in pData corresponds to 'barcode' in cell_df
    
    # 3. Merge and Filter
    print("Merging and filtering...")
    merged_df = cell_df.merge(pdata_df, left_on='barcode', right_on='cell', how='inner')
    
    # Filter criteria
    target_drugs = TARGET_DRUGS
    target_cell_lines = TARGET_CELL_LINES
    
    # Filter by cell line
    merged_df = merged_df[merged_df['cell_type'].isin(target_cell_lines)]
    
    # Filter by drug (product_name contains one of the target drugs) or Vehicle
    # We use a regex for "contains" logic
    drug_pattern = '|'.join(target_drugs)
    drug_mask = merged_df['product_name'].str.contains(drug_pattern, case=False, na=False)
    
    # Check for vehicle. 'vehicle' column might be boolean or string.
    # We include cells where vehicle is True.
    # Also check product_name for 'Vehicle' just in case.
    vehicle_mask = (merged_df['vehicle'] == True) | \
                   (merged_df['product_name'].str.contains('Vehicle', case=False, na=False))
                   
    merged_df = merged_df[drug_mask | vehicle_mask]
    
    n_filtered = len(merged_df)
    print(f"Cells after filtering: {n_filtered}")
    
    if n_filtered == 0:
        print("No cells matched the criteria. Exiting.")
        return

    if n_filtered > TARGET_CELLS:
        print(f"Sampling {TARGET_CELLS} cells...")
        # Sample from the filtered dataframe
        sampled_df = merged_df.sample(n=TARGET_CELLS, random_state=42)
    else:
        print("Fewer cells than target, keeping all.")
        sampled_df = merged_df.copy()

    # Sort by original index for efficient matrix reading (though not strictly required, it's good practice)
    sampled_df = sampled_df.sort_values('orig_index')
    
    kept_indices = sampled_df['orig_index'].tolist()
    kept_indices_set = set(kept_indices)

    # Subset cell annotations for AnnData
    # We want to keep the metadata in the final object
    cell_subset = sampled_df.reset_index(drop=True)

    # Create a mapping from original 1-based cell index to new 0-based index
    # Original 1-based index = orig_index + 1
    import re
    # New index will be the position in the sampled_df (0 to N-1)
    # Since we sorted sampled_df by orig_index, the new indices are 0, 1, 2... corresponding to the sorted kept_indices
    original_to_new_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(kept_indices)}

    # 5. Load gene annotations
    print("Loading gene annotations...")
    # The file has a single column "id gene_short_name" when read with default settings or tab
    with gzip.open(GENE_FILE, "rt") as f:
        gene_df_raw = pd.read_csv(f)

    if "id gene_short_name" in gene_df_raw.columns:
        split = gene_df_raw["id gene_short_name"].str.split(r"\s+", n=1, expand=True)
        split.columns = ["id", "gene_short_name"]
        gene_df = split
    elif "id" in gene_df_raw.columns:
        gene_df = gene_df_raw
    else:
        raise ValueError(f"Unexpected columns in gene file: {list(gene_df_raw.columns)}")

    n_genes_total = len(gene_df)
    print(f"Total genes in annotation: {n_genes_total}")

    # Use Ensembl-style id as the var index so var_names are meaningful
    gene_df.set_index("id", inplace=True)

    # 4. Read matrix and filter
    print("Reading matrix and filtering...")
    rows = []
    cols = []
    data = []

    # Matrix is 1-based
    # gene_idx (1..n_genes) -> row (0..n_genes-1)
    # cell_idx (1..n_cells) -> col (0..n_subset-1)

    with gzip.open(MATRIX_FILE, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i % 1000000 == 0:
                print(f"Processed {i} lines...", end='\r')
            
            if not row: continue
            try:
                g_idx = int(row[0])
                c_idx = int(row[1])
                count = float(row[2])
                
                # Check if this cell is in our subset
                # c_idx is 1-based. corresponding 0-based index is c_idx - 1
                orig_0_idx = c_idx - 1
                
                if orig_0_idx in kept_indices_set:
                    new_c_idx = original_to_new_map[orig_0_idx]
                    new_g_idx = g_idx - 1 # 0-based
                    
                    rows.append(new_g_idx)
                    cols.append(new_c_idx)
                    data.append(count)
                    
            except ValueError:
                continue

    print("\nConstructing sparse matrix...")
    # Shape: (n_genes, n_subset_cells)
    # But AnnData expects (n_obs, n_vars) -> (n_cells, n_genes)
    # So we should transpose.
    # Current: rows=genes, cols=cells.
    # We want: rows=cells, cols=genes.
    # So we swap rows and cols when creating COO matrix.
    # Also shape should be (n_subset_cells, n_genes)
    
    mat = sparse.coo_matrix((data, (cols, rows)), shape=(len(kept_indices), n_genes_total))
    mat = mat.tocsr()

    print("Creating AnnData...")
    # At this point, gene_df.index contains real gene IDs (e.g., Ensembl ids)
    # AnnData will use this index as var_names, so downstream DE will report
    # these IDs instead of numeric indices.
    adata = anndata.AnnData(X=mat, obs=cell_subset, var=gene_df)

    # --- FILTERING STEPS ---
    print("Filtering genes and cells...")
    
    print(f"Dimensions before filtering: {adata.n_obs} cells, {adata.n_vars} genes")

    # 1. Filter out cells with fewer than 200 expressed genes
    sc.pp.filter_cells(adata, min_genes=200)
    print(f"Cells after min_genes=200 filter: {adata.n_obs}")

    # 2. Filter out genes expressed in fewer than 20 cells
    sc.pp.filter_genes(adata, min_cells=20)
    print(f"Genes after min_cells=20 filter: {adata.n_vars}")

    # 3. Filter for protein coding genes (heuristic based on gene name)
    # We don't have 'gene_type' in the annotation file, so we'll use heuristics:
    # - Remove genes starting with 'LOC' (often uncharacterized)
    # - Remove genes starting with 'LINC' (long intergenic non-coding)
    # - Remove genes starting with 'MIR' (microRNA)
    # - Remove genes starting with 'SNOR' (snoRNA)
    # - Remove genes containing '-' (often antisense or read-through, though some protein coding have it)
    # A safer bet for "protein coding" without a proper GTF is to keep standard nomenclature.
    # Let's remove obvious non-coding patterns.
    
    if 'gene_short_name' in adata.var.columns:
        gene_names = adata.var['gene_short_name'].astype(str)
        
        # Define patterns to EXCLUDE
        exclude_mask = (
            gene_names.str.startswith('LOC') | 
            gene_names.str.startswith('LINC') | 
            gene_names.str.startswith('MIR') | 
            gene_names.str.startswith('SNOR') |
            gene_names.str.startswith('RPL') | # Ribosomal proteins (optional, but often excluded in analysis)
            gene_names.str.startswith('RPS')   # Ribosomal proteins
        )
        
        # Also, MT- genes (Mitochondrial) are protein coding but often filtered. 
        # Let's keep them for now or filter them if you want strictly nuclear protein coding.
        # Usually we keep them to calculate %MT but maybe filter later.
        
        # Let's just filter the obvious non-coding ones for now.
        adata = adata[:, ~exclude_mask].copy()
        print(f"Genes after removing LOC/LINC/MIR/SNOR/Ribosomal: {adata.n_vars}")

    print(f"Saving to {OUTPUT_FILE}...")
    adata.write(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
