import anndata
import scanpy as sc
import os
import pandas as pd  # Ensure pandas handles string assignments correctly
from sklearn import decomposition
import scipy
import numpy as np
from scripts.data_utils.test_save_embeddings import compute_or_load_embedding

def load_and_preprocess_multi_embedder(file1, file2, label_column="labels", use_basename=True, save=False, split_output=False):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    concatenates them, filters genes, and prepares data for multiple embedding methods.
    
    Parameters:
    -----------
    file1, file2 : str
        Paths to the two h5ad files
    label_column : str
        Column name to use for filtering matching cells
    use_basename : bool
        Whether to use basename for source labels
    save : bool
        If True, saves the processed files:
        - Combined: filename1_filename2_preprocessed.h5ad
        - Split: filename1_preprocessed.h5ad, filename2_preprocessed.h5ad
    split_output : bool
        If True, returns (adata1, adata2) tuple instead of combined dataset
    
    Returns:
    --------
    If split_output=False (default):
        AnnData with:
        - .X: Raw counts (for scVI/scANVI)
        - .layers['original_X']: Copy of original raw counts
        - .layers['normalized']: Log-normalized data (for scGPT, UCE)
        - .layers['standardized']: Standardized data (for traditional methods)
        - .obsm['X_pca']: PCA embeddings (50 components)
        - .uns: Preserved unstructured data from both datasets
    
    If split_output=True:
        Tuple of (adata1, adata2) - individual datasets with all processing layers
    """
    
    # Load the datasets
    adata = anndata.read_h5ad(file1)
    new = anndata.read_h5ad(file2)
    
    print(f"📦 Available keys in adata uns before step1 (Unstructured Data):")
    print(list(adata.uns.keys()))
    
    print(f"📦 Available keys in new uns before step2 (Unstructured Data):")
    print(list(new.uns.keys()))

    # PRESERVE .uns DATA BEFORE CONCATENATION
    uns_adata = adata.uns.copy()
    uns_new = new.uns.copy()

    # Debug print to check file paths
    print(f"file1 path: {file1}")
    print(f"file2 path: {file2}")

    # Extract just the filename before ".h5ad"
    def extract_filename(path):
        filename = os.path.basename(path)  # Get file name
        return filename.rsplit('.h5ad', 1)[0]  # Remove the extension

    # Store original source names for later splitting
    source1_name = extract_filename(file1) if use_basename else file1
    source2_name = extract_filename(file2) if use_basename else file2

    # Add source labels (simplified logic since both branches are identical)
    adata.obs["source"] = pd.Categorical([source1_name] * adata.n_obs)
    new.obs["source"] = pd.Categorical([source2_name] * new.n_obs)

    # Debug print to check source labels
    print("Unique source labels in adata:", adata.obs["source"].unique())
    print("Unique source labels in new:", new.obs["source"].unique())

    # Filter new data: keep only cells whose label_column values exist in adata
    if label_column in new.obs and label_column in adata.obs:
        cell_mask = new.obs[label_column].isin(adata.obs[label_column])
        new = new[cell_mask].copy()
    
    # Add after the filtering step:
    print(f"Cells in adata before concat: {adata.n_obs}")
    print(f"Cells in new before concat: {new.n_obs}")
    print(f"Cells in new after filtering: {new.n_obs}")

    
    # Concatenate the two datasets
    full = adata.concatenate(new)
    
    # RESTORE AND MERGE .uns DATA AFTER CONCATENATION
    full.uns = merge_uns_dictionaries(uns_adata, uns_new, source1_name, source2_name)
    
    print(f"📦 Available keys in full.uns after merging:")
    print(list(full.uns.keys()))
    
    # Add after concatenation:
    print(f"Total cells after concat: {full.n_obs}")
    print(f"Source labels after concat: {full.obs['source'].unique()}")
    
    
    print("Unique source labels in Full adata before other processes:", full.obs["source"].unique())

    # Filter genes with at least 1 count
    sc.pp.filter_genes(full, min_counts=1)

    # STORE ORIGINAL X VALUES BEFORE ANY MODIFICATIONS
    full.layers['original_X'] = full.X.copy()
    print("Stored original X values in layers['original_X']")

    # PRESERVE RAW COUNTS in .X for scVI/scANVI
    # Ensure X contains integers/raw counts
    if not np.issubdtype(full.X.dtype, np.integer):
        print("Warning: Converting X to integer counts")
        full.X = full.X.astype(int)

    # Create normalized layer for scGPT, UCE, etc.
    adata_norm = full.copy()
    adata_norm.X = adata_norm.X.astype(float)
    sc.pp.normalize_per_cell(adata_norm, counts_per_cell_after=1_000_000)
    sc.pp.log1p(adata_norm)
    
    # Store normalized data in layers (preserve .uns in full)
    full.layers['normalized'] = adata_norm.X.copy()

    # Create standardized layer for traditional ML methods
    adata_std = adata_norm.copy()
    if scipy.sparse.issparse(adata_std.X):
        adata_std.X = adata_std.X.toarray()
    
    # Standardize with NaN handling
    means = adata_std.X.mean(axis=0)
    stds = adata_std.X.std(axis=0)
    
    # Handle genes with zero variance (std=0) to avoid NaN
    stds[stds == 0] = 1.0  # Set std=1 for constant genes (no scaling effect)
    
    adata_std.X -= means
    adata_std.X /= stds
    
    # Additional safety check: replace any remaining NaN with 0
    adata_std.X = np.nan_to_num(adata_std.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Store standardized data in layers (preserve .uns in full)
    full.layers['standardized'] = adata_std.X.copy()

    # Compute PCA and store in obsm
    full.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(adata_std.X)

    # Optional: Add batch information for scVI
    full.obs['batch'] = full.obs['source'].copy()
    
    print("Unique source labels in Full adata:", full.obs["source"].unique())
    
    # VERIFY .uns IS STILL PRESENT
    print(f"📦 Final .uns keys in full dataset:")
    print(list(full.uns.keys()))

    # Save files if requested
    if save:
        # Generate output filenames
        base1 = extract_filename(file1)
        base2 = extract_filename(file2)
        
        # Save combined file
        combined_filename = f"{base1}_{base2}_preprocessed.h5ad"
        print(f"Saving combined dataset to: {combined_filename}")
        full.write_h5ad(combined_filename)
        
        # Save split files
        adata1_save, adata2_save = split_by_source(full, source1_name, source2_name)
        
        filename1 = f"{base1}_preprocessed.h5ad"
        filename2 = f"{base2}_preprocessed.h5ad"
        
        print(f"Saving split datasets to: {filename1}, {filename2}")
        adata1_save.write_h5ad(filename1)
        adata2_save.write_h5ad(filename2)

    # MODIFIED CLEANUP - DON'T DELETE MODULES THAT MIGHT AFFECT .uns
    import sys
    import gc
    
    # Clear matplotlib figures
    if 'matplotlib.pyplot' in sys.modules:
        import matplotlib.pyplot as plt
        plt.close('all')
    
    # Clear CUDA cache if using GPU
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Force garbage collection (but don't delete modules)
    gc.collect()

    # Return based on split_output flag
    if split_output:
        adata1, adata2 = split_by_source(full, source1_name, source2_name)
        return adata1, adata2
    else:
        return full


def merge_uns_dictionaries(uns1, uns2, source1_name, source2_name):
    """
    Merge .uns dictionaries from two datasets, handling conflicts appropriately.
    
    Parameters:
    -----------
    uns1, uns2 : dict
        The .uns dictionaries from the two datasets
    source1_name, source2_name : str
        Names of the source datasets for conflict resolution
    
    Returns:
    --------
    dict : Merged .uns dictionary
    """
    merged_uns = {}
    
    # Get all unique keys
    all_keys = set(uns1.keys()) | set(uns2.keys())
    
    for key in all_keys:
        if key in uns1 and key in uns2:
            # Both datasets have this key - handle conflicts
            if uns1[key] == uns2[key]:
                # Same value - keep it
                merged_uns[key] = uns1[key]
            else:
                # Different values - store both with source prefixes
                merged_uns[f"{source1_name}_{key}"] = uns1[key]
                merged_uns[f"{source2_name}_{key}"] = uns2[key]
                print(f"⚠️  Conflict in .uns['{key}'] - stored as separate keys with source prefixes")
        elif key in uns1:
            # Only in first dataset
            merged_uns[key] = uns1[key]
        else:
            # Only in second dataset  
            merged_uns[key] = uns2[key]
    
    return merged_uns


def split_by_source(adata, source1_name, source2_name):
    """
    Helper function to split the combined AnnData back into two datasets
    while preserving all layers, embeddings, and .uns data.
    """
    mask_source1 = adata.obs["source"] == source1_name
    mask_source2 = adata.obs["source"] == source2_name
    
    adata1 = adata[mask_source1].copy()
    adata2 = adata[mask_source2].copy()
    
    # Restore original .uns data for each dataset if available
    # Look for source-specific keys in the merged .uns
    adata1_uns = {}
    adata2_uns = {}
    
    for key, value in adata.uns.items():
        if key.startswith(f"{source1_name}_"):
            # Remove prefix and add to adata1
            original_key = key[len(f"{source1_name}_"):]
            adata1_uns[original_key] = value
        elif key.startswith(f"{source2_name}_"):
            # Remove prefix and add to adata2
            original_key = key[len(f"{source2_name}_"):]
            adata2_uns[original_key] = value
        else:
            # Common key - add to both
            adata1_uns[key] = value
            adata2_uns[key] = value
    
    # Update .uns for both datasets
    adata1.uns.update(adata1_uns)
    adata2.uns.update(adata2_uns)
    
    return adata1, adata2

# Usage examples for different embedders:
def prepare_for_scvi(adata):
    """Prepare data for scVI - uses raw counts in .X"""
    return adata  # .X already contains raw counts

def prepare_for_scgpt(adata):
    """Prepare data for scGPT - typically uses log-normalized data"""
    adata_scgpt = adata.copy()
    adata_scgpt.X = adata.layers['normalized'].copy()
    return adata_scgpt

def prepare_for_uce(adata):
    """Prepare data for UCE - check their specific requirements"""
    adata_uce = adata.copy()
    # UCE might need raw counts or normalized - check documentation
    # adata_uce.X = adata.layers['normalized'].copy()  # if normalized
    return adata_uce

def prepare_for_traditional_ml(adata):
    """Prepare data for traditional ML methods - uses standardized data"""
    adata_ml = adata.copy()
    adata_ml.X = adata.layers['standardized'].copy()
    return adata_ml

if __name__ == "__main__":
    combined_data = load_and_preprocess_multi_embedder(
    file1="../Datasets/baron_2016h.h5ad", 
    file2="../Datasets/xin_2016.h5ad",
    save=True,          # Saves all files
    split_output=True   # Returns individual datasets
)