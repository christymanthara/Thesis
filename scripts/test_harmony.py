def run_harmony_correction_simple(adata, batch_key, n_components=50, theta=2, random_state=42):
    """
    Simplified Harmony correction that skips complex preprocessing
    Use this if your data is already preprocessed or if you're having preprocessing issues
    """
    
    print("Running simplified Harmony correction...")
    
    # Make a copy
    adata_harmony = adata.copy()
    
    # Check if batch key exists
    if batch_key not in adata_harmony.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    
    # Basic filtering to remove problematic values
    print("Basic data cleaning...")
    
    # Remove genes with zero expression
    sc.pp.filter_genes(adata_harmony, min_cells=1)
    
    # Remove cells with zero expression
    sc.pp.filter_cells(adata_harmony, min_genes=1)
    
    # If data is not log-transformed, do basic normalization
    if adata_harmony.X.max() > 50:  # Likely raw counts
        print("Data appears to be raw counts, applying normalization...")
        sc.pp.normalize_total(adata_harmony, target_sum=1e4)
        sc.pp.log1p(adata_harmony)
    
    # Handle infinite/NaN values
    if hasattr(adata_harmony.X, 'data'):
        # Sparse matrix
        mask = np.isfinite(adata_harmony.X.data)
        adata_harmony.X.data = adata_harmony.X.data[mask] if mask.sum() < len(adata_harmony.X.data) else adata_harmony.X.data
        adata_harmony.X.data = np.clip(adata_harmony.X.data, -50, 50)
    else:
        # Dense matrix
        adata_harmony.X = np.nan_to_num(adata_harmony.X, nan=0, posinf=50, neginf=-50)
    
    # Simple highly variable genes selection
    print("Finding highly variable genes...")
    try:
        sc.pp.highly_variable_genes(adata_harmony, n_top_genes=2000, flavor='seurat_v3')
        adata_harmony = adata_harmony[:, adata_harmony.var.highly_variable]
    except:
        print("Using all genes (highly variable genes selection failed)")
    
    # Scale data
    print("Scaling data...")
    sc.pp.scale(adata_harmony, max_value=10)
    
    # PCA
    print("Computing PCA...")
    sc.tl.pca(adata_harmony, svd_solver='arpack', n_comps=min(n_components, adata_harmony.n_vars-1), random_state=random_state)
    
    # Run Harmony
    print("Running Harmony...")
    pca_embedding = adata_harmony.obsm['X_pca']
    batch_info = adata_harmony.obs[batch_key].astype(str)
    meta_data = pd.DataFrame({batch_key: batch_info})
    
    try:
        harmony_out = hm.run_harmony(
            pca_embedding,
            meta_data,
            vars_use=[batch_key],
            theta=theta,
            random_state=random_state,
            verbose=True
        )
        adata_harmony.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
        print("Harmony correction completed!")
    except Exception as e:
        print(f"Harmony failed: {e}")
        adata_harmony.obsm['X_pca_harmony'] = adata_harmony.obsm['X_pca'].copy()
    
    return adata_harmony
import scanpy as sc
import pandas as pd
import numpy as np
import harmonypy as hm
import matplotlib.pyplot as plt
import seaborn as sns

# Set scanpy settings
sc.settings.verbosity = 3  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')

def preprocess_data(adata, min_genes=200, min_cells=3, max_genes=5000, 
                   max_pct_mt=20, target_sum=1e4):
    """
    Basic preprocessing to handle problematic values
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    min_genes : int
        Minimum number of genes per cell
    min_cells : int
        Minimum number of cells per gene
    max_genes : int
        Maximum number of genes per cell
    max_pct_mt : float
        Maximum mitochondrial gene percentage
    target_sum : float
        Target sum for normalization
        
    Returns:
    --------
    adata : AnnData
        Preprocessed AnnData object
    """
    
    print("Starting basic preprocessing...")
    
    # Make a copy to avoid modifying original
    adata_pp = adata.copy()
    
    # Basic filtering
    print("Filtering cells and genes...")
    sc.pp.filter_cells(adata_pp, min_genes=min_genes)
    sc.pp.filter_genes(adata_pp, min_cells=min_cells)
    
    # Calculate mitochondrial gene percentage
    # Try different mitochondrial gene patterns
    mt_patterns = ['MT-', 'Mt-', 'mt-', 'MT_', 'Mt_', 'mt_']
    adata_pp.var['mt'] = False
    
    for pattern in mt_patterns:
        mt_mask = adata_pp.var_names.str.startswith(pattern)
        if mt_mask.sum() > 0:
            print(f"Found {mt_mask.sum()} mitochondrial genes with pattern '{pattern}'")
            adata_pp.var['mt'] = mt_mask
            break
    
    if not adata_pp.var['mt'].any():
        print("No mitochondrial genes found, skipping MT filtering")
        adata_pp.var['mt'] = False
    
    # Calculate QC metrics
    try:
        sc.pp.calculate_qc_metrics(adata_pp, percent_top=None, log1p=False, inplace=True)
    except Exception as e:
        print(f"Error calculating QC metrics: {e}")
        # Manually add basic QC metrics
        adata_pp.obs['n_genes_by_counts'] = (adata_pp.X > 0).sum(axis=1).A1 if hasattr(adata_pp.X, 'A1') else (adata_pp.X > 0).sum(axis=1)
        adata_pp.obs['total_counts'] = adata_pp.X.sum(axis=1).A1 if hasattr(adata_pp.X, 'A1') else adata_pp.X.sum(axis=1)
        if adata_pp.var['mt'].any():
            adata_pp.obs['pct_counts_mt'] = (adata_pp[:, adata_pp.var['mt']].X.sum(axis=1).A1 / adata_pp.obs['total_counts'] * 100) if hasattr(adata_pp.X, 'A1') else (adata_pp[:, adata_pp.var['mt']].X.sum(axis=1) / adata_pp.obs['total_counts'] * 100)
        else:
            adata_pp.obs['pct_counts_mt'] = 0
    
    # Filter cells based on QC metrics
    print("Filtering based on QC metrics...")
    sc.pp.filter_cells(adata_pp, max_genes=max_genes)
    
    # Only filter by MT percentage if we have MT genes
    if 'pct_counts_mt' in adata_pp.obs.columns and adata_pp.var['mt'].any():
        print(f"Filtering cells with >={max_pct_mt}% mitochondrial genes")
        adata_pp = adata_pp[adata_pp.obs.pct_counts_mt < max_pct_mt, :]
        print(f"Cells remaining after MT filtering: {adata_pp.n_obs}")
    else:
        print("Skipping mitochondrial gene filtering")
    
    # Save raw counts
    adata_pp.raw = adata_pp
    
    # Normalize to target sum and log transform
    print("Normalizing and log-transforming...")
    sc.pp.normalize_total(adata_pp, target_sum=target_sum)
    sc.pp.log1p(adata_pp)
    
    # Check for and handle infinite values
    print("Checking for infinite values...")
    if np.any(np.isinf(adata_pp.X.data if hasattr(adata_pp.X, 'data') else adata_pp.X)):
        print("Warning: Found infinite values, clipping...")
        if hasattr(adata_pp.X, 'data'):
            # Sparse matrix
            adata_pp.X.data = np.clip(adata_pp.X.data, -50, 50)
        else:
            # Dense matrix
            adata_pp.X = np.clip(adata_pp.X, -50, 50)
    
    print("Basic preprocessing completed!")
    return adata_pp

def run_harmony_correction(adata, batch_key, n_components=50, theta=2, random_state=42,
                          preprocess=True, n_top_genes=2000):
    """
    Apply Harmony batch effect correction to AnnData object
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    batch_key : str
        Key in adata.obs containing batch information
    n_components : int
        Number of PCA components to use (default: 50)
    theta : float
        Diversity clustering penalty parameter (default: 2)
    random_state : int
        Random seed for reproducibility
    preprocess : bool
        Whether to run basic preprocessing
    n_top_genes : int
        Number of highly variable genes to select
    
    Returns:
    --------
    adata : AnnData
        Updated AnnData object with Harmony corrected embedding
    """
    
    # Make a copy to avoid modifying original data
    adata_harmony = adata.copy()
    
    # Check if batch key exists
    if batch_key not in adata_harmony.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    
    # Basic preprocessing if requested
    if preprocess:
        adata_harmony = preprocess_data(adata_harmony)
    
    # Step 1: Find highly variable genes with safer parameters
    print("Finding highly variable genes...")
    if 'highly_variable' not in adata_harmony.var.columns:
        try:
            # Use safer parameters for highly variable genes
            sc.pp.highly_variable_genes(
                adata_harmony, 
                min_mean=0.01,  # Slightly higher minimum
                max_mean=5,     # Higher maximum
                min_disp=0.3,   # Lower dispersion threshold
                n_top_genes=n_top_genes,
                flavor='seurat_v3'  # More robust method
            )
        except ValueError as e:
            print(f"Error with highly_variable_genes: {e}")
            print("Trying alternative approach...")
            # Alternative: use scanpy's newer method
            sc.pp.highly_variable_genes(
                adata_harmony,
                n_top_genes=n_top_genes,
                flavor='seurat_v3',
                batch_key=batch_key if batch_key in adata_harmony.obs.columns else None
            )
    
    # Keep only highly variable genes
    if 'highly_variable' in adata_harmony.var.columns:
        adata_harmony = adata_harmony[:, adata_harmony.var.highly_variable]
        print(f"Kept {adata_harmony.n_vars} highly variable genes")
    
    # Step 2: Scale data
    print("Scaling data...")
    sc.pp.scale(adata_harmony, max_value=10)
    
    # Step 3: PCA
    print("Computing PCA...")
    sc.tl.pca(adata_harmony, svd_solver='arpack', n_comps=n_components, random_state=random_state)
    
    # Step 4: Run Harmony
    print("Running Harmony correction...")
    
    # Extract PCA embedding
    pca_embedding = adata_harmony.obsm['X_pca']
    
    # Get batch information
    batch_info = adata_harmony.obs[batch_key].astype(str)
    
    # Create metadata dataframe for Harmony
    meta_data = pd.DataFrame({batch_key: batch_info})
    
    # Run Harmony
    try:
        harmony_out = hm.run_harmony(
            pca_embedding,
            meta_data,
            vars_use=[batch_key],
            theta=theta,
            random_state=random_state,
            verbose=True
        )
        
        # Store Harmony corrected embedding
        adata_harmony.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
        print("Harmony correction completed successfully!")
        
    except Exception as e:
        print(f"Error running Harmony: {e}")
        print("Continuing without Harmony correction...")
        adata_harmony.obsm['X_pca_harmony'] = adata_harmony.obsm['X_pca'].copy()
    
    return adata_harmony

def plot_batch_comparison(adata, batch_key, save_path=None):
    """
    Plot comparison before and after Harmony correction
    """
    
    # Compute UMAP for original PCA
    if 'X_umap' not in adata.obsm.keys():
        print("Computing UMAP for original PCA...")
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata, random_state=42)
    
    # Store original UMAP
    X_umap_original = adata.obsm['X_umap'].copy()
    
    # Compute UMAP for Harmony corrected PCA
    if 'X_pca_harmony' in adata.obsm.keys():
        print("Computing UMAP for Harmony corrected PCA...")
        sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=10, n_pcs=40, key_added='harmony')
        sc.tl.umap(adata, neighbors_key='harmony', random_state=42)
        X_umap_harmony = adata.obsm['X_umap'].copy()
        
        # Restore original UMAP
        adata.obsm['X_umap'] = X_umap_original
        adata.obsm['X_umap_harmony'] = X_umap_harmony
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original
    sc.pl.umap(adata, color=batch_key, ax=axes[0], show=False, frameon=False)
    axes[0].set_title('Before Harmony Correction')
    
    # Plot Harmony corrected
    if 'X_umap_harmony' in adata.obsm.keys():
        # Temporarily set harmony UMAP for plotting
        temp_umap = adata.obsm['X_umap'].copy()
        adata.obsm['X_umap'] = adata.obsm['X_umap_harmony']
        sc.pl.umap(adata, color=batch_key, ax=axes[1], show=False, frameon=False)
        adata.obsm['X_umap'] = temp_umap  # Restore original
        axes[1].set_title('After Harmony Correction')
    else:
        axes[1].text(0.5, 0.5, 'Harmony correction not available', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_batch_mixing(adata, batch_key):
    """
    Simple evaluation of batch mixing using silhouette score
    """
    try:
        from sklearn.metrics import silhouette_score
        
        # Original PCA
        if 'X_pca' in adata.obsm.keys():
            sil_original = silhouette_score(adata.obsm['X_pca'], adata.obs[batch_key])
            print(f"Silhouette score (original PCA): {sil_original:.3f}")
        
        # Harmony corrected
        if 'X_pca_harmony' in adata.obsm.keys():
            sil_harmony = silhouette_score(adata.obsm['X_pca_harmony'], adata.obs[batch_key])
            print(f"Silhouette score (Harmony corrected): {sil_harmony:.3f}")
            print("Lower silhouette score indicates better batch mixing")
            
    except ImportError:
        print("sklearn not available for batch mixing evaluation")
    except Exception as e:
        print(f"Error evaluating batch mixing: {e}")

# Example usage:
if __name__ == "__main__":
    # Load your data
    print("Loading data...")
    adata = sc.read_h5ad('F:/Thesis/Datasets/baron_2016h.h5ad')
    
    print(f"Data shape: {adata.shape}")
    print(f"Available obs keys: {list(adata.obs.keys())}")
    
    # Check what batch keys are available
    batch_candidates = [col for col in adata.obs.columns if 'batch' in col.lower() or 'sample' in col.lower()]
    print(f"Potential batch keys: {batch_candidates}")
    
    # You might need to adjust the batch_key based on your data
    # Common batch keys: 'batch', 'sample', 'donor', 'batch_id', 'sample_id'
    batch_key = 'batch_id'  # Adjust this based on your data
    
    if batch_key not in adata.obs.columns:
        print(f"Warning: '{batch_key}' not found in data. Available columns: {list(adata.obs.columns)}")
        # You might need to create batch information or use a different key
        
    try:
        # First try the simple version
        print("Trying simplified Harmony correction...")
        adata_corrected = run_harmony_correction_simple(
            adata, 
            batch_key=batch_key, 
            theta=2
        )
        
        # Plot comparison
        plot_batch_comparison(adata_corrected, batch_key=batch_key)
        
        # Evaluate batch mixing
        evaluate_batch_mixing(adata_corrected, batch_key=batch_key)
        
        # Save corrected data
        adata_corrected.write('data_harmony_corrected.h5ad')
        
        print("Harmony batch correction pipeline completed successfully!")
        
    except Exception as e:
        print(f"Simple method failed: {e}")
        print("Trying full preprocessing method...")
        
        try:
            # Run Harmony correction with full preprocessing
            adata_corrected = run_harmony_correction(
                adata, 
                batch_key=batch_key, 
                theta=2,
                preprocess=True,
                n_top_genes=2000
            )
            
            # Plot comparison
            plot_batch_comparison(adata_corrected, batch_key=batch_key)
            
            # Evaluate batch mixing
            evaluate_batch_mixing(adata_corrected, batch_key=batch_key)
            
            # Save corrected data
            adata_corrected.write('data_harmony_corrected.h5ad')
            
            print("Harmony batch correction pipeline completed successfully!")
            
        except Exception as e2:
            print(f"Full preprocessing method also failed: {e2}")
            print("Please check your data and batch key configuration.")
            
            # Print data info to help debug
            print(f"\nData info:")
            print(f"Shape: {adata.shape}")
            print(f"X type: {type(adata.X)}")
            print(f"X range: {adata.X.min():.2f} to {adata.X.max():.2f}")
            print(f"obs columns: {list(adata.obs.columns)}")
            if batch_key in adata.obs.columns:
                print(f"Batch key '{batch_key}' unique values: {adata.obs[batch_key].unique()}")
            else:
                print(f"Batch key '{batch_key}' not found!")