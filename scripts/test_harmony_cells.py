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
        
def plot_batch_specific_labels(adata, batch_key, label_key, batch_values=None, 
                               use_harmony=True, save_path=None, figsize=(15, 6)):
    """
    Plot labels with batch-specific coloring
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    batch_key : str
        Key in adata.obs containing batch information
    label_key : str
        Key in adata.obs containing label information
    batch_values : list or None
        List of two batch values [batch1, batch2]. If None, uses first two unique values
    use_harmony : bool
        Whether to use Harmony-corrected UMAP (default: True)
    save_path : str or None
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Check if required keys exist
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    if label_key not in adata.obs.columns:
        raise ValueError(f"Label key '{label_key}' not found in adata.obs")
    
    # Determine which UMAP to use
    if use_harmony and 'X_umap_harmony' in adata.obsm.keys():
        umap_coords = adata.obsm['X_umap_harmony']
        title_suffix = " (Harmony Corrected)"
    else:
        umap_coords = adata.obsm['X_umap']
        title_suffix = " (Original)"
    
    # Get batch values
    unique_batches = adata.obs[batch_key].unique()
    if batch_values is None:
        if len(unique_batches) < 2:
            raise ValueError(f"Need at least 2 batches, found {len(unique_batches)}: {unique_batches}")
        batch_values = unique_batches[:2]
        print(f"Using batch values: {batch_values}")
    
    batch1, batch2 = batch_values
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get unique labels and create color map
    unique_labels = adata.obs[label_key].unique()
    n_labels = len(unique_labels)
    
    # Use a colormap with distinct colors
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, n_labels)) if n_labels <= 20 else cm.hsv(np.linspace(0, 1, n_labels))
    label_color_map = dict(zip(unique_labels, colors))
    
    # Plot 1: All labels with normal colors
    for i, label in enumerate(unique_labels):
        mask = adata.obs[label_key] == label
        axes[0].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=[label_color_map[label]], 
            label=label,
            s=20,
            alpha=0.7
        )
    
    axes[0].set_title(f'All Labels{title_suffix}')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Batch-specific coloring
    # First plot batch1 cells in grey
    batch1_mask = adata.obs[batch_key] == batch1
    if batch1_mask.sum() > 0:
        axes[1].scatter(
            umap_coords[batch1_mask, 0], 
            umap_coords[batch1_mask, 1],
            c='lightgrey', 
            label=f'{batch1} (all labels)',
            s=20,
            alpha=0.5
        )
    
    # Then plot batch2 cells with their label colors
    batch2_mask = adata.obs[batch_key] == batch2
    if batch2_mask.sum() > 0:
        batch2_data = adata.obs[batch2_mask]
        for label in unique_labels:
            label_mask = batch2_data[label_key] == label
            if label_mask.sum() > 0:
                # Get indices in the full dataset
                full_indices = batch2_data.index[label_mask]
                full_mask = adata.obs.index.isin(full_indices)
                
                axes[1].scatter(
                    umap_coords[full_mask, 0], 
                    umap_coords[full_mask, 1],
                    c=[label_color_map[label]], 
                    label=f'{batch2}: {label}',
                    s=20,
                    alpha=0.7
                )
    
    axes[1].set_title(f'{batch1} (Grey) vs {batch2} (Colored){title_suffix}')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total cells: {adata.n_obs}")
    print(f"Batch '{batch1}': {batch1_mask.sum()} cells")
    print(f"Batch '{batch2}': {batch2_mask.sum()} cells")
    print(f"Labels: {list(unique_labels)}")
    
    # Show label distribution per batch
    cross_tab = pd.crosstab(adata.obs[batch_key], adata.obs[label_key])
    print(f"\nLabel distribution per batch:")
    print(cross_tab)


def plot_comprehensive_batch_analysis(adata, batch_key, label_key, batch_values=None, save_path=None):
    """
    Create a comprehensive 4-panel plot showing:
    1. Labels before correction
    2. Labels after correction  
    3. Batch-specific labels (before correction)
    4. Batch-specific labels (after correction)
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Check required embeddings
    has_harmony = 'X_umap_harmony' in adata.obsm.keys()
    
    if not has_harmony:
        print("Warning: Harmony corrected UMAP not found. Computing now...")
        # You might need to run the harmony correction first
        return
    
    # Get batch values
    unique_batches = adata.obs[batch_key].unique()
    if batch_values is None:
        batch_values = unique_batches[:2]
    batch1, batch2 = batch_values
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Get unique labels and colors
    unique_labels = adata.obs[label_key].unique()
    n_labels = len(unique_labels)
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, n_labels)) if n_labels <= 20 else cm.hsv(np.linspace(0, 1, n_labels))
    label_color_map = dict(zip(unique_labels, colors))
    
    # Function to plot labels
    def plot_labels(ax, umap_coords, title):
        for label in unique_labels:
            mask = adata.obs[label_key] == label
            ax.scatter(
                umap_coords[mask, 0], 
                umap_coords[mask, 1],
                c=[label_color_map[label]], 
                label=label,
                s=15,
                alpha=0.7
            )
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    # Function to plot batch-specific labels
    def plot_batch_specific(ax, umap_coords, title):
        # Grey for batch1
        batch1_mask = adata.obs[batch_key] == batch1
        if batch1_mask.sum() > 0:
            ax.scatter(
                umap_coords[batch1_mask, 0], 
                umap_coords[batch1_mask, 1],
                c='lightgrey', 
                s=15,
                alpha=0.5
            )
        
        # Colors for batch2
        batch2_mask = adata.obs[batch_key] == batch2
        if batch2_mask.sum() > 0:
            batch2_data = adata.obs[batch2_mask]
            for label in unique_labels:
                label_mask = batch2_data[label_key] == label
                if label_mask.sum() > 0:
                    full_indices = batch2_data.index[label_mask]
                    full_mask = adata.obs.index.isin(full_indices)
                    ax.scatter(
                        umap_coords[full_mask, 0], 
                        umap_coords[full_mask, 1],
                        c=[label_color_map[label]], 
                        s=15,
                        alpha=0.7
                    )
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    # Plot 1: Labels before correction
    plot_labels(axes[0,0], adata.obsm['X_umap'], 'Labels (Before Correction)')
    
    # Plot 2: Labels after correction
    plot_labels(axes[0,1], adata.obsm['X_umap_harmony'], 'Labels (After Correction)')
    
    # Plot 3: Batch-specific before correction
    plot_batch_specific(axes[1,0], adata.obsm['X_umap'], 
                       f'{batch1} (Grey) vs {batch2} (Colored) - Before')
    
    # Plot 4: Batch-specific after correction
    plot_batch_specific(axes[1,1], adata.obsm['X_umap_harmony'], 
                       f'{batch1} (Grey) vs {batch2} (Colored) - After')
    
    # Add legends
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
def run_comprehensive_analysis(adata, batch_key, label_key):
    """
    Run the complete analysis with both batch correction and label visualization
    """
    
    print("Available batch values:", adata.obs[batch_key].unique())
    print("Available label values:", adata.obs[label_key].unique())
    
    # First run harmony correction if not done already
    if 'X_pca_harmony' not in adata.obsm.keys():
        print("Running Harmony correction first...")
        adata = run_harmony_correction_simple(adata, batch_key)
    
    # Compute UMAPs if not available
    if 'X_umap' not in adata.obsm.keys():
        print("Computing original UMAP...")
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata, random_state=42)
    
    if 'X_umap_harmony' not in adata.obsm.keys():
        print("Computing Harmony UMAP...")
        sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=10, n_pcs=40, key_added='harmony')
        sc.tl.umap(adata, neighbors_key='harmony', random_state=42)
        # Store harmony UMAP separately
        adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
        # Recompute original UMAP
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata, random_state=42)
    
    # Create the batch-specific label plots
    print("Creating batch-specific label visualization...")
    plot_batch_specific_labels(adata, batch_key, label_key, use_harmony=False)
    plot_batch_specific_labels(adata, batch_key, label_key, use_harmony=True)
    
    # Create comprehensive analysis
    print("Creating comprehensive analysis...")
    plot_comprehensive_batch_analysis(adata, batch_key, label_key)
    
    return adata


# Example usage:
if __name__ == "__main__":
    # Load your data
    print("Loading data...")
    adata = sc.read_h5ad('F:/Thesis/baron_2016h_muraro_transformed_filtered.h5ad')
    
    print(f"Data shape: {adata.shape}")
    print(f"Available obs keys: {list(adata.obs.keys())}")
    
    # Check what batch keys are available
    batch_candidates = [col for col in adata.obs.columns if 'batch' in col.lower() or 'sample' in col.lower()]
    print(f"Potential batch keys: {batch_candidates}")
    
    # You might need to adjust the batch_key based on your data
    # Common batch keys: 'batch', 'sample', 'donor', 'batch_id', 'sample_id'
    batch_key = 'batch'  # Adjust this based on your data
    label_key = 'labels'  # Adjust this based on your data
    
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
        
        # Or run individual plots:
        # Just the batch-specific label plots
        plot_batch_specific_labels(adata, batch_key, label_key, use_harmony=False)  # Before correction
        plot_batch_specific_labels(adata, batch_key, label_key, use_harmony=True)   # After correction

        # Or the full 4-panel analysis
        plot_comprehensive_batch_analysis(adata, batch_key, label_key)

        
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
            
            #adding the code to print batch-specific labels
            print("Plotting batch-specific labels... using label_key:", label_key)
            run_comprehensive_analysis(adata_corrected, batch_key=batch_key, label_key=label_key)
            
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