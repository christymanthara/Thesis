import scanpy as sc
import pandas as pd
import numpy as np
import harmonypy as hm
import matplotlib.pyplot as plt
import seaborn as sns

# Set scanpy settings
sc.settings.verbosity = 3  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')

def run_harmony_correction(adata, batch_key, n_components=50, theta=2, random_state=42):
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
    
    Returns:
    --------
    adata : AnnData
        Updated AnnData object with Harmony corrected embedding
    """
    
    # Make a copy to avoid modifying original data
    adata_harmony = adata.copy()
    
    # Step 1: Basic preprocessing if not already done
    print("Checking preprocessing...")
    if 'highly_variable' not in adata_harmony.var.columns:
        print("Finding highly variable genes...")
        sc.pp.highly_variable_genes(adata_harmony, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_harmony.raw = adata_harmony
        adata_harmony = adata_harmony[:, adata_harmony.var.highly_variable]
    
    # Step 2: Scale data if not already done
    if 'X_scaled' not in adata_harmony.layers.keys():
        print("Scaling data...")
        sc.pp.scale(adata_harmony, max_value=10)
    
    # Step 3: PCA if not already computed
    if 'X_pca' not in adata_harmony.obsm.keys():
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
    
    print("Harmony correction completed!")
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
    
    # Compute UMAP for Harmony corrected PCA
    if 'X_umap_harmony' not in adata.obsm.keys():
        print("Computing UMAP for Harmony corrected PCA...")
        sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=10, n_pcs=40, key_added='harmony')
        sc.tl.umap(adata, neighbors_key='harmony', random_state=42)
        adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
        
        # Recompute original UMAP
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata, random_state=42)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original
    sc.pl.umap(adata, color=batch_key, ax=axes[0], show=False, frameon=False)
    axes[0].set_title('Before Harmony Correction')
    
    # Plot Harmony corrected
    sc.pl.umap(adata, color=batch_key, use_rep='X_umap_harmony', ax=axes[1], show=False, frameon=False)
    axes[1].set_title('After Harmony Correction')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_batch_mixing(adata, batch_key):
    """
    Simple evaluation of batch mixing using silhouette score
    """
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

# Example usage:
if __name__ == "__main__":
    # Load your data
    adata = sc.read_h5ad('F:/Thesis/Datasets/baron_2016h.h5ad')
    
    # Make sure your batch information is in adata.obs
    # For example, if you have different batches/samples:
    # adata.obs['batch'] = ['batch1', 'batch2', 'batch1', ...]  # your batch labels
    
    # Run Harmony correction
    adata_corrected = run_harmony_correction(adata, batch_key='batch_id', theta=2)
    
    # Plot comparison
    plot_batch_comparison(adata_corrected, batch_key='batch_id')
    
    # Evaluate batch mixing
    evaluate_batch_mixing(adata_corrected, batch_key='batch_id')
    
    # Save corrected data
    adata_corrected.write('data_harmony_corrected.h5ad')
    
    print("Harmony batch correction pipeline ready!")