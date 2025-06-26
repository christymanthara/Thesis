import anndata
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE, TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
from os import path
from sklearn import decomposition
import pandas as pd

def preprocess_anndata(file_path):
    """
    Load and preprocess AnnData object using all genes.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
        
    Returns:
    --------
    adata_norm : AnnData
        Preprocessed AnnData object
    """
    # Load dataset
    adata = anndata.read_h5ad(file_path)
    sc.pp.filter_genes(adata, min_counts=1)
    
    # Normalize data
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1_000_000)
    sc.pp.log1p(adata_norm)
    adata_norm.X = adata_norm.X.toarray()
    adata_norm.X -= adata_norm.X.mean(axis=0)
    adata_norm.X /= adata_norm.X.std(axis=0)
    
    return adata_norm

def compute_pca(adata):
    """
    Compute PCA for AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        Input AnnData object
    """
    print(f"Computing PCA for {adata.shape[1]} genes")
    # U, S, V = np.linalg.svd(adata.X, full_matrices=False)
    # U[:, np.sum(V, axis=1) < 0] *= -1
    # adata.obsm["pca"] = np.dot(U, np.diag(S))[:, np.argsort(S)[::-1]][:, :50]
    pca_components = decomposition.PCA(n_components=50).fit_transform(adata.X)
    adata.obsm["pca"] = pca_components

def compute_tsne(adata):
    """
    Compute t-SNE for AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        Input AnnData object (must have PCA computed)
    """
    print(f"Computing t-SNE for {adata.shape[1]} genes")
    
    # Compute affinities using multiscale approach
    affinities = affinity.Multiscale(
        adata.obsm["pca"],
        perplexities=[50, 500],
        metric="cosine",
        n_jobs=8,
        random_state=3,
    )
    
    # Initialize embedding using PCA
    init = initialization.pca(adata.obsm["pca"], random_state=42)
    
    # Create t-SNE embedding
    embedding = TSNEEmbedding(
        init, affinities, negative_gradient_method="fft", n_jobs=8
    )
    
    # Optimize embedding in two phases
    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
    
    # Store result
    adata.obsm["X_pavlin_tsne"] = np.array(embedding)

def process_single_anndata(file_path, save_output=True):
    """
    Complete pipeline to process a single AnnData file with all genes.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
    save_output : bool, default=True
        Whether to save the processed data
        
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with PCA and t-SNE embeddings
    """
    # Get filename for saving
    file_name = path.splitext(path.basename(file_path))[0]
    
    
    
    # Preprocess data
    adata = preprocess_anndata(file_path)
    
    # Add source labels (simplified logic since both branches are identical)
    adata.obs["source"] = pd.Categorical([file_name] * adata.n_obs)
    
    
    
    # Compute PCA
    compute_pca(adata)
    print(f"PCA shape: {adata.obsm['pca'].shape}")
    
    # Compute t-SNE
    compute_tsne(adata)
    print(f"t-SNE shape: {adata.obsm['X_pavlin_tsne'].shape}")
    
    # Save if requested
    if save_output:
        output_file = f"{file_name}_embedding_tsne_all_genes.h5ad"
        adata.write_h5ad(output_file)
        print(f"Saved processed data to: {output_file}")
    
    return adata

# Example usage
if __name__ == "__main__":
    # Process single dataset
    adata = process_single_anndata("Datasets/baron_2016h.h5ad")
    
    # Or without saving
    # adata = process_single_anndata("Datasets/baron_2016h.h5ad", save_output=False)