import anndata
import scanpy as sc
import os
from sklearn import decomposition

def load_and_preprocess(file1, file2, label_column="labels", use_basename=True):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    concatenates them, filters genes, normalizes, log-transforms, standardizes,
    and computes PCA. Returns the processed AnnData object.
    
    Parameters:
    - file1 (str): Path to the first .h5ad file.
    - file2 (str): Path to the second .h5ad file.
    - label_column (str): The column name in `obs` used to filter matching cells.
    - use_basename (bool): If True, use os.path.basename for source labels. Otherwise, use the full file paths.
    
    Returns:
    - full (AnnData): The processed, concatenated AnnData object with PCA computed.
    """
    # Load the datasets
    adata = anndata.read_h5ad(file1)
    new = anndata.read_h5ad(file2)

    # Set source labels based on the flag
    if use_basename:
        adata.obs["source"] = os.path.basename(file1)
        new.obs["source"] = os.path.basename(file2)
    else:
        adata.obs["source"] = file1
        new.obs["source"] = file2

    # Filter new data: keep only cells whose label_column values exist in adata
    if label_column in new.obs and label_column in adata.obs:
        cell_mask = new.obs[label_column].isin(adata.obs[label_column])
        new = new[cell_mask].copy()

    # Concatenate the two datasets
    full = adata.concatenate(new)

    # Filter genes with at least 1 count
    sc.pp.filter_genes(full, min_counts=1)

    # Normalize and log-transform
    adata_norm = full.copy()
    sc.pp.normalize_per_cell(adata_norm, counts_per_cell_after=1_000_000)
    sc.pp.log1p(adata_norm)

    # Convert sparse matrix to dense and standardize
    adata_norm.X = adata_norm.X.toarray()
    adata_norm.X -= adata_norm.X.mean(axis=0)
    adata_norm.X /= adata_norm.X.std(axis=0)

    # Compute PCA and store in obsm
    full.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(adata_norm.X)
    
    return full
