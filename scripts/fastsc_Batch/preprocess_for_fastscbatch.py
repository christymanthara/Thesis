import anndata
import scanpy as sc
import pandas as pd
import os
import numpy as np

def preprocess_for_fastscbatch(file1, file2, label_column="labels", batch_column="batch_id", use_basename=True):
    """
    Preprocess two AnnData files for fast-scBatch:
    - Load data, label sources, filter by common labels
    - Concatenate, filter genes, normalize, log-transform
    - Return the preprocessed AnnData (no PCA, no standardization)
    """
    # Load the datasets
    adata1 = anndata.read_h5ad(file1)
    adata2 = anndata.read_h5ad(file2)

    def extract_filename(path):
        return os.path.basename(path).rsplit('.h5ad', 1)[0]

    if use_basename:
        adata1.obs["source"] = pd.Categorical([extract_filename(file1)] * adata1.n_obs)
        adata2.obs["source"] = pd.Categorical([extract_filename(file2)] * adata2.n_obs)
    else:
        adata1.obs["source"] = pd.Categorical([file1] * adata1.n_obs)
        adata2.obs["source"] = pd.Categorical([file2] * adata2.n_obs)

    # Filter adata2: keep only cells whose label_column values exist in adata1
    if label_column in adata1.obs and label_column in adata2.obs:
        adata2 = adata2[adata2.obs[label_column].isin(adata1.obs[label_column])].copy()

    # Concatenate the datasets (maintains batch/source info)
    adata = adata1.concatenate(adata2, batch_key=batch_column)

    # Filter genes expressed in at least 1 cell
    sc.pp.filter_genes(adata, min_counts=1)

    # Normalize per cell and log-transform
    adata.X = adata.X.astype(float)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    return adata
