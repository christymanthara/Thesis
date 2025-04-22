import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from scipy.stats import rankdata



# Step 1: Preprocess your AnnData
adata = preprocess_for_fastscbatch("file1.h5ad", "file2.h5ad")

# Step 2: Extract data matrix and batch
X_df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
batch_df = pd.DataFrame(adata.obs['batch_id'].values, index=adata.obs_names)

# Step 3 (revised): Correlation matrix + Quantile normalization
raw_corr = pd.DataFrame(np.corrcoef(adata.X.T), index=adata.obs_names, columns=adata.obs_names)
batch_series = adata.obs["batch_id"]
corrected_corr = quantile_normalize_correlation_matrix(raw_corr, batch_series)

# (Optional) Apply quantile normalization on D_df here to fully match paper

# Step 4: Run fast-scBatch
corrected_data = solver(X=X_df, D=corrected_corr,
    batch=batch_df,
    k=10,           # PCA components per batch
    c=10,           # number of clusters per batch
    p=0.15,         # sampling ratio
    EPOCHS=(30, 90, 80),
    lr=(0.002, 0.004, 0.008),
    corr_method="pearson",
    cluster_method="spectral",
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True
)

# Step 5: Add corrected matrix back to AnnData
adata.obsm["X_fastscbatch"] = corrected_matrix.T.loc[adata.obs_names].values
