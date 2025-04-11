import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import sys
from tqdm import trange

# Ensure fast_scBatch is importable
sys.path.append("../")
import data_utils.fast_scBatch as fast_scBatch

def run_fast_scbatch_pipeline(
    input_h5ad_path,
    corr_csv_path,
    output_h5ad_path,
    p=0.3,
    k=50,
    lr=(0.0002, 0.0001, 0.0003),
    EPOCHS=(0, 0, 500),
    verbose=True,
    plot_umap=True
):
    # Load the data
    cell = anndata.read_h5ad(input_h5ad_path)
    batch = cell.obs[["batch"]].copy()
    ctype = cell.obs[["celltype"]].copy()
    cells = cell.to_df().T

    # Load correlation matrix
    corr = pd.read_csv(corr_csv_path, index_col=0)
    corr.columns = cells.columns
    corr.index = cells.columns

    # Get shape
    p_cells, n_cells = cells.shape

    # Run the solver
    res = fast_scBatch.solver(
        cells, corr, batch, p=p, k=k,
        lr=lr, EPOCHS=EPOCHS, verbose=verbose
    )

    # Create AnnData object
    adata = anndata.AnnData(X=res.T, obs=ceùll.obs, var=cell.var)
    adata.write(output_h5ad_path)
    print(adata)

    # Optional UMAP visualization
    if plot_umap:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=["batch", "celltype"], wspace=0.5)

    return adata

