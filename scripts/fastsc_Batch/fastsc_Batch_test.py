import anndata
import torch
import pandas as pd
import numpy as np
from tqdm import trange
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils.fast_scBatch import solver  # Assuming solver is defined in fast_scBatch.py


def read_h5ad_data(file_path):
    """
    Reads an h5ad file and returns expression matrix, batch labels, and AnnData object.

    Parameters:
        file_path (str): Path to the .h5ad file.

    Returns:
        rawdat (ndarray): Gene expression matrix (cells x genes).
        bat (Series): Batch labels.
        adata (AnnData): Full AnnData object.
    """
    adata = anndata.read_h5ad(file_path)
    rawdat = adata.X
    bat = adata.obs["Batch"].copy().iloc[:, 0] if isinstance(adata.obs["Batch"], pd.DataFrame) else adata.obs["Batch"]
    return rawdat, bat, adata


def compute_reference_corr(rawdat, bat):
    """
    Computes modified correlation matrix by aligning batch-specific correlations to a reference batch.

    Parameters:
        rawdat (ndarray): Gene expression matrix (cells x genes).
        bat (Series): Batch labels.

    Returns:
        corr (Tensor): Transformed correlation matrix.
    """
    rawcorr = torch.corrcoef(torch.tensor(rawdat))

    mx = bat.value_counts().idxmax()
    ref_block = rawdat[bat == mx]
    ref_corr = torch.corrcoef(torch.tensor(ref_block))
    vec = ref_corr.reshape(-1).sort()[0]
    dattype = bat.unique()

    corr = rawcorr.clone()
    for i in dattype:
        for j in dattype:
            if i == mx and j == mx:
                continue
            block = corr[bat == i][:, bat == j]
            block_ = block.reshape(-1)
            idx = block_.argsort()
            pos = torch.zeros_like(idx, dtype=torch.float)
            pos[idx] = torch.arange(len(idx), dtype=torch.float)
            pos = pos / len(block_) * len(vec)
            pos = torch.maximum(pos.int() - 1, torch.zeros_like(pos, dtype=torch.int))
            block_ = vec[pos]
            block_ = block_.reshape(block.shape)
            msk = (torch.tensor(bat == i, dtype=torch.bool).unsqueeze(1) &
                   torch.tensor(bat == j, dtype=torch.bool).unsqueeze(0))
            corr.masked_scatter_(msk, block_)

    # Smooth the matrix
    for _ in range(1):
        batmsk = {i: (bat != i).to_numpy() for i in dattype}
        for i in trange(len(corr)):
            curbat = batmsk[bat[i]]
            block = corr[i, curbat]
            idx = block.argsort().int()
            pos = torch.zeros_like(idx, dtype=torch.float)
            pos[idx] = torch.arange(len(idx), dtype=torch.float)
            pos = pos / len(block) * len(vec)
            pos = torch.maximum(pos.int() - 1, torch.zeros_like(pos, dtype=torch.int))
            block = vec[pos]
            msk = torch.zeros_like(corr, dtype=torch.bool)
            msk[i] |= torch.tensor(curbat, dtype=torch.bool)
            corr.masked_scatter_(msk, block)

    # Symmetrize
    corr = (corr + corr.T) / 2
    return corr


def save_correlation_matrix(corr, out_path):
    """
    Saves correlation matrix to a CSV file.

    Parameters:
        corr (Tensor): Correlation matrix.
        out_path (str): Output path for CSV file.
    """
    pd.DataFrame(corr.numpy()).to_csv(out_path)


def run_fastscbatch_pipeline(h5ad_path, corr_path, output_path, solver_fn):
    """
    Runs the full FastSCBatch correction pipeline.

    Parameters:
        h5ad_path (str): Path to input .h5ad file.
        corr_path (str): Path to correlation .csv file.
        output_path (str): Path to save corrected .h5ad file.
        solver_fn (callable): FastSCBatch solver function.
    """
    cell = anndata.read_h5ad(h5ad_path)
    batch = cell.obs[["Batch"]].copy()
    ctype = cell.obs[["Group"]].copy() if "Group" in cell.obs else None
    cells = cell.to_df().T

    corr = pd.read_csv(corr_path, index_col=0)
    corr.columns = cells.columns
    corr.index = cells.columns

    p, n = cells.shape
    res = solver_fn(
        cells, corr, batch, p=0.3, k=50,
        lr=(0.0002, 0.0001, 0.0003), EPOCHS=(0, 0, 500), verbose=True
    )

    adata = anndata.AnnData(X=res.T, obs=cell.obs, var=cell.var)
    adata.write(output_path)


# EXAMPLE USAGE:
# rawdat, bat, adata = read_h5ad_data("sample.h5ad")
# corr = compute_reference_corr(rawdat, bat)
# save_correlation_matrix(corr, "sample_corr.csv")
# run_fastscbatch_pipeline("sample.h5ad", "sample_corr.csv", "output_corrected.h5ad", fast_scBatch.solver)
if __name__ == "__main__":
    # Example usage of the functions
    h5ad_path = "realdata/pancreas/converted_data.h5ad"
    corr_path = "sample_corr.csv"
    output_path = "output_corrected.h5ad"

    rawdat, bat, adata = read_h5ad_data(h5ad_path)
    corr = compute_reference_corr(rawdat, bat)
    save_correlation_matrix(corr, corr_path)
    run_fastscbatch_pipeline(h5ad_path, corr_path, output_path, solver)  # Assuming solver is defined elsewhere