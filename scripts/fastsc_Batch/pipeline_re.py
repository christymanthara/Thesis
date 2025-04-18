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
from fast_scBatch import solver


def preprocess_for_fastscbatch(file1, file2, label_column="labels", batch_column="batch_id", use_basename=True):
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

    if label_column in adata1.obs and label_column in adata2.obs:
        adata2 = adata2[adata2.obs[label_column].isin(adata1.obs[label_column])].copy()

    # adata = adata1.concatenate(adata2, batch_key=batch_column)
    adata = anndata.concat([adata1, adata2], label=batch_column, keys=["batch1", "batch2"])
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata

# def torch_quantile_normalize_vector(v1, v2):
#     v1_sorted, v2_sorted = torch.sort(v1), torch.sort(v2)
#     ranks = torch.argsort(torch.argsort(v1))
#     return v2_sorted[ranks]

def torch_quantile_normalize_vector(v1, v2):
    """
    Normalize v1 to the distribution of v2 using quantile normalization.
    """
    # Sort v1 and get ranks (argsort of argsort)
    sort_idx = torch.argsort(v1)
    ranks = torch.argsort(sort_idx)  # gives rank positions
    v2_sorted = torch.sort(v2)[0]
    
    # Make sure ranks is long type for indexing
    return v2_sorted[ranks.long()]


def torch_quantile_normalize_correlation(D, batch_tensor):
    device = D.device
    n = D.shape[0]
    unique_batches = torch.unique(batch_tensor)
    print(f"Unique batches found: {unique_batches.tolist()}")
    
    batch_to_indices = {
        b.item(): (batch_tensor == b).nonzero(as_tuple=True)[0] for b in unique_batches
    }
    for b, idx in batch_to_indices.items():
        print(f"Batch {b} has {len(idx)} samples")

    # reference: largest batch block
    ref_batch = max(batch_to_indices, key=lambda b: len(batch_to_indices[b]))
    ref_idx = batch_to_indices[ref_batch]
    print(f"Reference batch selected: {ref_batch} with {len(ref_idx)} samples")

    ref_block = D[ref_idx][:, ref_idx].flatten()
    ref_sorted = torch.sort(ref_block)[0]
    print("Reference block flattened and sorted")

    # normalize all within and between-batch blocks
    for i in unique_batches:
        i = i.item()
        i_idx = batch_to_indices[i]
        if i != ref_batch:
            print(f"Normalizing within-batch block for batch {i}")
            block = D[i_idx][:, i_idx].flatten()
            D[i_idx][:, i_idx] = torch_quantile_normalize_vector(block, ref_sorted).reshape(len(i_idx), len(i_idx))
        for j in unique_batches:
            j = j.item()
            if i != j:
                j_idx = batch_to_indices[j]
                print(f"Normalizing between-batch block: batch {i} vs batch {j}")
                block = D[i_idx][:, j_idx].flatten()
                D[i_idx][:, j_idx] = torch_quantile_normalize_vector(block, ref_sorted).reshape(len(i_idx), len(j_idx))

    # column-wise normalization
    print("Starting column-wise normalization")
    for i in range(n):
        if i % 100 == 0 or i == n - 1:  # log sparsely for large matrices
            print(f"Normalizing column {i+1}/{n}")
        D[:, i] = torch_quantile_normalize_vector(D[:, i], ref_sorted)

    # symmetrize
    print("Symmetrizing matrix")
    return 0.5 * (D + D.T)



def plot_corr_heatmap(D, title="Correlation Matrix", figsize=(6,5)):
    plt.figure(figsize=figsize)
    sns.heatmap(D.cpu().numpy(), cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()



def run_fastscbatch_pipeline(file1, file2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    # Step 1: Preprocess
    adata = preprocess_for_fastscbatch(file1, file2)
    batch = adata.obs["batch_id"]
    # X_df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
    # X_df = pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
    if not isinstance(adata.X, np.ndarray):
        X_matrix = adata.X.toarray()
    else:
        X_matrix = adata.X
    X_df = pd.DataFrame(X_matrix.T, index=adata.var_names, columns=adata.obs_names)

    batch_df = pd.DataFrame(batch.values, index=adata.obs_names)

    # Step 2: Correlation matrix (raw)
    print("Computing raw correlation matrix...")
    # corr_raw = torch.tensor(np.corrcoef(adata.X.T), dtype=torch.float32, device=device)
    # Ensure dense array
    X_dense = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    corr_raw = torch.tensor(np.corrcoef(X_dense.T), dtype=torch.float32, device=device)


    # Step 3: Quantile normalization
    batch_tensor = torch.tensor(batch.cat.codes.values, dtype=torch.long, device=device)
    print("Applying quantile normalization...")
    corr_corrected = torch_quantile_normalize_correlation(corr_raw.clone(), batch_tensor)

    # Step 4: Visual diagnostics
    plot_corr_heatmap(corr_raw, "Raw Correlation Matrix")
    plot_corr_heatmap(corr_corrected, "Corrected Correlation Matrix (Quantile Normalized)")

    # Step 5: Prepare for solver
    D_df = pd.DataFrame(corr_corrected.cpu().numpy(), index=adata.obs_names, columns=adata.obs_names)
    corrected = solver(
        X=X_df,
        D=D_df,
        batch=batch_df,
        k=10,
        c=10,
        p=0.15,
        EPOCHS=(30, 90, 80),
        lr=(0.002, 0.004, 0.008),
        device=device,
        verbose=True
    )

    # Step 6: Store in AnnData
    adata.obsm["X_fastscbatch"] = corrected.T.loc[adata.obs_names].values
    print("✅ Fast-scBatch pipeline completed!")

    return adata


if __name__ == "__main__":
    # Example usage
    # print_adata_stats("extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad")
    # print_adata_stats("extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad")
    # print_adata_stats("Datasets/baron_2016h.h5ad")
    # adata_corrected = run_fastscbatch_pipeline("extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad", 
    #                                            "extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad")
    adata_corrected = run_fastscbatch_pipeline("Datasets/baron_2016h.h5ad","Datasets/xin_2016.h5ad")
    sc.pp.neighbors(adata_corrected, use_rep="X_fastscbatch")
    sc.tl.umap(adata_corrected)
    sc.pl.umap(adata_corrected, color=["labels", "batch_id"])
