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

@torch.jit.script
def faster_quantile_normalize_vector(v1, v2):
    """
    Normalize v1 to the distribution of v2 using quantile normalization.
    Handles different sized vectors safely.
    """
    # Sort v1 and get ranks
    v1_sorted, sort_indices = torch.sort(v1)
    ranks = torch.argsort(sort_indices)
    
    # Get quantiles from v2 based on relative positions
    if len(v1) != len(v2):
        # Need to interpolate v2 values to match v1 length
        v2_sorted = torch.sort(v2)[0]
        
        # Create interpolated values
        if len(v2) > 1:
            indices = torch.linspace(0, len(v2)-1, len(v1), device=v1.device)
            indices_floor = indices.floor().long()
            indices_ceil = indices.ceil().long()
            weights_ceil = indices - indices_floor
            weights_floor = 1 - weights_ceil
            
            # Interpolate
            v2_interp = weights_floor * v2_sorted[indices_floor] + weights_ceil * v2_sorted[indices_ceil]
        else:
            # If v2 has only one element, use it for all positions
            v2_interp = v2_sorted.repeat(len(v1))
    else:
        # Same length, no interpolation needed
        v2_interp = torch.sort(v2)[0]
    
    # Return normalized values
    return v2_interp[ranks]

@torch.compile
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


# Modified optimization function that handles blocks of different sizes
@torch.compile
def optimized_quantile_normalize_correlation(D, batch_tensor):
    device = D.device
    n = D.shape[0]
    unique_batches = torch.unique(batch_tensor)
    print(f"Unique batches found: {unique_batches.tolist()}")
    
    # Create batch indices dictionary
    batch_to_indices = {
        b.item(): (batch_tensor == b).nonzero(as_tuple=True)[0] for b in unique_batches
    }
    for b, idx in batch_to_indices.items():
        print(f"Batch {b} has {len(idx)} samples")
    
    # Select reference batch (largest batch)
    ref_batch = max(batch_to_indices, key=lambda b: len(batch_to_indices[b]))
    ref_idx = batch_to_indices[ref_batch]
    print(f"Reference batch selected: {ref_batch} with {len(ref_idx)} samples")
    
    ref_block = D[ref_idx][:, ref_idx].flatten()
    ref_sorted = torch.sort(ref_block)[0]
    print("Reference block flattened and sorted")
    
    # Pre-allocate a tensor for the normalized matrix
    D_normalized = D.clone()
    
    # Process blocks more efficiently with batch processing
    print("Processing batch blocks...")
    batch_progress = 0
    total_batches = len(unique_batches) * (len(unique_batches) + 1) // 2  # Triangular number
    
    for i_idx, i in enumerate(unique_batches):
        i = i.item()
        i_indices = batch_to_indices[i]
        
        # Process within-batch block
        block = D[i_indices][:, i_indices].flatten()
        D_normalized[i_indices][:, i_indices] = faster_quantile_normalize_vector(block, ref_sorted).reshape(len(i_indices), len(i_indices))
        batch_progress += 1
        print(f"Progress: {batch_progress}/{total_batches} - Within batch {i}")
        
        # Process between-batch blocks - only upper triangle to avoid redundancy
        for j in unique_batches[i_idx+1:]:
            j = j.item()
            j_indices = batch_to_indices[j]
            
            block = D[i_indices][:, j_indices].flatten()
            normalized_block = faster_quantile_normalize_vector(block, ref_sorted)
            reshaped_block = normalized_block.reshape(len(i_indices), len(j_indices))
            
            # Set both blocks to maintain symmetry
            D_normalized[i_indices][:, j_indices] = reshaped_block
            D_normalized[j_indices][:, i_indices] = reshaped_block.T
            
            batch_progress += 1
            print(f"Progress: {batch_progress}/{total_batches} - Between batches {i} and {j}")
    
    # Ensure perfect symmetry
    print("Symmetrizing matrix")
    return 0.5 * (D_normalized + D_normalized.T)


@torch.compile
def memory_efficient_quantile_normalize(D, batch_tensor, chunk_size=1000):
    """
    Memory-efficient implementation of quantile normalization
    - Processes correlation matrix in chunks
    - Uses float32 instead of float64
    - Avoids large temporary arrays
    """
    device = D.device
    n = D.shape[0]
    D = D.to(torch.float32)  # Convert to float32 to save memory
    unique_batches = torch.unique(batch_tensor)
    
    # Create batch indices dictionary
    batch_to_indices = {
        b.item(): (batch_tensor == b).nonzero(as_tuple=True)[0] for b in unique_batches
    }
    
    # Select reference batch (largest batch)
    ref_batch = max(batch_to_indices, key=lambda b: len(batch_to_indices[b]))
    ref_idx = batch_to_indices[ref_batch]
    
    # Create reference distribution in chunks
    print("Creating reference distribution...")
    ref_values = []
    for i in range(0, len(ref_idx), chunk_size):
        end_i = min(i + chunk_size, len(ref_idx))
        for j in range(0, len(ref_idx), chunk_size):
            end_j = min(j + chunk_size, len(ref_idx))
            chunk = D[ref_idx[i:end_i]][:, ref_idx[j:end_j]].flatten()
            ref_values.append(chunk)
    
    ref_values = torch.cat(ref_values)
    ref_sorted = torch.sort(ref_values)[0]
    del ref_values  # Free memory
    
    # Process matrix in small chunks
    print("Processing matrix in chunks...")
    for b1_idx, b1 in enumerate(unique_batches):
        b1 = b1.item()
        indices1 = batch_to_indices[b1]
        
        for b2_idx, b2 in enumerate(unique_batches):
            if b2_idx < b1_idx:  # Skip lower triangular part (will mirror later)
                continue
                
            b2 = b2.item()
            indices2 = batch_to_indices[b2]
            print(f"Processing batch {b1} vs {b2}")
            
            # Process this batch pair in small chunks
            for i in range(0, len(indices1), chunk_size):
                end_i = min(i + chunk_size, len(indices1))
                for j in range(0, len(indices2), chunk_size):
                    end_j = min(j + chunk_size, len(indices2))
                    
                    # Get chunk indices
                    i_indices = indices1[i:end_i]
                    j_indices = indices2[j:end_j]
                    
                    # Process chunk
                    chunk = D[i_indices][:, j_indices].clone()
                    flat_chunk = chunk.flatten()
                    
                    # Normalize chunk
                    sort_indices = torch.argsort(flat_chunk)
                    interp_indices = torch.linspace(0, len(ref_sorted)-1, len(flat_chunk), device=device).long()
                    normalized_values = ref_sorted[interp_indices]
                    
                    # Place values back according to original order
                    result_chunk = torch.empty_like(flat_chunk)
                    result_chunk[sort_indices] = normalized_values
                    D[i_indices][:, j_indices] = result_chunk.reshape(len(i_indices), len(j_indices))
                    
                    # Set symmetric part if this is not a diagonal block
                    if b1 != b2:
                        D[j_indices][:, i_indices] = result_chunk.reshape(len(i_indices), len(j_indices)).T
    
    # Final symmetrize step
    print("Symmetrizing matrix...")
    # Process symmetrization in chunks to avoid large temporary arrays
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(i, n, chunk_size):  # Start from i to only process upper triangle
            end_j = min(j + chunk_size, n)
            if i == j:
                # Diagonal block - already symmetric
                continue
            else:
                # Average the upper and lower triangular parts
                upper = D[i:end_i, j:end_j]
                lower = D[j:end_j, i:end_i].T
                avg = 0.5 * (upper + lower)
                D[i:end_i, j:end_j] = avg
                D[j:end_j, i:end_i] = avg.T
    
    return D


# def plot_corr_heatmap(D, title="Correlation Matrix", figsize=(6,5)):
#     plt.figure(figsize=figsize)
#     sns.heatmap(D.cpu().numpy(), cmap='coolwarm', xticklabels=False, yticklabels=False)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()

def plot_corr_heatmap(D, title="Correlation Matrix", figsize=(6,5), max_size=1000):
    """
    Memory-efficient heatmap plotting that downsamples large matrices
    
    Parameters:
    - D: correlation matrix tensor
    - title: plot title
    - figsize: figure size
    - max_size: maximum size for visualization (will downsample if larger)
    """
    # Convert to CPU and numpy
    D_np = D.cpu().numpy()
    
    # Check if downsampling is needed
    n = D_np.shape[0]
    if n > max_size:
        print(f"Matrix too large ({n}x{n}), downsampling to {max_size}x{max_size} for visualization")
        # Calculate stride for downsampling
        stride = n // max_size
        D_small = D_np[::stride, ::stride]
        plt.figure(figsize=figsize)
        sns.heatmap(D_small, cmap='coolwarm', xticklabels=False, yticklabels=False)
        plt.title(f"{title} (downsampled {stride}x)")
    else:
        plt.figure(figsize=figsize)
        sns.heatmap(D_np, cmap='coolwarm', xticklabels=False, yticklabels=False)
        plt.title(title)
    
    plt.tight_layout()
    plt.show()
    
    # Free memory
    del D_np
    if 'D_small' in locals():
        del D_small
    plt.close()


# def run_fastscbatch_pipeline(file1, file2):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Running on:", device)

#     # Step 1: Preprocess
#     adata = preprocess_for_fastscbatch(file1, file2)
#     batch = adata.obs["batch_id"]
#     # X_df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
#     # X_df = pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
#     if not isinstance(adata.X, np.ndarray):
#         X_matrix = adata.X.toarray()
#     else:
#         X_matrix = adata.X
#     X_df = pd.DataFrame(X_matrix.T, index=adata.var_names, columns=adata.obs_names)

#     batch_df = pd.DataFrame(batch.values, index=adata.obs_names)

#     # Step 2: Correlation matrix (raw)
#     # print("Computing raw correlation matrix...")
#     # # corr_raw = torch.tensor(np.corrcoef(adata.X.T), dtype=torch.float32, device=device)
#     # # Ensure dense array
#     # X_dense = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
#     # corr_raw = torch.tensor(np.corrcoef(X_dense.T), dtype=torch.float32, device=device)

#     # Step 2: Correlation matrix calculation
#     print("Computing raw correlation matrix...")
#     # Use float32 to save memory
#     X_dense = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
#     X_dense = X_dense.astype(np.float32)  # Use float32 to save memory
    
#     # Calculate correlation in chunks if matrix is large
#     n_features = X_dense.shape[1]
#     if n_features > 10000:  # Arbitrary threshold
#         print(f"Large feature space detected ({n_features}), calculating correlation in chunks")
#         # Calculate correlation in chunks to save memory
#         corr_raw = calculate_correlation_in_chunks(X_dense.T, chunk_size=2000)
#     else:
#         corr_raw = np.corrcoef(X_dense.T)
    
#     # Move to device after calculation
#     corr_raw = torch.tensor(corr_raw, dtype=torch.float32, device=device)
    
#     # Free memory
#     del X_dense
#     torch.cuda.empty_cache() if device == "cuda" else None


#     # Step 3: Quantile normalization
#     batch_tensor = torch.tensor(batch.cat.codes.values, dtype=torch.long, device=device)
#     print("Applying quantile normalization...")
#     # corr_corrected = torch_quantile_normalize_correlation(corr_raw.clone(), batch_tensor)
#     # corr_corrected = optimized_quantile_normalize_correlation(corr_raw.clone(), batch_tensor)
#     corr_corrected = memory_efficient_quantile_normalize(corr_raw.clone(), batch_tensor, chunk_size=500)

#     # Step 4: Visual diagnostics
#     plot_corr_heatmap(corr_raw, "Raw Correlation Matrix")
#     plot_corr_heatmap(corr_corrected, "Corrected Correlation Matrix (Quantile Normalized)")

#     # Step 5: Prepare for solver
#     D_df = pd.DataFrame(corr_corrected.cpu().numpy(), index=adata.obs_names, columns=adata.obs_names)
#     corrected = solver(
#         X=X_df,
#         D=D_df,
#         batch=batch_df,
#         k=10,
#         c=10,
#         p=0.15,
#         EPOCHS=(30, 90, 80),
#         lr=(0.002, 0.004, 0.008),
#         device=device,
#         verbose=True
#     )

#     # Step 6: Store in AnnData
#     adata.obsm["X_fastscbatch"] = corrected.T.loc[adata.obs_names].values
#     print("✅ Fast-scBatch pipeline completed!")

#     return adata

def run_fastscbatch_pipeline(file1, file2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)
    
    # Step 1: Preprocess
    adata = preprocess_for_fastscbatch(file1, file2)
    batch = adata.obs["batch_id"]
    
    # Free memory after preprocessing
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Step 2: Correlation matrix calculation
    print("Computing raw correlation matrix...")
    # Use float32 to save memory
    X_dense = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    X_dense = X_dense.astype(np.float32)  # Use float32 to save memory
    
    # Calculate correlation in chunks if matrix is large
    n_features = X_dense.shape[1]
    if n_features > 10000:  # Arbitrary threshold
        print(f"Large feature space detected ({n_features}), calculating correlation in chunks")
        # Calculate correlation in chunks to save memory
        corr_raw = calculate_correlation_in_chunks(X_dense.T, chunk_size=2000)
    else:
        corr_raw = np.corrcoef(X_dense.T)
    
    # Move to device after calculation
    corr_raw = torch.tensor(corr_raw, dtype=torch.float32, device=device)
    
    # Free memory
    del X_dense
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Step 3: Quantile normalization
    batch_tensor = torch.tensor(batch.cat.codes.values, dtype=torch.long, device=device)
    print("Applying quantile normalization...")
    corr_corrected = memory_efficient_quantile_normalize(corr_raw.clone(), batch_tensor, chunk_size=500)
    
    # Free memory
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Step 4: Visual diagnostics with downsampling for large matrices
    plot_corr_heatmap(corr_raw, "Raw Correlation Matrix", max_size=1000)
    plot_corr_heatmap(corr_corrected, "Corrected Correlation Matrix (Quantile Normalized)", max_size=1000)
    
    # Free more memory
    del corr_raw
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Step 5: Prepare for solver
    D_df = pd.DataFrame(corr_corrected.cpu().numpy(), index=adata.obs_names, columns=adata.obs_names)
    del corr_corrected  # Free memory immediately
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # For X_df, prevent duplicate computation of X.T
    if not isinstance(adata.X, np.ndarray):
        X_matrix = adata.X.toarray()
    else:
        X_matrix = adata.X
    X_df = pd.DataFrame(X_matrix.T, index=adata.var_names, columns=adata.obs_names)
    del X_matrix  # Free memory
    
    batch_df = pd.DataFrame(batch.values, index=adata.obs_names)
    
    # Step 6: Run solver
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
    
    # Step 7: Store in AnnData
    adata.obsm["X_fastscbatch"] = corrected.T.loc[adata.obs_names].values
    print("✅ Fast-scBatch pipeline completed!")
    
    return adata

# Helper function for chunked correlation calculation
def calculate_correlation_in_chunks(X, chunk_size=2000):
    """Calculate correlation matrix in chunks to save memory"""
    n = X.shape[0]
    corr = np.zeros((n, n), dtype=np.float32)
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        print(f"Computing correlation chunk {i//chunk_size + 1}/{(n+chunk_size-1)//chunk_size}")
        
        # Process upper triangular part of the correlation matrix
        for j in range(i, n, chunk_size):
            end_j = min(j + chunk_size, n)
            
            # Calculate correlation for this chunk
            X_i = X[i:end_i]
            X_j = X[j:end_j]
            
            # Normalize data for correlation calculation
            X_i_norm = (X_i - X_i.mean(axis=1, keepdims=True)) / X_i.std(axis=1, keepdims=True)
            X_j_norm = (X_j - X_j.mean(axis=1, keepdims=True)) / X_j.std(axis=1, keepdims=True)
            
            # Calculate correlation
            chunk_corr = X_i_norm @ X_j_norm.T / X_i_norm.shape[1]
            
            # Store in the correlation matrix
            corr[i:end_i, j:end_j] = chunk_corr
            
            # Mirror for lower triangular if not on diagonal
            if i != j:
                corr[j:end_j, i:end_i] = chunk_corr.T
    
    return corr

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
