import numpy as np
import pandas as pd
from scipy.stats import rankdata

def quantile_normalize_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Aligns v1 to the distribution of v2 via quantile normalization.
    Equivalent to φ(v1, v2).
    """
    sorter = np.argsort(v1)
    ranks = rankdata(v1, method="ordinal") - 1  # ranks start at 0
    v2_sorted = np.sort(v2)
    normalized = np.zeros_like(v1)
    normalized[sorter] = v2_sorted[ranks]
    return normalized

def quantile_normalize_correlation_matrix(D: pd.DataFrame, batch_labels: pd.Series) -> pd.DataFrame:
    """
    Applies quantile normalization across batch blocks in the correlation matrix D.
    """
    cells = D.index
    D = D.copy()
    batches = batch_labels.loc[cells].values
    unique_batches = np.unique(batches)

    # Group cells by batch
    batch_to_indices = {b: np.where(batches == b)[0] for b in unique_batches}
    
    # Step 1: Choose the largest batch as reference
    ref_batch = max(batch_to_indices, key=lambda b: len(batch_to_indices[b]))
    ref_idx = batch_to_indices[ref_batch]
    ref_block = D.values[np.ix_(ref_idx, ref_idx)].flatten()
    ref_sorted = np.sort(ref_block)

    # Step 2: Block-wise normalization
    for b in unique_batches:
        idx = batch_to_indices[b]
        if b != ref_batch:
            block = D.values[np.ix_(idx, idx)].flatten()
            normalized = quantile_normalize_vector(block, ref_sorted)
            D.values[np.ix_(idx, idx)] = normalized.reshape(len(idx), len(idx))

        for b2 in unique_batches:
            if b != b2:
                idx2 = batch_to_indices[b2]
                between_block = D.values[np.ix_(idx, idx2)].flatten()
                normalized = quantile_normalize_vector(between_block, ref_sorted)
                D.values[np.ix_(idx, idx2)] = normalized.reshape(len(idx), len(idx2))

    # Step 3: Cell-wise normalization (columns)
    for i in range(D.shape[1]):
        col = D.iloc[:, i].values
        normalized = quantile_normalize_vector(col, ref_sorted)
        D.iloc[:, i] = normalized

    # Step 4: Symmetrization
    D_sym = 0.5 * (D.values + D.values.T)
    return pd.DataFrame(D_sym, index=D.index, columns=D.columns)
