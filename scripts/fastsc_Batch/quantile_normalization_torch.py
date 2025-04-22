def torch_quantile_normalize_vector(v1, v2):
    v1_sorted, v2_sorted = torch.sort(v1), torch.sort(v2)
    ranks = torch.argsort(torch.argsort(v1))
    return v2_sorted[ranks]

def torch_quantile_normalize_correlation(D, batch_tensor):
    device = D.device
    n = D.shape[0]
    unique_batches = torch.unique(batch_tensor)
    batch_to_indices = {b.item(): (batch_tensor == b).nonzero(as_tuple=True)[0] for b in unique_batches}

    # reference: largest batch block
    ref_batch = max(batch_to_indices, key=lambda b: len(batch_to_indices[b]))
    ref_idx = batch_to_indices[ref_batch]
    ref_block = D[ref_idx][:, ref_idx].flatten()
    ref_sorted = torch.sort(ref_block)[0]

    # normalize all within and between-batch blocks
    for i in unique_batches:
        i_idx = batch_to_indices[i.item()]
        if i != ref_batch:
            block = D[i_idx][:, i_idx].flatten()
            D[i_idx][:, i_idx] = torch_quantile_normalize_vector(block, ref_sorted).reshape(len(i_idx), len(i_idx))
        for j in unique_batches:
            if i != j:
                j_idx = batch_to_indices[j.item()]
                block = D[i_idx][:, j_idx].flatten()
                D[i_idx][:, j_idx] = torch_quantile_normalize_vector(block, ref_sorted).reshape(len(i_idx), len(j_idx))

    # column-wise normalization
    for i in range(n):
        D[:, i] = torch_quantile_normalize_vector(D[:, i], ref_sorted)

    # symmetrize
    return 0.5 * (D + D.T)
