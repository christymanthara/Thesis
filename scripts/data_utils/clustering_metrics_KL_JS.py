import numpy as np
from scipy.special import rel_entr
from sklearn.neighbors import KernelDensity

def compute_kl_divergence(
    adata1, 
    adata2, 
    mode='gene',              # 'gene', 'embedding', or 'cluster'
    gene=None,                # Required if mode == 'gene'
    embedding_key='X_scVI',   # Used if mode == 'embedding'
    cluster_key='cell_type',  # Used if mode == 'cluster'
    bins=50,
    sample_size=100,
    bandwidth=0.1,            # Added bandwidth parameter for KDE
    epsilon=1e-10
):
    if mode == 'gene':
        if gene is None:
            raise ValueError("You must specify a gene name when mode='gene'")

        x1 = adata1[:, gene].X.A.flatten() if hasattr(adata1[:, gene].X, "A") else adata1[:, gene].X.flatten()
        x2 = adata2[:, gene].X.A.flatten() if hasattr(adata2[:, gene].X, "A") else adata2[:, gene].X.flatten()

        # Create histograms (PDF)
        max_val = max(x1.max(), x2.max())
        hist1, _ = np.histogram(x1, bins=bins, range=(0, max_val), density=True)
        hist2, _ = np.histogram(x2, bins=bins, range=(0, max_val), density=True)

        P = np.maximum(hist1, epsilon)
        Q = np.maximum(hist2, epsilon)

    elif mode == 'embedding':
        emb1 = adata1.obsm[embedding_key]
        emb2 = adata2.obsm[embedding_key]
        
        # Use samples from both datasets for evaluation
        if sample_size > emb1.shape[0]:
            sample_size = emb1.shape[0]
        if sample_size > emb2.shape[0]:
            sample_size = emb2.shape[0]
            
        # Sample points to evaluate density at
        np.random.seed(42)  # For reproducibility
        idx1 = np.random.choice(emb1.shape[0], size=sample_size//2, replace=False)
        idx2 = np.random.choice(emb2.shape[0], size=sample_size//2, replace=False)
        
        sample_points1 = emb1[idx1]
        sample_points2 = emb2[idx2]
        sample_points = np.vstack([sample_points1, sample_points2])
        
        # Fit KDEs with specified bandwidth
        kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(emb1)
        kde2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(emb2)
        
        # Get log densities
        log_p1 = kde1.score_samples(sample_points)
        log_p2 = kde2.score_samples(sample_points)
        
        # Convert to probabilities
        P = np.exp(log_p1)
        Q = np.exp(log_p2)
        
        # Ensure no zeros
        P = np.maximum(P, epsilon)
        Q = np.maximum(Q, epsilon)

    elif mode == 'cluster':
        counts1 = adata1.obs[cluster_key].value_counts(normalize=True).sort_index()
        counts2 = adata2.obs[cluster_key].value_counts(normalize=True).sort_index()

        all_types = sorted(set(counts1.index) | set(counts2.index))
        P = np.array([counts1.get(ct, 0) for ct in all_types])
        Q = np.array([counts2.get(ct, 0) for ct in all_types])

        P = np.maximum(P, epsilon)
        Q = np.maximum(Q, epsilon)

    else:
        raise ValueError("Invalid mode. Choose from 'gene', 'embedding', or 'cluster'.")

    # Normalize and compute KL divergence
    P /= P.sum()
    Q /= Q.sum()
    kl_div = np.sum(rel_entr(P, Q))

    return kl_div

def compute_js_divergence(
    adata1, 
    adata2, 
    mode='gene',              # 'gene', 'embedding', or 'cluster'
    gene=None,                # Required if mode == 'gene'
    embedding_key='X_scVI',   # Used if mode == 'embedding'
    cluster_key='cell_type',  # Used if mode == 'cluster'
    bins=50,
    sample_size=100,
    bandwidth=0.1,            # Added bandwidth parameter for KDE
    epsilon=1e-10
):
    if mode == 'gene':
        if gene is None:
            raise ValueError("You must specify a gene name when mode='gene'")

        x1 = adata1[:, gene].X.A.flatten() if hasattr(adata1[:, gene].X, "A") else adata1[:, gene].X.flatten()
        x2 = adata2[:, gene].X.A.flatten() if hasattr(adata2[:, gene].X, "A") else adata2[:, gene].X.flatten()

        max_val = max(x1.max(), x2.max())
        hist1, _ = np.histogram(x1, bins=bins, range=(0, max_val), density=True)
        hist2, _ = np.histogram(x2, bins=bins, range=(0, max_val), density=True)

        P = np.maximum(hist1, epsilon)
        Q = np.maximum(hist2, epsilon)

    elif mode == 'embedding':
        emb1 = adata1.obsm[embedding_key]
        emb2 = adata2.obsm[embedding_key]
        
        # Use samples from both datasets for evaluation
        if sample_size > emb1.shape[0]:
            sample_size = emb1.shape[0]
        if sample_size > emb2.shape[0]:
            sample_size = emb2.shape[0]
            
        # Sample points to evaluate density at
        np.random.seed(42)  # For reproducibility
        idx1 = np.random.choice(emb1.shape[0], size=sample_size//2, replace=False)
        idx2 = np.random.choice(emb2.shape[0], size=sample_size//2, replace=False)
        
        sample_points1 = emb1[idx1]
        sample_points2 = emb2[idx2]
        sample_points = np.vstack([sample_points1, sample_points2])
        
        # Fit KDEs with specified bandwidth
        kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(emb1)
        kde2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(emb2)
        
        # Get log densities
        log_p1 = kde1.score_samples(sample_points)
        log_p2 = kde2.score_samples(sample_points)
        
        # Convert to probabilities
        P = np.exp(log_p1)
        Q = np.exp(log_p2)
        
        # Ensure no zeros
        P = np.maximum(P, epsilon)
        Q = np.maximum(Q, epsilon)

    elif mode == 'cluster':
        counts1 = adata1.obs[cluster_key].value_counts(normalize=True).sort_index()
        counts2 = adata2.obs[cluster_key].value_counts(normalize=True).sort_index()

        all_types = sorted(set(counts1.index) | set(counts2.index))
        P = np.array([counts1.get(ct, 0) for ct in all_types])
        Q = np.array([counts2.get(ct, 0) for ct in all_types])

        P = np.maximum(P, epsilon)
        Q = np.maximum(Q, epsilon)

    else:
        raise ValueError("Invalid mode. Choose from 'gene', 'embedding', or 'cluster'.")

    # Normalize
    P /= P.sum()
    Q /= Q.sum()

    M = 0.5 * (P + Q)

    js_div = 0.5 * np.sum(rel_entr(P, M)) + 0.5 * np.sum(rel_entr(Q, M))

    return js_div