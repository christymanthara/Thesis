import scanpy as sc
import os
import sys
from data_utils.processing import load_and_preprocess_single

# Add the directory containing clustering_metrics_KL_JS.py to the path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from clustering_metrics_KL_JS import compute_kl_divergence, compute_js_divergence

# If you're updating the functions directly, import them as usual:
from data_utils.clustering_metrics_KL_JS import compute_kl_divergence, compute_js_divergence

def main():
    # === Load your two AnnData files ===
    adata1 = "extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad"
    adata2 = "extracted_csv/GSM2230762_mouse2_umifm_counts_mouse.h5ad"

    print(f"Loading and preprocessing {adata1}")
    adata1_embedded = load_and_preprocess_single(adata1, use_basename=True)
    
    print(f"Loading and preprocessing {adata2}")
    adata2_embedded = load_and_preprocess_single(adata2, use_basename=True)
    
    # === Parameters ===
    embedding_key = 'X_pca'
    sample_size = 1000        # Number of samples used to estimate density
    
    # Try multiple bandwidth values
    bandwidths = [0.01, 0.1, 0.5, 1.0]
    
    print("\nComputing divergences with different bandwidths:")
    print("-" * 50)
    
    for bandwidth in bandwidths:
        print(f"\nBandwidth: {bandwidth}")
        
        # === Compute Divergences ===
        kl = compute_kl_divergence(
            adata1_embedded, adata2_embedded, 
            mode='embedding', 
            embedding_key=embedding_key,
            sample_size=sample_size,
            bandwidth=bandwidth
        )

        js = compute_js_divergence(
            adata1_embedded, adata2_embedded, 
            mode='embedding', 
            embedding_key=embedding_key,
            sample_size=sample_size,
            bandwidth=bandwidth
        )

        # === Print Results ===
        print(f"KL Divergence ({embedding_key}): {kl:.4f}")
        print(f"JS Divergence ({embedding_key}): {js:.4f}")

if __name__ == "__main__":
    main()