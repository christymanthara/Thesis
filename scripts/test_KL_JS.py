import scanpy as sc
from data_utils.clustering_metrics_KL_JS import compute_kl_divergence, compute_js_divergence
from data_utils.processing import load_and_preprocess_single

def main():
    # === Load your two AnnData files ===
    # adata1 = sc.read("extracted_csv\GSM2230757_human1_umifm_counts_human.h5ad")
    # adata2 = sc.read("extracted_csv\GSM2230758_human2_umifm_counts_human.h5ad")

    adata1 = "extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad"
    adata2 = "extracted_csv/GSM2230762_mouse2_umifm_counts_mouse.h5ad"

    adata1_embedded = load_and_preprocess_single(adata1, use_basename=True)
    adata2_embedded = load_and_preprocess_single(adata2, use_basename=True)
    # === Parameters ===
    # embedding_key = 'X_scVI'   # You can change to 'X_pca', 'X_umap', etc.
    embedding_key = 'X_pca'
    sample_size =   1000        # Number of samples used to estimate density

    # === Compute Divergences ===
    kl = compute_kl_divergence(
        adata1_embedded, adata2_embedded, 
        mode='embedding', 
        embedding_key=embedding_key,
        sample_size=sample_size
    )

    js = compute_js_divergence(
        adata1_embedded, adata2_embedded, 
        mode='embedding', 
        embedding_key=embedding_key,
        sample_size=sample_size
    )

    # === Print Results ===
    print(f"KL Divergence ({embedding_key}): {kl:.4f}")
    print(f"JS Divergence ({embedding_key}): {js:.4f}")

if __name__ == "__main__":
    main()
