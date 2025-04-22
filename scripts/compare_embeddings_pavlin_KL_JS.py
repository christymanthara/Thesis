import openTSNE
import scanpy as sc
import matplotlib.pyplot as plt
from data_utils.processing import load_and_preprocess_single
from data_utils.clustering_metrics_KL_JS import compute_kl_divergence, compute_js_divergence
import os


def tsne_side_by_side_with_metrics(file1, file2, output_pdf=None):
    """
    Loads and preprocesses two datasets independently, computes t-SNE on each,
    computes KL and JS divergence, and plots the results side by side with metrics.
    """

    # Load and preprocess both datasets
    print(f"Loading and preprocessing {file1}")
    adata1 = load_and_preprocess_single(file1, use_basename=True)

    print(f"Loading and preprocessing {file2}")
    adata2 = load_and_preprocess_single(file2, use_basename=True)

    # Extract PCA embeddings
    X1 = adata1.obsm["X_pca"]
    X2 = adata2.obsm["X_pca"]

    # === Compute Divergences ===
    embedding_key = 'X_pca'
    sample_size = 1000
    bandwidth = 0.5  # You can loop over multiple if needed

    print("\nComputing Divergences...")
    kl = compute_kl_divergence(
        adata1, adata2,
        mode='embedding',
        embedding_key=embedding_key,
        sample_size=sample_size,
        bandwidth=bandwidth
    )
    js = compute_js_divergence(
        adata1, adata2,
        mode='embedding',
        embedding_key=embedding_key,
        sample_size=sample_size,
        bandwidth=bandwidth
    )
    print(f"KL Divergence: {kl:.4f}")
    print(f"JS Divergence: {js:.4f}")

    # === Run t-SNE on both datasets ===
    def run_tsne(X):
        affinities = openTSNE.affinity.Multiscale(X, perplexities=[50, 500], metric="cosine", n_jobs=8)
        init = openTSNE.initialization.pca(X, random_state=0)
        tsne = openTSNE.TSNEEmbedding(init, affinities, negative_gradient_method="fft", n_jobs=8)
        tsne.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
        tsne.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
        return tsne

    tsne1 = run_tsne(X1)
    tsne2 = run_tsne(X2)

    # === Plot side by side with metrics ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(tsne1[:, 0], tsne1[:, 1], s=3, c="tab:blue", alpha=0.6)
    axes[0].set_title(f"{os.path.basename(file1)}\n(t-SNE)", fontsize=10)
    axes[0].axis("off")

    axes[1].scatter(tsne2[:, 0], tsne2[:, 1], s=3, c="tab:green", alpha=0.6)
    axes[1].set_title(f"{os.path.basename(file2)}\n(t-SNE)", fontsize=10)
    axes[1].axis("off")

    # Add divergence metrics text below the plots
    fig.text(0.5, 0.02, f"KL Divergence = {kl:.4f}    |    JS Divergence = {js:.4f}", 
             ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if output_pdf is None:
        file1_name = os.path.splitext(os.path.basename(file1))[0]
        file2_name = os.path.splitext(os.path.basename(file2))[0]
        output_pdf = f"tsne_with_metrics_{file1_name}_{file2_name}.pdf"

    print(f"Saving plot to {output_pdf}")
    plt.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    # tsne_side_by_side_with_metrics(
    #     "extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad",
    #     "extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad"
    # )
    tsne_side_by_side_with_metrics("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
