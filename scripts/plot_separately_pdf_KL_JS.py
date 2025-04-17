import openTSNE
import scanpy as sc
import matplotlib.pyplot as plt
from data_utils.processing import load_and_preprocess_single,load_and_preprocess_separately
from data_utils.clustering_metrics_KL_JS import compute_kl_divergence, compute_js_divergence
import os
import utils


def tsne_side_by_side_with_metrics(file1, file2, output_pdf=None):
    """
    Loads and preprocesses two datasets independently, computes t-SNE on each,
    computes KL and JS divergence, and plots the results side by side with metrics.
    """

    # Load and preprocess both datasets
    print(f"Loading and preprocessing {file1}")
    # adata1 = load_and_preprocess_single(file1, use_basename=True)

    print(f"Loading and preprocessing {file2}")
    # adata2 = load_and_preprocess_single(file2, use_basename=True)
    adata1,adata2 = load_and_preprocess_separately(file1,file2)

    # Extract PCA embeddings
    X1 = adata1.obsm["X_pca"]
    X2 = adata2.obsm["X_pca"]

    # === Compute Divergences ===
    embedding_key = 'X_pca'
    sample_size = 2000
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
        affinities = openTSNE.affinity.Multiscale(X, perplexities=[50, 500], metric="cosine", n_jobs=4)
        init = openTSNE.initialization.pca(X, random_state=0)
        tsne = openTSNE.TSNEEmbedding(init, affinities, negative_gradient_method="fft", n_jobs=4)
        tsne.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
        tsne.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
        return tsne

    # tsne1 = run_tsne(X1)
    # tsne2 = run_tsne(X2)

    adata_full = adata1.concatenate(adata2)
    X = adata_full.obsm["X_pca"]

   #-------------------------------------------pavlins plotting
     # Compute affinities using multiscale perplexities
    affinities = openTSNE.affinity.Multiscale(
        adata_full.obsm["X_pca"],
        perplexities=[50, 500],
        metric="cosine",
        n_jobs=8,
        random_state=0,
    )
    # Initialize embedding using PCA
    init = openTSNE.initialization.pca(adata_full.obsm["X_pca"], random_state=0)
    embedding = openTSNE.TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=8,
    )
    
    # Optimize the embedding in two phases
    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
    
    # Generate default output_pdf name if not provided
    if output_pdf is None:
        # Extract file names without extensions
        file1_name = os.path.splitext(os.path.basename(file1))[0]
        file2_name = os.path.splitext(os.path.basename(file2))[0]
        # Create the output file name
        output_pdf = f"Pavlins_tsne_with_metrics_genefiltered_{file1_name}_{file2_name}.pdf"
    
    # Plot using the provided utils.plot function and save the plot
    utils.plot(embedding, adata_full.obs["source"], save_path=output_pdf,kl_divergence=kl,
    js_divergence=js,save_as_svg=True)




if __name__ == "__main__":
    tsne_side_by_side_with_metrics(
        "extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad",
        "extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad"
    )
    # tsne_side_by_side_with_metrics("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
