import openTSNE
import scanpy as sc
from data_utils.processing import load_and_preprocess_re
from data_utils.clustering_metrics_KL_JS import compute_kl_divergence, compute_js_divergence
import utils  # assuming utils.plot is available for plotting
import os

def tsne_pavlin_re(file1, file2, output_pdf=None):
    """
    Loads two AnnData files using the common preprocessing function, computes t-SNE
    using openTSNE, and saves the plot as a PDF.
    
    Parameters:
    - file1 (str): Path to the first .h5ad file.
    - file2 (str): Path to the second .h5ad file.
    - output_pdf (str): Path where the t-SNE plot should be saved. If None, 
                         the filename will be generated based on file names.
    """
    # Preprocess and obtain the concatenated AnnData object.
    adata1,adata2 = load_and_preprocess_re(file1, file2, use_basename=True)

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

    full = adata1.concatenate(adata2)
    X = full.obsm["X_scVI"]

    
    # Compute affinities using multiscale perplexities
    affinities = openTSNE.affinity.Multiscale(
        full.obsm["X_pca"],
        perplexities=[50, 500],
        metric="cosine",
        n_jobs=8,
        random_state=0,
    )
    # Initialize embedding using PCA
    init = openTSNE.initialization.pca(full.obsm["X_pca"], random_state=0)
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
        output_pdf = f"tsne_plot_withKL_JS_{file1_name}_{file2_name}.pdf"
    
    # Plot using the provided utils.plot function and save the plot
    utils.plot(embedding, full.obs["source"], save_path=output_pdf)

if __name__ == "__main__":
    # tsne_pavlin("../datasets/baron_2016h.h5ad", "../datasets/xin_2016.h5ad")
    # tsne_pavlin("../extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad", "../extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad")
    tsne_pavlin_re("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
   