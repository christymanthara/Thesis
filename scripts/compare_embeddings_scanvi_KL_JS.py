import openTSNE
import scanpy as sc
import matplotlib.pyplot as plt
from data_utils.scvi_embedding import load_and_preprocess_for_scvi
from data_utils.clustering_metrics_KL_JS import compute_kl_divergence, compute_js_divergence
import os
import scvi
import utils
from data_utils.scanvi_embedding import load_and_preprocess_for_scanvi  # Import the new function


def tsne_side_by_side_with_metrics(file1, file2, output_pdf=None, use_scanvi=False,label_column=None,):
    """
    Loads and preprocesses two datasets, computes t-SNE on each,
    computes KL and JS divergence, and plots the results side by side with metrics.
    
    Parameters:
    -----------
    file1, file2 : str
        Paths to the AnnData files to process
    output_pdf : str, optional
        Path to save the output PDF file
    use_scanvi : bool, default=False
        If True, use scANVI for integration, otherwise use scVI
    """

    # Load and preprocess both datasets
    print(f"Loading and preprocessing {file1} and {file2}")
    
    if use_scanvi:
        # Use scANVI for integration
        print("Info: Using scANVI")
        adata1, adata2 = load_and_preprocess_for_scanvi(file1, file2, label_column=label_column,)
        embedding_key = 'X_scANVI'  # Changed to match the new embedding key
    else:
        # Use the original scVI method
        print("Info: Using scvi")
        adata1, adata2 = load_and_preprocess_for_scvi(file1, file2)
        embedding_key = 'X_scVI'

    # Extract embeddings
    X1 = adata1.obsm[embedding_key]
    X2 = adata2.obsm[embedding_key]

    # === Compute Divergences ===
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
        affinities = openTSNE.affinity.Multiscale(X, perplexities=[50, 500], metric="cosine", n_jobs=4)
        init = openTSNE.initialization.pca(X, random_state=0)
        tsne = openTSNE.TSNEEmbedding(init, affinities, negative_gradient_method="fft", n_jobs=4)
        tsne.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
        tsne.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
        return tsne
    
    # concatenate the 2 data and plot with tsne
    adata_full = adata1.concatenate(adata2)
    
    # Use the appropriate embedding key
    X = adata_full.obsm[embedding_key]

    # Compute affinities using multiscale perplexities
    affinities = openTSNE.affinity.Multiscale(
        X,
        perplexities=[50, 500],
        metric="cosine",
        n_jobs=8,
        random_state=0,
    )
    # Initialize embedding using PCA
    init = openTSNE.initialization.pca(X, random_state=0)
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
        # Create the output file name with indication of which method was used
        method = "scANVI" if use_scanvi else "scVI"
        output_pdf = f"tsne_plot_{method}_KL_JS_concatenated_{file1_name}_{file2_name}.pdf"
    
    # Plot using the provided utils.plot function and save the plot
    utils.plot(embedding, adata_full.obs["source"], save_path=output_pdf, kl_divergence=kl,
    js_divergence=js, save_as_svg=True)


if __name__ == "__main__":
    # Example usage with scVI (original functionality)
    # tsne_side_by_side_with_metrics("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad", use_scanvi=False)
    
    # Example usage with scANVI (new functionality)
    # tsne_side_by_side_with_metrics("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad", use_scanvi=True)
    tsne_side_by_side_with_metrics(
        "extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad",
        "extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad",
        use_scanvi=True,
        label_column="assigned_cluster",
    )