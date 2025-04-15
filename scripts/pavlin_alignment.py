import openTSNE
import scanpy as sc
from data_utils.processing import load_and_preprocess
import utils  # assuming utils.plot is available for plotting
import os

def tsne_pavlin(file1, file2, output_pdf=None):
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
    full = load_and_preprocess(file1, file2, use_basename=True)
    
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
        output_pdf = f"tsne_plot_{file1_name}_{file2_name}.pdf"
    
    # Plot using the provided utils.plot function and save the plot
    utils.plot(embedding, full.obs["source"], save_path=output_pdf)

if __name__ == "__main__":
    # tsne_pavlin("../datasets/baron_2016h.h5ad", "../datasets/xin_2016.h5ad")
    # tsne_pavlin("../extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad", "../extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad")
    tsne_pavlin("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
   