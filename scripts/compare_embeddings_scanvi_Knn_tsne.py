import anndata
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.neighbors import KNeighborsClassifier
from data_utils.scanvi_embedding import load_and_preprocess_for_scanvi
from sklearn.metrics import accuracy_score
# import scvi
import utils
import os

from compute_tsne_embeddings import compute_tsne_embedding

def plot_scanvi_tsne(file1,file2 , output_pdf=None, skip_preprocessing=False):
    # Step 1: Set up scanvi model for reference data
    if skip_preprocessing:
        # Load data directly without preprocessing
        print(f"Loading {file1} directly (skipping preprocessing)")
        adata = anndata.read_h5ad(file1)
        
        print(f"Loading {file2} directly (skipping preprocessing)")
        new = anndata.read_h5ad(file2)
        
        # Check if X_scANVI embeddings exist
        if 'X_scANVI' not in adata.obsm or 'X_scANVI' not in new.obsm:
            raise ValueError("X_scANVI embeddings not found in the provided files. Set skip_preprocessing=False to generate them.")
    else:
        # Original preprocessing code
        print(f"Loading and preprocessing {file1} and {file2}")
        adata, new = load_and_preprocess_for_scanvi(file1, file2)

    
    # Step 2: Compute t-SNE embeddings using openTSNE
    print("Computing t-SNE embeddings from X_scVI...")
    print("Computing t-SNE for %d genes" % adata.shape[1])
    adata_ = compute_tsne_embedding(adata,embedding_key  ="X_scANVI", output_key="scanvi_tsne")
    print("Computing t-SNE for %d genes" % new.shape[1])
    new_ = compute_tsne_embedding(new, embedding_key="X_scANVI", output_key="scanvi_tsne")

    # Step 3: Run KNN classification using scanvi embeddings
    # Using scVI latent representations
    
    # you cannot use the metric cosine for tsne
    
    knn_scanvi = KNeighborsClassifier()
    knn_scanvi.fit(adata_.obsm["X_scANVI"], adata_.obs["labels"].values.astype(str))
    scanvi_accuracy = accuracy_score(knn_scanvi.predict(new_.obsm["X_scANVI"]), new_.obs["labels"].values.astype(str))
    print(f"KNN accuracy using scanvi embeddings: {scanvi_accuracy:.4f}")

    # Using TSNE embeddings derived from scanvi
    knn_tsne = KNeighborsClassifier()
    knn_tsne.fit(adata_.obsm["scanvi_tsne"], adata_.obs["labels"].values.astype(str))
    tsne_accuracy = accuracy_score(knn_tsne.predict(new_.obsm["scanvi_tsne"]), new_.obs["labels"].values.astype(str))
    print(f"KNN accuracy using TSNE embeddings: {tsne_accuracy:.4f}")

    colors = utils.get_colors_for(adata_)
    cell_order = list(colors.keys())
    num_cell_types = len(np.unique(adata_.obs["labels"]))

    # Create plot
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    
    # Add overall title to the figure with actual obs columns
    file1_name = os.path.splitext(os.path.basename(file1))[0]
    file2_name = os.path.splitext(os.path.basename(file2))[0]
    
    # Get the obsm keys (embeddings like 'X_scANVI', 'tsne', etc.)
    obsm_keys = list(adata_.obsm.keys())
    obsm_str = ','.join(obsm_keys)
    
    fig.suptitle(f"{file1_name}_{file2_name}_obsm[{obsm_str}]", fontsize=14, y=0.95)


    # Plot reference embedding
    utils.plot(adata_.obsm["scanvi_tsne"], adata_.obs["labels"], ax=ax[0], title="Reference embedding (tsne from scanvi)", 
            colors=colors, s=3, label_order=cell_order,
            legend_kwargs=dict(loc="upper center", bbox_to_anchor=(0.5, 0.05), 
                                bbox_transform=fig.transFigure, labelspacing=1, 
                                ncol=num_cell_types // 2 + 1),
                                knn_scanvi_accuracy=scanvi_accuracy,  # Pass the accuracies
                                knn_tsne_accuracy=tsne_accuracy)

    # Plot transformed samples
    colors_bw = {1: "#666666"}
    utils.plot(adata_.obsm["scanvi_tsne"], np.ones_like(adata_.obs["labels"]), ax=ax[1], 
            colors=colors_bw, alpha=0.05, s=3, draw_legend=False)
    utils.plot(new_.obsm["scanvi_tsne"], new_.obs["labels"], ax=ax[1], colors=colors, 
            draw_legend=False, s=6, label_order=cell_order, alpha=0.2)
    ax[1].set_title("Transformed samples (TSNE)")

    # Set equal axis for all plots
    for ax_ in ax.ravel(): 
        ax_.axis("equal")

    # Determine coordinate range from TSNE data
    tsne_min = min(adata_.obsm["scanvi_tsne"].min(), new_.obsm["scanvi_tsne"].min())
    tsne_max = max(adata_.obsm["scanvi_tsne"].max(), new_.obsm["scanvi_tsne"].max())
    coord_range = tsne_min - 1, tsne_max + 1

    for ax_ in ax.ravel():
        ax_.set_xlim(*coord_range), ax_.set_ylim(*coord_range)

    # Add subplot labels
    for ax_, letter in zip(ax, string.ascii_lowercase): 
        plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=15, fontweight="bold")
        
    # Add KNN accuracy text to the figure
    fig.text(0.5, 0.10, f"KNN(scanvi): {scanvi_accuracy:.4f}    |    KNN(TSNE): {tsne_accuracy:.4f}", 
            ha='center', fontsize=12)
    

    # Generate default output_pdf name if not provided
    if output_pdf is None:
        # Extract file names without extensions
        file1_name = os.path.splitext(os.path.basename(file1))[0]
        file2_name = os.path.splitext(os.path.basename(file2))[0]
        # Create the output file name
        output_pdf = f"tsne_plot_scanVI_Knn_concatenated{file1_name}_{file2_name}.pdf"
    
    plt.savefig(output_pdf, dpi=600, bbox_inches="tight", transparent=True)

if __name__ == "__main__":
    
    plot_scanvi_tsne("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
    
    # plot_scanvi_tsne("Datasets/baron_2016h_scanvi.h5ad", "Datasets/xin_2016_scanvi.h5ad", skip_preprocessing=True)
    # plot_scanvi_tsne("Datasets/hrvatin_2018_scanvi.h5ad", "Datasets/chen_2017_scanvi.h5ad", skip_preprocessing=True)
