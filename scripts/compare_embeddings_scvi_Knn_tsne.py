import anndata
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.neighbors import KNeighborsClassifier
from data_utils.scvi_embedding import load_and_preprocess_for_scvi
from sklearn.metrics import accuracy_score
from umap import UMAP
# import scvi
import utils
import os
# from openTSNE.callbacks import ErrorLogger
from compute_tsne_embeddings import compute_tsne_embedding


def plot_scvi_umap(file1,file2 , output_pdf=None, skip_preprocessing=False, qc_filtered_genes = False):
    # Step 1: Set up scVI model for reference data
    if skip_preprocessing:
        # Load data directly without preprocessing
        print(f"Loading {file1} directly (skipping preprocessing)")
        adata = anndata.read_h5ad(file1)

        print(f"Loading {file2} directly (skipping preprocessing)")
        new = anndata.read_h5ad(file2)
        
        # Check if X_scVI embeddings exist
        if 'X_scVI' not in adata.obsm or 'X_scVI' not in new.obsm:
            raise ValueError("X_scVI embeddings not found in the provided files. Set skip_preprocessing=False to generate them.")
    else:
        # Original preprocessing code
        print(f"Loading and preprocessing {file1} and {file2}")
        adata, new = load_and_preprocess_for_scvi(file1, file2, qc_filtered=qc_filtered_genes)

    
    # Step 2: Compute t-SNE embeddings using openTSNE
    print("Computing t-SNE embeddings from X_scVI...")
    print("Computing t-SNE for %d genes" % adata.shape[1])
    adata_ = compute_tsne_embedding(adata,embedding_key  ="X_scVI", output_key="scvi_tsne")
    print("Computing t-SNE for %d genes" % new.shape[1])
    new_ = compute_tsne_embedding(new, embedding_key="X_scVI", output_key="scvi_tsne")
    # Step 3: Run KNN classification using scVI embeddings
    # Using scVI latent representations
    
    
    knn_scvi = KNeighborsClassifier()
    knn_scvi.fit(adata_.obsm["X_scVI"], adata_.obs["labels"].values.astype(str))
    scvi_accuracy = accuracy_score(knn_scvi.predict(new_.obsm["X_scVI"]), new_.obs["labels"].values.astype(str))
    print(f"KNN accuracy using scVI embeddings: {scvi_accuracy:.4f}")

    # Using UMAP embeddings derived from scVI
    knn_umap = KNeighborsClassifier()
    knn_umap.fit(adata.obsm["scvi_tsne"], adata_.obs["labels"].values.astype(str))
    umap_accuracy = accuracy_score(knn_umap.predict(new_.obsm["scvi_tsne"]), new_.obs["labels"].values.astype(str))
    print(f"KNN accuracy using tsne embeddings: {umap_accuracy:.4f}")

    colors = utils.get_colors_for(adata_)
    cell_order = list(colors.keys())
    num_cell_types = len(np.unique(adata_.obs["labels"]))

    # Create plot
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    
    # Add overall title to the figure with actual obs columns
    file1_name = os.path.splitext(os.path.basename(file1))[0]
    file2_name = os.path.splitext(os.path.basename(file2))[0]
    
    # Get the obsm keys (embeddings like 'X_scVI', 'umap', etc.)
    obsm_keys = list(adata_.obsm.keys())
    obsm_str = ','.join(obsm_keys)
    
    fig.suptitle(f"{file1_name}_{file2_name}_obsm[{obsm_str}]", fontsize=14, y=0.95)


    # Plot reference embedding
    utils.plot(adata_.obsm["scvi_tsne"], adata_.obs["labels"], ax=ax[0], title="Reference embedding (UMAP from scVI)", 
            colors=colors, s=3, label_order=cell_order,
            legend_kwargs=dict(loc="upper center", bbox_to_anchor=(0.5, 0.05), 
                                bbox_transform=fig.transFigure, labelspacing=1, 
                                ncol=num_cell_types // 2 + 1),
                                knn_scvi_accuracy=scvi_accuracy,  # Pass the accuracies
                                knn_umap_accuracy=umap_accuracy)

    # Plot transformed samples
    colors_bw = {1: "#666666"}
    utils.plot(adata_.obsm["scvi_tsne"], np.ones_like(adata_.obs["labels"]), ax=ax[1], 
            colors=colors_bw, alpha=0.05, s=3, draw_legend=False)
    utils.plot(new_.obsm["scvi_tsne"], new_.obs["labels"], ax=ax[1], colors=colors, 
            draw_legend=False, s=6, label_order=cell_order, alpha=0.2)
    ax[1].set_title("Transformed samples (UMAP)")

    # Set equal axis for all plots
    for ax_ in ax.ravel(): 
        ax_.axis("equal")

    # Determine coordinate range from UMAP data
    tsne_min = min(adata_.obsm["scvi_tsne"].min(), new_.obsm["scvi_tsne"].min())
    tsne_max = max(adata_.obsm["scvi_tsne"].max(), new_.obsm["scvi_tsne"].max())
    coord_range = tsne_min - 1, tsne_max + 1

    for ax_ in ax.ravel():
        ax_.set_xlim(*coord_range), ax_.set_ylim(*coord_range)

    # Add subplot labels
    for ax_, letter in zip(ax, string.ascii_lowercase): 
        plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=15, fontweight="bold")
        
    # Add KNN accuracy text to the figure
    fig.text(0.5, 0.10, f"KNN(scVI): {scvi_accuracy:.4f}    |    KNN(tsne)): {umap_accuracy:.4f}", 
            ha='center', fontsize=12)
    

    # Generate default output_pdf name if not provided
    if output_pdf is None:
        # Extract file names without extensions
        file1_name = os.path.splitext(os.path.basename(file1))[0]
        file2_name = os.path.splitext(os.path.basename(file2))[0]
        # Create the output file name
        if qc_filtered_genes:
            output_pdf = f"tsne_plot_scVI_Knn_concatenated_qc_filtered{file1_name}_{file2_name}.pdf"
        else:
            output_pdf = f"tsne_plot_scVI_Knn_concatenated{file1_name}_{file2_name}.pdf"
    
    plt.savefig(output_pdf, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Saved plot to {output_pdf}")

if __name__ == "__main__":
    
    # plot_scvi_umap("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
    
    # plot_scvi_umap("Datasets/baron_2016h_scvi.h5ad", "Datasets/xin_2016_scvi.h5ad", skip_preprocessing=True)
    # plot_scvi_umap("Datasets/baron_2016h_scvi.h5ad", "Datasets/xin_2016_scvi.h5ad", qc_filtered_genes = True)
    plot_scvi_umap("Datasets/baron_2016h_scvi.h5ad", "Datasets/xin_2016_scvi.h5ad", skip_preprocessing=True, qc_filtered_genes = False)
    
    # plot_scvi_umap("Datasets/hrvatin_2018_scvi.h5ad", "Datasets/chen_2017_scvi.h5ad", skip_preprocessing=True)
