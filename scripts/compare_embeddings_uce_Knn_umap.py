import anndata
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from umap import UMAP
import utils
import os

def plot_uce_umap(file1,file2 , output_pdf=None, skip_preprocessing=False):
    # Step 1: Set up uce model for reference data
    if skip_preprocessing:
        # Load data directly without preprocessing
        print(f"Loading {file1} directly (skipping preprocessing)")
        adata = anndata.read_h5ad(file1)
        
        print(f"Loading {file2} directly (skipping preprocessing)")
        new = anndata.read_h5ad(file2)
        
        # Check if X_uce embeddings exist
        if 'X_uce' not in adata.obsm or 'X_uce' not in new.obsm:
            raise ValueError("X_uce embeddings not found in the provided files. Set skip_preprocessing=False to generate them.")
    else:
        # Original preprocessing code
        print(f"Loading and preprocessing {file1} and {file2}. run the uce pipeline manually. Exiting now")
        # To be done
        # run the uce pipeline
        pass

    
    # Step 2: Compute UMAP embeddings for both datasets
    # For reference data
    umap_model = UMAP(n_components=2)
    adata.obsm["umap"] = umap_model.fit_transform(adata.obsm["X_uce"])

    # For new data, use the same UMAP model to ensure consistent embedding
    new.obsm["umap"] = umap_model.transform(new.obsm["X_uce"])

    # Step 3: Run KNN classification using uce embeddings
    # Using uce latent representations
    knn_uce = KNeighborsClassifier(metric='euclidean')
    knn_uce.fit(adata.obsm["X_uce"], adata.obs["labels"].values.astype(str))
    uce_accuracy = accuracy_score(knn_uce.predict(new.obsm["X_uce"]), new.obs["labels"].values.astype(str))
    print(f"KNN accuracy using uce embeddings: {uce_accuracy:.4f}")
    # run the metric with cosine for the image generation

    # Using UMAP embeddings derived from uce
    knn_umap = KNeighborsClassifier()
    knn_umap.fit(adata.obsm["umap"], adata.obs["labels"].values.astype(str))
    umap_accuracy = accuracy_score(knn_umap.predict(new.obsm["umap"]), new.obs["labels"].values.astype(str))
    print(f"KNN accuracy using UMAP embeddings: {umap_accuracy:.4f}")

    colors = utils.get_colors_for(adata)
    cell_order = list(colors.keys())
    num_cell_types = len(np.unique(adata.obs["labels"]))

    # Create plot
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    
    # Add overall title to the figure with actual obs columns
    file1_name = os.path.splitext(os.path.basename(file1))[0]
    file2_name = os.path.splitext(os.path.basename(file2))[0]
    
    # Get the obsm keys (embeddings like 'X_uce', 'umap', etc.)
    obsm_keys = list(adata.obsm.keys())
    obsm_str = ','.join(obsm_keys)
    
    fig.suptitle(f"{file1_name}_{file2_name}_obsm[{obsm_str}]", fontsize=14, y=0.95)


    # Plot reference embedding
    utils.plot(adata.obsm["umap"], adata.obs["labels"], ax=ax[0], title="Reference embedding (UMAP from uce)", 
            colors=colors, s=3, label_order=cell_order,
            legend_kwargs=dict(loc="upper center", bbox_to_anchor=(0.5, 0.05), 
                                bbox_transform=fig.transFigure, labelspacing=1, 
                                ncol=num_cell_types // 2 + 1),
                                knn_uce_accuracy=uce_accuracy,  # Pass the accuracies
                                knn_umap_accuracy=umap_accuracy)

    # Plot transformed samples
    colors_bw = {1: "#666666"}
    utils.plot(adata.obsm["umap"], np.ones_like(adata.obs["labels"]), ax=ax[1], 
            colors=colors_bw, alpha=0.05, s=3, draw_legend=False)
    utils.plot(new.obsm["umap"], new.obs["labels"], ax=ax[1], colors=colors, 
            draw_legend=False, s=6, label_order=cell_order, alpha=0.2)
    ax[1].set_title("Transformed samples (UMAP)")

    # Set equal axis for all plots
    for ax_ in ax.ravel(): 
        ax_.axis("equal")

    # Determine coordinate range from UMAP data
    umap_min = min(adata.obsm["umap"].min(), new.obsm["umap"].min())
    umap_max = max(adata.obsm["umap"].max(), new.obsm["umap"].max())
    coord_range = umap_min - 1, umap_max + 1

    for ax_ in ax.ravel():
        ax_.set_xlim(*coord_range), ax_.set_ylim(*coord_range)

    # Add subplot labels
    for ax_, letter in zip(ax, string.ascii_lowercase): 
        plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=15, fontweight="bold")
        
    # Add KNN accuracy text to the figure
    fig.text(0.5, 0.10, f"KNN(uce): {uce_accuracy:.4f}    |    KNN(UMAP): {umap_accuracy:.4f}", 
            ha='center', fontsize=12)
    
    # Generate default output_pdf name if not provided
    if output_pdf is None:
        # Extract file names without extensions
        file1_name = os.path.splitext(os.path.basename(file1))[0]
        file2_name = os.path.splitext(os.path.basename(file2))[0]
        # Create the output file name
        output_pdf = f"umap_plot_uce_Knn_euclidean_concatenated{file1_name}_{file2_name}.pdf"
    
    plt.savefig(output_pdf, dpi=600, bbox_inches="tight", transparent=True)

if __name__ == "__main__":
    
    # plot_uce_umap("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
    
    plot_uce_umap("Datasets/baron_2016h_uce_adata.h5ad", "Datasets/xin_2016_uce_adata.h5ad", skip_preprocessing=True)
    
