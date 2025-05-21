import anndata
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.neighbors import KNeighborsClassifier
from data_utils.scvi_embedding import load_and_preprocess_for_scvi
from sklearn.metrics import accuracy_score
from umap import UMAP
import scvi
import utils
import os

def plot_scvi_umap(file1,file2 , output_pdf=None):
    # Load the data
    # adata = anndata.read_h5ad("baron_embedding_tsne_3000_genes.h5ad")
    # new = anndata.read_h5ad("baron_transform_tsne_1000_genes.h5ad")

    # Load and preprocess both datasets
    print(f"Loading and preprocessing {file1}")
    # adata1 = load_and_preprocess_single(file1, use_basename=True)

    print(f"Loading and preprocessing {file2}")
    # adata2 = load_and_preprocess_single(file2, use_basename=True)
    adata,new = load_and_preprocess_for_scvi(file1,file2)


    # Step 1: Set up scVI model for reference data
    
    # Step 2: Compute UMAP embeddings for both datasets
    # For reference data
    umap_model = UMAP(n_components=2)
    adata.obsm["umap"] = umap_model.fit_transform(adata.obsm["X_scVI"])

    # For new data, use the same UMAP model to ensure consistent embedding
    new.obsm["umap"] = umap_model.transform(new.obsm["X_scVI"])

    # Step 3: Run KNN classification using scVI embeddings
    # Using scVI latent representations
    knn_scvi = KNeighborsClassifier()
    knn_scvi.fit(adata.obsm["X_scVI"], adata.obs["labels"].values.astype(str))
    scvi_accuracy = accuracy_score(knn_scvi.predict(new.obsm["X_scVI"]), new.obs["labels"].values.astype(str))
    print(f"KNN accuracy using scVI embeddings: {scvi_accuracy:.4f}")

    # Using UMAP embeddings derived from scVI
    knn_umap = KNeighborsClassifier()
    knn_umap.fit(adata.obsm["umap"], adata.obs["labels"].values.astype(str))
    umap_accuracy = accuracy_score(knn_umap.predict(new.obsm["umap"]), new.obs["labels"].values.astype(str))
    print(f"KNN accuracy using UMAP embeddings: {umap_accuracy:.4f}")

    # Step 5: Visualization with the same plotting approach, but using UMAP coordinates

    # Get color mapping function
    # def get_colors_for(adata):
    #     """Get pretty colors for each class."""
    #     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    #             "#7f7f7f",  # This is the grey one
    #             "#e377c2", "#bcbd22", "#17becf",
    #             "#0000A6", "#63FFAC", "#004D43", "#8FB0FF"]

    #     colors = dict(zip(adata.obs["labels"].value_counts().sort_values(ascending=False).index, colors))
    #     colors["Other"] = "#7f7f7f"
    #     assert all(l in colors for l in adata.obs["labels"].unique())
    #     return colors

    colors = utils.get_colors_for(adata)
    cell_order = list(colors.keys())
    num_cell_types = len(np.unique(adata.obs["labels"]))

    # Create plot
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

    # Plot reference embedding
    utils.plot(adata.obsm["umap"], adata.obs["labels"], ax=ax[0], title="Reference embedding (UMAP from scVI)", 
            colors=colors, s=3, label_order=cell_order,
            legend_kwargs=dict(loc="upper center", bbox_to_anchor=(0.5, 0.05), 
                                bbox_transform=fig.transFigure, labelspacing=1, 
                                ncol=num_cell_types // 2 + 1))

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

    # Save figure
    plt.savefig("transform_pancreas_scvi_umap.pdf", dpi=600, bbox_inches="tight", transparent=True)

    # # Optional: If you want to compare with original t-SNE results
    # compare_fig, compare_ax = plt.subplots(figsize=(10, 5))
    # compare_ax.bar(['t-SNE', 'scVI', 'UMAP from scVI'], 
    #             [accuracy_score(knn.predict(new.obsm["tsne"]), new.obs["labels"].values.astype(str)), 
    #             scvi_accuracy, umap_accuracy])
    # compare_ax.set_title('KNN Classification Accuracy Comparison')
    # compare_ax.set_ylim(0, 1)
    
    # Generate default output_pdf name if not provided
    if output_pdf is None:
        # Extract file names without extensions
        file1_name = os.path.splitext(os.path.basename(file1))[0]
        file2_name = os.path.splitext(os.path.basename(file2))[0]
        # Create the output file name
        output_pdf = f"tsne_plot_scVI_KL_JS_concatenated{file1_name}_{file2_name}.pdf"
    
    plt.savefig(output_pdf, dpi=600, bbox_inches="tight", transparent=True)

if __name__ == "__main__":
    # tsne_side_by_side_with_metrics(
    #     "extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad",
    #     "extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad"
    # )
    plot_scvi_umap("Datasets/baron_2016h.h5ad", "Datasets/xin_2016.h5ad")
