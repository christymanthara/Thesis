import anndata
import scanpy as sc
import os
import matplotlib.pyplot as plt

def process_and_plot_umap(file1, file2, output_file="umap_plot.pdf"):
    """
    Loads two AnnData files, merges them, computes UMAP, and saves the plot as a PDF.
    
    Parameters:
    - file1 (str): Path to the first .h5ad file.
    - file2 (str): Path to the second .h5ad file.
    - output_file (str): Path where the UMAP plot should be saved.
    """

    # Load the datasets
    adata = anndata.read_h5ad(file1)
    new = anndata.read_h5ad(file2)

    # Add source labels
    adata.obs["source"] = os.path.basename(file1)  # Just filename, no full path
    new.obs["source"] = os.path.basename(file2)

    # Filter new data to match labels from adata
    cell_mask = new.obs["labels"].isin(adata.obs["labels"])
    new = new[cell_mask].copy()

    # Concatenate datasets
    full = adata.concatenate(new)

    # Filter genes with at least 1 count
    sc.pp.filter_genes(full, min_counts=1)

    # Compute neighborhood graph using PCA representation
    sc.pp.neighbors(full, n_neighbors=15, n_pcs=50, metric="cosine")

    # Compute UMAP
    sc.tl.umap(full)

    # Explicitly create the figure before plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(full, color="source", ax=ax, show=False)  # `show=False` prevents immediate display
    plt.savefig(output_file, bbox_inches="tight")  # Ensure proper saving
    plt.close()

    print(f"UMAP plot saved as {output_file}")

# Example usage:
# process_and_plot_umap("file1.h5ad", "file2.h5ad", "umap_plot.pdf")

if __name__ == "__main__":
    process_and_plot_umap("../datasets/baron_2016h.h5ad", "../datasets/xin_2016.h5ad")
