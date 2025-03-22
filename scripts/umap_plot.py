import scanpy as sc
import umap
import matplotlib.pyplot as plt
from data_utils.processing import load_and_preprocess

def process_and_plot_umap(file1, file2, output_file="umap_plot.pdf"):
    """
    Loads two AnnData files using the common preprocessing function, computes UMAP,
    and saves the plot as a PDF.
    
    Parameters:
    - file1 (str): Path to the first .h5ad file.
    - file2 (str): Path to the second .h5ad file.
    - output_file (str): Path where the UMAP plot should be saved.
    """
    # Preprocess and obtain the concatenated AnnData object
    full = load_and_preprocess(file1, file2, use_basename=True)
    
    # Create UMAP using the PCA representation
    umap_instance = umap.UMAP()
    full.obsm["X_umap"] = umap_instance.fit_transform(full.obsm["X_pca"])
    
    # Optionally compute the neighborhood graph and UMAP via Scanpy as well
    sc.pp.neighbors(full, n_neighbors=15, n_pcs=50, metric="cosine")
    sc.tl.umap(full)
    
    # Plot the UMAP embedding with cells colored by source
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(full, color="source", ax=ax, show=False)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    
    print(f"UMAP plot saved as {output_file}")

if __name__ == "__main__":
    process_and_plot_umap("../datasets/baron_2016h.h5ad", "../datasets/xin_2016.h5ad")
