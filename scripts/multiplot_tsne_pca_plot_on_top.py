import anndata
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
from matplotlib.backends.backend_pdf import PdfPages


def get_common_prefix(file_names):
    """Returns the common prefix of the provided file names."""
    return os.path.commonprefix(file_names).rstrip("_")

def multiplot_tsne_pca(file_names: list):
    """
    Generates PCA and t-SNE plots for each file in file_names.
    The plots are arranged side-by-side in the PDF.
    If there is only one file, one image is plotted per page; if more, images are arranged in columns.
    
    Parameters:
        file_names (list): List of file names (strings) to process.
    """
    ncols = len(file_names)

    # Extract common prefix for naming the output file
    common_prefix = get_common_prefix([os.path.basename(f) for f in file_names])
    pdf_file = f"{common_prefix}_tsne_plot.pdf"

    with PdfPages(pdf_file) as pdf:
        # PCA Plot Page
        fig, ax = plt.subplots(ncols=ncols, figsize=(8 * ncols, 8))
        if ncols == 1:
            ax = [ax]
        for i, file_name in enumerate(file_names):
            adata = anndata.read_h5ad(file_name)
            legend_kwargs = dict(loc="center", bbox_to_anchor=(0.5, -0.05), 
                                 ncol=len(np.unique(adata.obs["labels"])))
            colors = utils.get_colors_for(adata)
            draw_legend = (i == 1)
            utils.plot(adata.obsm["pca"],
                       adata.obs["labels"],
                       s=2,
                       colors=colors,
                       draw_legend=draw_legend,
                       ax=ax[i],
                       alpha=1,
                       title=file_name,
                       label_order=list(colors.keys()),
                       legend_kwargs=legend_kwargs if draw_legend else None)
            ax[i].text(0, 1.02, chr(97 + i), transform=ax[i].transAxes,
                       fontsize=15, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # t-SNE Plot Page
        fig, ax = plt.subplots(ncols=ncols, figsize=(8 * ncols, 8))
        if ncols == 1:
            ax = [ax]
        for i, file_name in enumerate(file_names):
            adata = anndata.read_h5ad(file_name)
            legend_kwargs = dict(loc="center", bbox_to_anchor=(0.5, -0.05), 
                                 ncol=len(np.unique(adata.obs["labels"])))
            colors = utils.get_colors_for(adata)
            draw_legend = (i == 1)
            utils.plot(adata.obsm["X_tsne"],
                       adata.obs["labels"],
                       s=2,
                       colors=colors,
                       draw_legend=draw_legend,
                       ax=ax[i],
                       alpha=1,
                       title=file_name,
                       label_order=list(colors.keys()),
                       legend_kwargs=legend_kwargs if draw_legend else None)
            ax[i].text(0, 1.02, chr(97 + i), transform=ax[i].transAxes,
                       fontsize=15, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Plots saved as {pdf_file}")

if __name__ == "__main__":
    # Example usage:
    # For a single file:
    # multiplot_tsne_pca(["baron_2016h_embedding_tsne_250_genes.h5ad"])
    
    # For multiple files, uncomment and modify as needed:
    multiplot_tsne_pca([
        "baron_2016h_embedding_tsne_250_genes.h5ad",
        "baron_2016h_embedding_tsne_3000_genes.h5ad",
        "baron_2016h_embedding_tsne_all_genes.h5ad"
    ])
