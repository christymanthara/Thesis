import anndata
import numpy as np
import matplotlib.pyplot as plt
import utils
from matplotlib.backends.backend_pdf import PdfPages

def multiplot_tsne_pca(file_name: str):
    # Load the data
    adata = anndata.read_h5ad(file_name)
    
    # Generate file names for saving
    pdf_file = file_name.replace(".h5ad", "_tsne_plot.pdf")
    
    # Define legend arguments
    legend_kwargs = dict(loc="center", bbox_to_anchor=(0.5, -0.05), ncol=len(np.unique(adata.obs["labels"])))
    
    # Get colors
    colors = utils.get_colors_for(adata)
    
    with PdfPages(pdf_file) as pdf:
        # PCA Plot
        fig, ax = plt.subplots(ncols=3, figsize=(24, 8))
        utils.plot(adata.obsm["pca"], adata.obs["labels"], s=2, colors=colors, draw_legend=False, ax=ax[0], alpha=1, title="250 genes", label_order=list(colors.keys()))
        # utils.plot(adata.obsm["pca"], adata.obs["labels"], s=2, colors=colors, draw_legend=True, ax=ax[1], alpha=1, title="3000 genes", label_order=list(colors.keys()), legend_kwargs=legend_kwargs)
        # utils.plot(adata.obsm["pca"], adata.obs["labels"], s=2, colors=colors, draw_legend=False, ax=ax[2], alpha=1, title="17499 genes", label_order=list(colors.keys()))
        
        plt.text(0, 1.02, "a", transform=ax[0].transAxes, fontsize=15, fontweight="bold")
        # plt.text(0, 1.02, "b", transform=ax[1].transAxes, fontsize=15, fontweight="bold")
        # plt.text(0, 1.02, "c", transform=ax[2].transAxes, fontsize=15, fontweight="bold")
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # t-SNE Plot
        fig, ax = plt.subplots(ncols=3, figsize=(24, 8))
        utils.plot(adata.obsm["tsne"], adata.obs["labels"], s=2, colors=colors, draw_legend=False, ax=ax[0], alpha=1, title="250 genes", label_order=list(colors.keys()))
        # utils.plot(adata.obsm["tsne"], adata.obs["labels"], s=2, colors=colors, draw_legend=True, ax=ax[1], alpha=1, title="3000 genes", label_order=list(colors.keys()), legend_kwargs=legend_kwargs)
        # utils.plot(adata.obsm["tsne"], adata.obs["labels"], s=2, colors=colors, draw_legend=False, ax=ax[2], alpha=1, title="17499 genes", label_order=list(colors.keys()))
        
        plt.text(0, 1.02, "a", transform=ax[0].transAxes, fontsize=15, fontweight="bold")
        # plt.text(0, 1.02, "b", transform=ax[1].transAxes, fontsize=15, fontweight="bold")
        # plt.text(0, 1.02, "c", transform=ax[2].transAxes, fontsize=15, fontweight="bold")
        
        pdf.savefig(fig)
        plt.close(fig)
    
if __name__ == "__main__":
    multiplot_tsne_pca("baron_2016h_embedding_tsne_250_genes.h5ad")
