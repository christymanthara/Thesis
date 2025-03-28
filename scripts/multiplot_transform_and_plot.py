import anndata
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from openTSNE import affinity, TSNEEmbedding
import utils
import os

from preprocess_plot_on_top import run_gene_preprocess_pipeline

def multiplot_transform(adata_path: str, new_path: str):
    adata = anndata.read_h5ad(adata_path)
    new = anndata.read_h5ad(new_path)
    adata_3000 = anndata.read_h5ad("../scripts/baron_2016h_embedding_tsne_3000_genes.h5ad")
   
    shared_genes = adata.var_names[adata.var_names.isin(new.var_names)]
    adata = adata[:, adata.var_names.isin(shared_genes)]
    new = new[:, new.var_names.isin(shared_genes)]
    
    adata = adata[:, adata.var_names.argsort()].copy()
    new = new[:, new.var_names.argsort()].copy()
    assert all(adata.var_names == new.var_names)

    gene_mask_250 = utils.select_genes(adata.X, n=250, threshold=0)
    gene_mask_1000 = utils.select_genes(adata.X, n=1000, threshold=0)

    adata_250 = adata[:, gene_mask_250].copy()
    adata_1000 = adata[:, gene_mask_1000].copy()
    adata_full = adata

    new_250 = new[:, gene_mask_250].copy()
    new_1000 = new[:, gene_mask_1000].copy()
    new_full = new
    
    affinity_obs = {}
    for adata_, new_ in [(adata_250, new_250), (adata_1000, new_1000), (adata_full, new_full)]:
        print("Running transform for %d genes" % adata_.shape[1])
        affinities = affinity.PerplexityBasedNN(
            adata_.X.toarray() if sp.issparse(adata_.X) else adata_.X,
            perplexity=30,
            metric="cosine",
            n_jobs=8,
            random_state=3,
        )
        affinity_obs[adata_.shape[1]] = affinities
        embedding = TSNEEmbedding(
            adata_3000.obsm["X_tsne"],
            affinities,
            negative_gradient_method="fft",
            n_jobs=8,
        )
        new_embedding = embedding.prepare_partial(new_.X.toarray(), k=10)
        new_.obsm["tsne_init"] = new_embedding.copy()
        
        new_embedding.optimize(250, learning_rate=0.1, momentum=0.8, inplace=True)
        new_.obsm["X_tsne"] = new_embedding

    # Convert and remove unsupported objects before saving
    for new_, genes in [(new_250, 250), (new_1000, 1000), (new_full, "all")]:
        new_.obsm["X_tsne"] = np.array(new_.obsm["X_tsne"])  
        new_.obsm["tsne_init"] = np.array(new_.obsm["tsne_init"])  
        new_.write_h5ad(f"new_embedding_tsne_{genes}_genes.h5ad")  

    new_250 = anndata.read_h5ad("new_embedding_tsne_250_genes.h5ad")
    new_1000 = anndata.read_h5ad("new_embedding_tsne_1000_genes.h5ad")
    new_full = anndata.read_h5ad("new_embedding_tsne_all_genes.h5ad")

    colors = utils.get_colors_for(adata)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=False, ax=ax[0, 0], alpha=0.1, title="Initialization", label_order=list(colors.keys()))
    utils.plot(new_250.obsm["tsne_init"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[0, 0], alpha=1, label_order=list(colors.keys()))

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=False, ax=ax[0, 1], alpha=0.1, title="250 genes", label_order=list(colors.keys()))
    utils.plot(new_250.obsm["X_tsne"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[0, 1], alpha=1, label_order=list(colors.keys()))

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=False, ax=ax[1, 0], alpha=0.1, title="1000 genes", label_order=list(colors.keys()))
    utils.plot(new_1000.obsm["X_tsne"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[1, 0], alpha=1, label_order=list(colors.keys()))

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=True, ax=ax[1, 1], alpha=0.1, title="17078 genes", label_order=list(colors.keys()),
            legend_kwargs=dict(bbox_transform=fig.transFigure, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=len(np.unique(adata.obs["labels"]))))

    utils.plot(new_full.obsm["X_tsne"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[1, 1], alpha=1, label_order=list(colors.keys()))

    # Extract filenames without extensions
    filename1 = os.path.splitext(os.path.basename(adata_path))[0]
    filename2 = os.path.splitext(os.path.basename(new_path))[0]

    # Format the output filename
    output_pdf = f"{filename1}-{filename2}-plotontop.pdf"
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"Plot saved as {output_pdf}")

if __name__ == "__main__":
    multiplot_transform("../Datasets/baron_2016h.h5ad", "../Datasets/xin_2016.h5ad")
