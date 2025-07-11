import anndata
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from openTSNE import affinity, TSNEEmbedding
import utils
import os
from data_utils import clustering_metrics_AMI_ARI

from pavlin_preprocess_plot_on_top import run_gene_preprocess_pipeline

def multiplot_transform_tsne(adata_path: str, new_path: str):
    adata = anndata.read_h5ad(adata_path)
    new = anndata.read_h5ad(new_path)
    # adata_3000 = anndata.read_h5ad("../scripts/baron_2016h_embedding_tsne_3000_genes.h5ad")
    print("Original labels in new (before filtering):", new.obs["labels"].value_counts())
    print("Original labels in adata (before filtering):", adata.obs["labels"].value_counts())
    

    adata_3000 = anndata.read_h5ad("baron_2016h_embedding_tsne_3000_genes.h5ad")
   
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


    metrics = {}
    gene_counts = []  # To store the number of genes for each dataset
    ari_values = []   # To store ARI values
    ami_values = []   # To store AMI values
    
    # After creating the embeddings, calculate metrics
    for new_, genes in [(new_250, 250), (new_1000, 1000), (new_full, new_full.shape[1])]:
        # Calculate ARI and AMI using embeddings
        embedding_metrics = clustering_metrics_AMI_ARI.calculate_clustering_metrics(
            new_.obsm["X_tsne"], 
            new.obs["labels"]
        )
        
        gene_key = str(genes)
        metrics[gene_key] = embedding_metrics
        
        # Store values for plotting
        gene_counts.append(genes)
        ari_values.append(embedding_metrics['ARI'])
        ami_values.append(embedding_metrics['AMI'])
        
        print(f"Embedding with {gene_key} genes:")
        print(f"  ARI: {embedding_metrics['ARI']:.4f}")
        print(f"  AMI: {embedding_metrics['AMI']:.4f}")
        
        # Save metrics as annotations
        new_.uns["ari"] = embedding_metrics["ARI"]
        new_.uns["ami"] = embedding_metrics["AMI"]
        
        # Convert and remove unsupported objects before saving
        new_.obsm["X_tsne"] = np.array(new_.obsm["X_tsne"])
        new_.obsm["tsne_init"] = np.array(new_.obsm["tsne_init"])
        new_.write_h5ad(f"new_embedding_tsne_{genes}_genes.h5ad")

    # Load saved AnnData objects
    new_250 = anndata.read_h5ad("new_embedding_tsne_250_genes.h5ad")
    new_1000 = anndata.read_h5ad("new_embedding_tsne_1000_genes.h5ad")
    new_full = anndata.read_h5ad(f"new_embedding_tsne_{new_full.shape[1]}_genes.h5ad")

    # Create ARI/AMI plot
    create_ari_ami_plot(gene_counts, ari_values, ami_values, adata_path, new_path)

    colors = utils.get_colors_for(adata)

    # Create figure and axes BEFORE trying to set titles
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    
    # Now you can set titles for subplots
    ax[0, 1].set_title(f"250 genes (ARI: {metrics['250']['ARI']:.4f}, AMI: {metrics['250']['AMI']:.4f})")
    ax[1, 0].set_title(f"1000 genes (ARI: {metrics['1000']['ARI']:.4f}, AMI: {metrics['1000']['AMI']:.4f})")
    ax[1, 1].set_title(f"All genes ({new_full.shape[1]}) (ARI: {metrics[str(new_full.shape[1])]['ARI']:.4f}, AMI: {metrics[str(new_full.shape[1])]['AMI']:.4f})")

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=False, ax=ax[0, 0], alpha=0.1, title="Initialization", label_order=list(colors.keys()))
    utils.plot(new_250.obsm["tsne_init"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[0, 0], alpha=1, label_order=list(colors.keys()))

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=False, ax=ax[0, 1], alpha=0.1, title="250 genes", label_order=list(colors.keys()))
    utils.plot(new_250.obsm["X_tsne"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[0, 1], alpha=1, label_order=list(colors.keys()))

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=False, ax=ax[1, 0], alpha=0.1, title="1000 genes", label_order=list(colors.keys()))
    utils.plot(new_1000.obsm["X_tsne"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[1, 0], alpha=1, label_order=list(colors.keys()))

    utils.plot(adata_3000.obsm["X_tsne"], adata.obs["labels"], s=3, colors=colors, draw_legend=True, ax=ax[1, 1], alpha=0.1, title=f"{new_full.shape[1]} genes", label_order=list(colors.keys()),
            legend_kwargs=dict(bbox_transform=fig.transFigure, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=len(np.unique(adata.obs["labels"]))))

    utils.plot(new_full.obsm["X_tsne"], new.obs["labels"], s=12, colors=colors, draw_legend=False, ax=ax[1, 1], alpha=1, label_order=list(colors.keys()))

    # Extract filenames without extensions
    filename1 = os.path.splitext(os.path.basename(adata_path))[0]
    filename2 = os.path.splitext(os.path.basename(new_path))[0]

    # Format the output filename
    output_pdf = f"{filename1}-{filename2}-plotontop_tsne.pdf"
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"Plot saved as {output_pdf}")

def create_ari_ami_plot(gene_counts, ari_values, ami_values, adata_path, new_path):
    """
    Create a plot showing ARI and AMI values against the number of genes.
    
    Parameters:
    -----------
    gene_counts : list
        List of gene counts
    ari_values : list
        List of ARI values corresponding to gene counts
    ami_values : list
        List of AMI values corresponding to gene counts
    adata_path : str
        Path to first dataset file
    new_path : str
        Path to second dataset file
    """
    # Extract filenames without extensions
    filename1 = os.path.splitext(os.path.basename(adata_path))[0]
    filename2 = os.path.splitext(os.path.basename(new_path))[0]
    
    # Create figure for ARI and AMI plot
    plt.figure(figsize=(10, 6))
    
    # Plot ARI and AMI values
    plt.plot(gene_counts, ari_values, 'o-', label='ARI', color='blue', linewidth=2, markersize=8)
    plt.plot(gene_counts, ami_values, 's-', label='AMI', color='red', linewidth=2, markersize=8)
    
    # Set labels and title
    plt.xlabel('Number of Genes', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(f'ARI and AMI vs Number of Genes\n{filename1} to {filename2} transformation (t-SNE)', fontsize=14)
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, (g, ari, ami) in enumerate(zip(gene_counts, ari_values, ami_values)):
        plt.annotate(f'{ari:.4f}', (g, ari), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
        plt.annotate(f'{ami:.4f}', (g, ami), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9)
    
    # Add legend
    plt.legend(loc='best', fontsize=10)
    
    # Format the output filename for ARI/AMI plot
    output_pdf = f"{filename1}_{filename2}_ARI_AMI_tsne.pdf"
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf")
    print(f"ARI/AMI plot saved as {output_pdf}")

if __name__ == "__main__":
    # multiplot_transform("../Datasets/baron_2016h.h5ad", "../Datasets/xin_2016.h5ad")
    multiplot_transform_tsne("datasets/baron_2016h.h5ad", "datasets/xin_2016.h5ad") #use when running from the root directory