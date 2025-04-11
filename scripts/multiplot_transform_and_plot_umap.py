import anndata
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
import umap
from data_utils import clustering_metrics
import utils


def multiplot_transform_umap(adata_path: str, new_path: str):
    adata = anndata.read_h5ad(adata_path)
    new = anndata.read_h5ad(new_path)

    print("Original labels in new (before filtering):", new.obs["labels"].value_counts())
    print("Original labels in adata (before filtering):", adata.obs["labels"].value_counts())
    
    
    shared_genes = adata.var_names[adata.var_names.isin(new.var_names)]
    # print(f"Shared genes: {len(shared_genes)}")
    # print(f"Shared genes: {shared_genes}")
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

    # Run UMAP on original data to get adata.obsm["X_umap"]
    print("Running UMAP on original dataset (adata_full)")

    X_orig = adata_full.X.toarray() if sp.issparse(adata_full.X) else adata_full.X

    reducer_orig = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_orig = reducer_orig.fit_transform(X_orig)

    

    adata_full.obsm["X_umap"] = embedding_orig
    
    # Initialize lists to store gene counts, ARI and AMI values
    metrics = {}
    gene_counts = []  # To store the number of genes for each dataset
    ari_values = []   # To store ARI values
    ami_values = []   # To store AMI values
    
    for new_, genes in [(new_full, new_full.shape[1])]:
        print(f"Running UMAP for {genes} genes")
        
        X = new_.X.toarray() if sp.issparse(new_.X) else new_.X

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer_orig.transform(X)

        new_.obsm["X_umap"] = embedding
        print(f"the values are {adata.obs['labels']}")

        embedding_metrics = clustering_metrics.calculate_clustering_metrics(
            embedding,
            new_.obs["labels"]
        )

        gene_key = str(genes)
        metrics[gene_key] = embedding_metrics
        
        # Store values for plotting
        gene_counts.append(genes)
        ari_values.append(embedding_metrics['ARI'])
        ami_values.append(embedding_metrics['AMI'])

        print(f"UMAP with {gene_key} genes:")
        print(f"  ARI: {embedding_metrics['ARI']:.4f}")
        print(f"  AMI: {embedding_metrics['AMI']:.4f}")

        new_.uns["ari"] = embedding_metrics["ARI"]
        new_.uns["ami"] = embedding_metrics["AMI"]

        new_.obsm["X_umap"] = np.array(embedding)
        new_.write_h5ad(f"new_embedding_umap_{genes}_genes.h5ad")
    
    # Create ARI/AMI plot
    create_ari_ami_plot(gene_counts, ari_values, ami_values, adata_path, new_path)

    # Reload with annotations
    new_250 = anndata.read_h5ad("new_embedding_umap_250_genes.h5ad")
    new_1000 = anndata.read_h5ad("new_embedding_umap_1000_genes.h5ad")
    new_full = anndata.read_h5ad(f"new_embedding_umap_{new_full.shape[1]}_genes.h5ad")

    colors = utils.get_colors_for(adata)
    print(f"Colors: {colors}")
    classes = adata.obs["labels"].cat.categories  # or sorted(adata.obs["labels"].unique())

    print("Unique labels in new_250:", new_250.obs["labels"].unique())
    print("Unique labels in new_1000:", new_1000.obs["labels"].unique())
    print("Unique labels in new_full:", new_full.obs["labels"].unique())

    print("Original labels in new (before filtering):", new.obs["labels"].value_counts())


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

    # ax[0, 1].set_title(f"250 genes (ARI: {metrics['250']['ARI']:.4f}, AMI: {metrics['250']['AMI']:.4f})")
    # ax[1, 0].set_title(f"1000 genes (ARI: {metrics['1000']['ARI']:.4f}, AMI: {metrics['1000']['AMI']:.4f})")
    ax[1, 1].set_title(f"All genes ({new_full.shape[1]}) (ARI: {metrics[str(new_full.shape[1])]['ARI']:.4f}, AMI: {metrics[str(new_full.shape[1])]['AMI']:.4f})")

    utils.plot(adata_full.obsm["X_umap"], adata.obs["labels"], s=3, colors=colors,classes=classes, draw_legend=False, ax=ax[0, 0], alpha=0.1, title="Initialization", label_order=list(colors.keys()))
    utils.plot(new_250.obsm["X_umap"], new.obs["labels"], s=12, colors=colors,classes=classes, draw_legend=False, ax=ax[0, 0], alpha=1, title="Initialization", label_order=list(colors.keys()))
    
    utils.plot(adata_full.obsm["X_umap"], adata.obs["labels"], s=3, colors=colors,classes=classes, draw_legend=False, ax=ax[0, 1], alpha=0.1, title="250 genes", label_order=list(colors.keys()))
    utils.plot(new_250.obsm["X_umap"], new_250.obs["labels"], s=12, colors=colors,classes=classes, draw_legend=False, ax=ax[0, 1], alpha=1, title="250 genes", label_order=list(colors.keys()))
    
    utils.plot(adata_full.obsm["X_umap"], adata.obs["labels"], s=3, colors=colors,classes=classes, draw_legend=False, ax=ax[1, 0], alpha=0.1, title="1000 genes", label_order=list(colors.keys()))
    utils.plot(new_1000.obsm["X_umap"], new_1000.obs["labels"], s=12, colors=colors,classes=classes, draw_legend=False, ax=ax[1, 0], alpha=1, title="1000 genes", label_order=list(colors.keys()))
    
    utils.plot(adata_full.obsm["X_umap"], adata.obs["labels"], s=3, colors=colors,classes=classes, draw_legend=True, ax=ax[1, 1], alpha=0.1, title=f"{new_full.shape[1]} genes", label_order=list(colors.keys()),
            legend_kwargs=dict(bbox_transform=fig.transFigure, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=len(np.unique(adata.obs["labels"]))))
    utils.plot(new_full.obsm["X_umap"], new_full.obs["labels"], s=12, colors=colors,classes=classes, draw_legend=True, ax=ax[1, 1], alpha=1, title=f"{new_full.shape[1]} genes", label_order=list(colors.keys()),
               legend_kwargs=dict(bbox_transform=fig.transFigure, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=len(np.unique(adata.obs["labels"]))))

    filename1 = os.path.splitext(os.path.basename(adata_path))[0]
    filename2 = os.path.splitext(os.path.basename(new_path))[0]

    output_pdf = f"{filename1}-{filename2}-umap-plot.pdf"
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"UMAP plot saved as {output_pdf}")

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
    plt.title(f'ARI and AMI vs Number of Genes\n{filename1} to {filename2} transformation (UMAP)', fontsize=14)
    
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
    output_pdf = f"{filename1}_{filename2}_ARI_AMI_umap.pdf"
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf")
    print(f"ARI/AMI plot saved as {output_pdf}")

if __name__ == "__main__":
    # multiplot_transform("../Datasets/baron_2016h.h5ad", "../Datasets/xin_2016.h5ad")
    multiplot_transform_umap("datasets/baron_2016h.h5ad", "datasets/xin_2016.h5ad") #use when running from the root directory