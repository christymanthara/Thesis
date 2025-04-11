# shared_embedding_plot.py

import anndata
import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
import utils
from data_utils import clustering_metrics

def prepare_data(adata, new):
    shared_genes = adata.var_names[adata.var_names.isin(new.var_names)]
    adata = adata[:, shared_genes].copy()
    new = new[:, shared_genes].copy()

    adata = adata[:, adata.var_names.argsort()]
    new = new[:, new.var_names.argsort()]
    assert all(adata.var_names == new.var_names)

    gene_mask_250 = utils.select_genes(adata.X, n=250, threshold=0)
    gene_mask_1000 = utils.select_genes(adata.X, n=1000, threshold=0)

    return {
        250: (adata[:, gene_mask_250].copy(), new[:, gene_mask_250].copy()),
        1000: (adata[:, gene_mask_1000].copy(), new[:, gene_mask_1000].copy()),
        "all": (adata, new)
    }

def run_pipeline(
    adata_path, new_path, method_name, embed_func, save_prefix, init_data=None, init_key="X_init"
):
    adata = anndata.read_h5ad(adata_path)
    new = anndata.read_h5ad(new_path)

    gene_subsets = prepare_data(adata, new)
    metrics = {}
    embeddings = {}

    for gene_key, (adata_subset, new_subset) in gene_subsets.items():
        print(f"Running {method_name} for {gene_key} genes")
        embedding_ref, embedding_new = embed_func(adata_subset, new_subset, gene_key, init_data)
        new_subset.obsm[f"X_{method_name.lower()}"] = np.array(embedding_new)
        new_subset.uns["ari"] = clustering_metrics.ari = clustering_metrics.calculate_clustering_metrics(embedding_new, new_subset.obs["labels"])["ARI"]
        new_subset.uns["ami"] = clustering_metrics.ami = clustering_metrics.calculate_clustering_metrics(embedding_new, new_subset.obs["labels"])["AMI"]
        new_subset.write_h5ad(f"{save_prefix}_{gene_key}_genes.h5ad")
        metrics[str(gene_key)] = clustering_metrics.calculate_clustering_metrics(embedding_new, new_subset.obs["labels"])
        embeddings[str(gene_key)] = new_subset

    return adata, new, metrics, embeddings
