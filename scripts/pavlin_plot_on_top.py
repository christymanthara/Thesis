import anndata
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import utils
from openTSNE import TSNE, TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
from os import path

def preprocess_anndata(file_path, n_genes=None):
    # Load dataset
    adata = anndata.read_h5ad(file_path)
    sc.pp.filter_genes(adata, min_counts=1)
    
    # Normalize data
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1_000_000)
    sc.pp.log1p(adata_norm)
    adata_norm.X = adata_norm.X.toarray()
    adata_norm.X -= adata_norm.X.mean(axis=0)
    adata_norm.X /= adata_norm.X.std(axis=0)
    
    # Select genes if specified
    if n_genes is not None:
        gene_mask = utils.select_genes(adata.X, n=n_genes, threshold=0)
        adata_norm = adata_norm[:, gene_mask].copy()
    
    return adata_norm

def compute_pca(adata_list):
    for adata_ in adata_list:
        print(f"Computing PCA for {adata_.shape[1]} genes")
        U, S, V = np.linalg.svd(adata_.X, full_matrices=False)
        U[:, np.sum(V, axis=1) < 0] *= -1
        adata_.obsm["pca"] = np.dot(U, np.diag(S))[:, np.argsort(S)[::-1]][:, :50]

def compute_tsne(adata_list):
    for adata_ in adata_list:
        print(f"Computing t-SNE for {adata_.shape[1]} genes")
        # affinities = affinity.PerplexityBasedNN(
        affinities = affinity.Multiscale(
            adata_.obsm["pca"],
            perplexities=[50, 500], #when using affinity.Multiscale
            metric="cosine",
            n_jobs=8,
            random_state=3,
        )
        init = initialization.pca(adata_.obsm["pca"], random_state=42)
        embedding = TSNEEmbedding(
            init, affinities, negative_gradient_method="fft", n_jobs=8
        )
        embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
        embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
        adata_.obsm["tsne"] = np.array(embedding)

def run_gene_preprocess_pipeline(file_path):
    file_name = path.splitext(path.basename(file_path))[0]
    # Process dataset
    adata_250 = preprocess_anndata(file_path, n_genes=250)
    adata_3000 = preprocess_anndata(file_path, n_genes=3000)
    adata_full = preprocess_anndata(file_path)
    
    # Compute PCA and t-SNE
    # compute_pca([adata_250, adata_3000, adata_full])
    # compute_tsne([adata_250, adata_3000, adata_full])
    compute_pca([adata_250])
    compute_tsne([adata_250])

    # Save processed data
    adata_250.write_h5ad(f"{file_name}_embedding_tsne_250_genes.h5ad")
    # adata_250.write_h5ad("test_embedding_tsne_250_genes.h5ad")
    # adata_3000.write_h5ad("baron_embedding_tsne_3000_genes.h5ad")
    # adata_full.write_h5ad("baron_embedding_tsne_all_genes.h5ad")

if __name__ == "__main__":
    run_gene_preprocess_pipeline("../datasets/baron_2016h.h5ad")

