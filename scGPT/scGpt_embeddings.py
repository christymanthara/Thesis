import numpy as np
from scipy.stats import mode
import scanpy as sc
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import sys
import scgpt as scg
import anndata


def embed_and_visualize(
    reference_adata,
    query_adata,
    model_dir,
    # cell_type_key="celltype",
    cell_type_key="labels",
    gene_col="gene_name",
    batch_size=64,
    return_combined=True
):
    """
    Embed reference and query AnnData objects using scGPT and visualize the combined data.
    
    Parameters:
    -----------
    reference_adata : AnnData
        Reference dataset to be embedded
    query_adata : AnnData
        Query dataset to be embedded
    model_dir : str or Path
        Path to the scGPT model directory
    cell_type_key : str, default="celltype"
        Key in adata.obs containing cell type annotations
    gene_col : str, default="gene_name"
        Column name for gene identifiers
    batch_size : int, default=64
        Batch size for embedding
    return_combined : bool, default=True
        Whether to return the combined AnnData object
        
    Returns:
    --------
    If return_combined=True, returns the combined AnnData object with embeddings
    Otherwise, returns a tuple of (reference_embed_adata, query_embed_adata)
    """
    # import scanpy as sc
    # import scgpt as scg
    
    # Check for FAISS
    try:
        import faiss
        faiss_imported = True
    except ImportError:
        faiss_imported = False
        print("FAISS not installed! We highly recommend installing it for fast similarity search.")
        print("To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")
    
    # Embed reference data
    ref_embed_adata = scg.tasks.embed_data(
        reference_adata,
        model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
    )
    
    # Embed query data
    query_embed_adata = scg.tasks.embed_data(
        query_adata,
        model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
    )
    
    if not return_combined:
        return ref_embed_adata, query_embed_adata
    
    # Concatenate the two datasets
    adata_concat = query_embed_adata.concatenate(ref_embed_adata, batch_key="dataset")
    
    # Mark reference vs. query dataset
    adata_concat.obs["is_ref"] = ["Query"] * len(query_embed_adata) + ["Reference"] * len(ref_embed_adata)
    adata_concat.obs["is_ref"] = adata_concat.obs["is_ref"].astype("category")
    
    # Ensure the cell type column exists
    if cell_type_key not in adata_concat.obs.columns:
        raise ValueError(f"'{cell_type_key}' not found in obs. Available columns: {adata_concat.obs.columns.tolist()}")


    # Mask the query dataset cell types
    adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].astype("category")
    adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].cat.add_categories(["To be predicted"])
    adata_concat.obs[cell_type_key][: len(query_embed_adata)] = "To be predicted"
    
    # Calculate UMAP from scGPT embeddings
    sc.pp.neighbors(adata_concat, use_rep="X_scGPT")
    sc.tl.umap(adata_concat)
    
    # Visualize
    sc.pl.umap(
        adata_concat, color=["is_ref", cell_type_key], wspace=0.4, frameon=False, ncols=1
    )
    
    return adata_concat



if __name__ == "__main__":
    # file1 = "temp/Datasets/lung/sample_proc_lung_test.h5ad"
    file1 = "/home/thechristyjo/Documents/Thesis/Datasets/baron_2016h.h5ad"
    # file2 = "temp/Datasets/lung/sample_proc_lung_train.h5ad"
    file2 = "/home/thechristyjo/Documents/Thesis/datasets/xin_2016.h5ad"
    adata = anndata.io.read_h5ad(file1)
    adata.var['gene_name'] = adata.var.index
    adata2 = anndata.io.read_h5ad(file2)
    adata2.var['gene_name'] = adata2.var.index
    # model_dir = "scripts/scGPT/model"
    model_dir = "/home/thechristyjo/Documents/Thesis/scGPT/scGPT/model"
    embed_and_visualize(adata,adata2, model_dir)