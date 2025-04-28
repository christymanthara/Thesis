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
import os
from scripts.data_utils.processing import load_and_label_scGPT


def embed_and_visualize(
    reference_adata,
    query_adata,
    model_dir,
    file1_name="file1",
    file2_name="file2",
    cell_type_key="cell_type",
    # cell_type_key="labels",
    gene_col="gene_name",
    batch_size=64,
    return_combined=True
):
    """
    Embed reference and query AnnData objects using scGPT and visualize the combined data.
    
    Saves:
    - ref_embed_adata as {file1_name}_scGPT.h5ad
    - query_embed_adata as {file2_name}_scGPT.h5ad
    - adata_concat as adata_concat_scGPT_{file1_name}_{file2_name}.h5ad
    """

    # Check for FAISS
    try:
        import faiss
        faiss_imported = True
    except ImportError:
        faiss_imported = False
        print("FAISS not installed! We highly recommend installing it for fast similarity search.")
        print("To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")
    
    # Filenames for embedded data
    ref_embed_file = f"{file1_name}_scGPT.h5ad"
    query_embed_file = f"{file2_name}_scGPT.h5ad"

    # Load or compute reference embedding
    if os.path.exists(ref_embed_file):
        print(f"Loading existing reference embedding from {ref_embed_file}")
        ref_embed_adata = anndata.read_h5ad(ref_embed_file)
    else:
        print(f"Embedding reference data and saving to {ref_embed_file}")
        ref_embed_adata = scg.tasks.embed_data(
            reference_adata,
            model_dir,
            gene_col=gene_col,
            batch_size=batch_size,
        )
        ref_embed_adata.write_h5ad(ref_embed_file)
    
    # Load or compute query embedding
    if os.path.exists(query_embed_file):
        print(f"Loading existing query embedding from {query_embed_file}")
        query_embed_adata = anndata.read_h5ad(query_embed_file)
    else:
        print(f"Embedding query data and saving to {query_embed_file}")
        query_embed_adata = scg.tasks.embed_data(
            query_adata,
            model_dir,
            gene_col=gene_col,
            batch_size=batch_size,
        )
        query_embed_adata.write_h5ad(query_embed_file)

    # # Embed reference data
    # ref_embed_adata = scg.tasks.embed_data(
    #     reference_adata,
    #     model_dir,
    #     gene_col=gene_col,
    #     batch_size=batch_size,
    # )
    
    # # Embed query data
    # query_embed_adata = scg.tasks.embed_data(
    #     query_adata,
    #     model_dir,
    #     gene_col=gene_col,
    #     batch_size=batch_size,
    # )

    # Save embedded datasets
    # ref_embed_adata.write_h5ad(f"{file1_name}_scGPT.h5ad")
    # query_embed_adata.write_h5ad(f"{file2_name}_scGPT.h5ad")
    
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
    
    # Save concatenated dataset
    concat_filename = f"adata_concat_scGPT_{file1_name}_{file2_name}.h5ad"
    adata_concat.write_h5ad(concat_filename)

    print(f"Saved {file1_name}_scGPT.h5ad, {file2_name}_scGPT.h5ad, and {concat_filename}")

    
    return adata_concat

def concat_and_embed(
    adata1,
    adata2,
    model_dir,
    file1_name="file1",
    file2_name="file2",
    cell_type_key="cell_type",
    gene_col="gene_name",
    batch_size=64,
):
    """
    Concatenate two AnnData objects and embed them using scGPT.
    
    Saves:
    - adata_concat as adata_concat_scGPT_{file1_name}_{file2_name}.h5ad
    """
    
    # # Concatenate the two datasets
    # adata_concat = adata1.concatenate(adata2, batch_key="dataset")
    
    # # Mark reference vs. query dataset
    # adata_concat.obs["is_ref"] = ["Query"] * len(adata2) + ["Reference"] * len(adata1)
    
    # # Ensure the cell type column exists
    # if cell_type_key not in adata_concat.obs.columns:
    #     raise ValueError(f"'{cell_type_key}' not found in obs. Available columns: {adata_concat.obs.columns.tolist()}")
    
    # # Mask the query dataset cell types
    # adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].astype("category")
    # adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].cat.add_categories(["To be predicted"])
    # adata_concat.obs[cell_type_key][: len(adata2)] = "To be predicted"
    
    # # Calculate UMAP from scGPT embeddings
    # sc.pp.neighbors(adata_concat, use_rep="X_scGPT")
    # sc.tl.umap(adata_concat)
    
    # # Visualize
    # sc.pl.umap(
    #     adata_concat, color=["is_ref", cell_type_key], wspace=0.4, frameon=False, ncols=1
    # )
    
    # # Save concatenated dataset
    # concat_filename = f"adata_concat_scGPT_{file1_name}_{file2_name}.h5ad"
    # adata_concat.write_h5ad(concat_filename)

     # Load and label
    combined_adata, source1, source2 = load_and_label_scGPT(file1, file2)

    # Save the concatenated raw file
    concat_filename = f"{source1}{source2}.h5ad"
    combined_adata.write(concat_filename)
    print(f"Concatenated dataset saved to {concat_filename}")

    # Prepare genes
    combined_adata.var['gene_name'] = combined_adata.var.index

    # Embed the whole dataset
    embedded_adata = scg.tasks.embed_data(
        combined_adata,
        model_dir,
        gene_col="gene_name",
        batch_size=64
    )

    # UMAP and plotting
    sc.pp.neighbors(embedded_adata, use_rep="X_scGPT")
    sc.tl.umap(embedded_adata)

    # Color by source
    sc.pl.umap(embedded_adata, color=["source"], wspace=0.4, frameon=False, ncols=1)

    # Save
    # embedded_adata.write_h5ad("concat_embedded.h5ad")
    embedded_filename = f"concat_embedded_{source1}_{source2}.h5ad"
    embedded_adata.write(embedded_filename)
    print(f"Embedded dataset saved to {embedded_filename}")


if __name__ == "__main__":
    # file1 = "temp/Datasets/lung/sample_proc_lung_test.h5ad"
    file1 = "/home/thechristyjo/Documents/Thesis/Datasets/baron_2016h.h5ad"
    # file1 = "/home/thechristyjo/Documents/Thesis/Datasets/chen_2017.h5ad"
    # file1 = "Datasets/macosko_2015.h5ad"
    # file1 = "/home/thechristyjo/Documents/Thesis/Datasets/sample_proc_lung_train.h5ad"


    # file2 = "temp/Datasets/lung/sample_proc_lung_train.h5ad"
    file2 = "/home/thechristyjo/Documents/Thesis/datasets/xin_2016.h5ad"
    # file2 = "/home/thechristyjo/Documents/Thesis/Datasets/hrvatin_2018.h5ad"
    # file2 = "Datasets/shekhar_2016.h5ad"
    # file2 = "/home/thechristyjo/Documents/Thesis/Datasets/sample_proc_lung_test.h5ad"

    adata = anndata.read_h5ad(file1)
    adata.var['gene_name'] = adata.var.index
    
    adata2 = anndata.read_h5ad(file2)
    adata2.var['gene_name'] = adata2.var.index
    
    model_dir = "/home/thechristyjo/Documents/Thesis/scGPT/model"
    
    # Extract short filenames without paths and extensions
    file1_name = os.path.splitext(os.path.basename(file1))[0]
    file2_name = os.path.splitext(os.path.basename(file2))[0]
    
    # embed_and_visualize(adata, adata2, model_dir, file1_name=file1_name, file2_name=file2_name)
    concat_and_embed(
        adata, adata2, model_dir, file1_name=file1_name, file2_name=file2_name, cell_type_key="labels"
    )
