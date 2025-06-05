import anndata
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.neighbors import KNeighborsClassifier
from data_utils.scanvi_embedding import preprocess_single_adata_for_scanvi
from data_utils.scvi_embedding import process_single_adata_for_scvi
import utils
import os

def compute_embeddings_scvi_scanvi_scgpt(
    adata: str,
):
    """
    Compute embeddings for scVI, scANVI, and scGPT if missing.
    
    Args:
        adata: Path to the AnnData file or AnnData object
    """
    # Load adata if it's a file path
    if isinstance(adata, str):
        adata_obj = anndata.read_h5ad(adata)
        file_path = adata
    else:
        adata_obj = adata
        file_path = None
    
    # Check if X_scGPT exists
    if 'X_scGPT' not in adata_obj.obsm:
        print("X_scGPT embedding is missing. Please compute scGPT embedding first.")
        return
    
    # Check if X_scANVI exists, if not compute it
    if 'X_scANVI' not in adata_obj.obsm:
        print("X_scANVI embedding is missing. Computing scANVI embedding...")
        adata_obj = preprocess_single_adata_for_scanvi(adata, batch_key="batch_id", save_output=False)
    
    # Check if X_scVI exists, if not compute it
    if 'X_scVI' not in adata_obj.obsm:
        print("X_scVI embedding is missing. Computing scVI embedding...")
        adata_obj = process_single_adata_for_scvi(adata, batch_key="batch_id", save_output=False)
    
    # Generate output filename with embedding suffixes
    if file_path:
        base_name, ext = os.path.splitext(file_path)
        # output_filename = f"{base_name}_X_scvi_X_scanvi_X_scGPT{ext}"
        output_filename = f"{base_name}_X_scvi_X_scanvi_X_scGPT_test{ext}"
        
    else:
        output_filename = "adata_with_embeddings_X_scvi_X_scanvi_X_scGPT.h5ad"
    
    # Save the final file with all embeddings
    adata_obj.write(output_filename)
    print(f"Final AnnData file with all embeddings saved to: {output_filename}")
    
    return adata_obj


if __name__ == "__main__":
    # compute_embeddings_scvi_scanvi_scgpt("adata_concat_scGPT_baron_2016h_xin_2016.h5ad")
    compute_embeddings_scvi_scanvi_scgpt("adata_concat_scGPT_baron_2016h_xin_2016_X_scvi_X_scanvi_X_scGPT.h5ad")