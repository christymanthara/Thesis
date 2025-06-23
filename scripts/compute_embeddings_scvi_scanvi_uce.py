import anndata
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.neighbors import KNeighborsClassifier
# from .data_utils.scanvi_embedding import preprocess_single_adata_for_scanvi, preprocess_single_adata_for_scanvi_split_visualization,preprocess_single_adata_for_scanvi_split_visualization_without_plots
from .data_utils.scanvi_embedding import preprocess_single_adata_for_scanvi, preprocess_single_adata_for_scanvi_split_visualization,preprocess_single_adata_for_scanvi_split_visualization_without_plots_with_fingerprint
from .data_utils.scvi_embedding import process_single_adata_for_scvi, process_single_adata_for_scvi_split_visualization
import scripts.utils
import os

def compute_embeddings_scvi_scanvi_uce(
    adata: str,
    embedding_dir="embeddings",
    force_recompute=False
):
    """
    Compute embeddings for UCE, scANVI, and scGPT if missing.
    
    Args:
        adata: Path to the AnnData file or AnnData object
    """
    # Load adata if it's a file path
    if isinstance(adata, str):
        adata_obj = anndata.read_h5ad(adata)
        print(f"Loaded AnnData object from file")
        file_path = adata
    else:
        adata_obj = adata
        file_path = None
    
    # Check if X_uce exists
    # if 'X_uce' not in adata_obj.obsm:
    #     print("X_uce embedding is missing. Please compute UCE embedding first.")
    #     return
    
    # original_x_uce = adata_obj.obsm['X_uce'].copy()
    
    # Check if both embeddings are missing
    print("Unique source labels in adata while computing embeddings:", adata_obj.obs["source"].unique())
    
    missing_scvi = 'X_scVI' not in adata_obj.obsm
    missing_scanvi = 'X_scANVI' not in adata_obj.obsm
    
    if missing_scvi and missing_scanvi:
        # Both are missing - compute scANVI first (which includes scVI as initialization)
        print("Both X_scVI and X_scANVI embeddings are missing. Computing both...")
        # adata_obj = preprocess_single_adata_for_scanvi(adata, batch_key="batch", save_output=False) #batch stores the dataset name.
        adata_obj = preprocess_single_adata_for_scanvi_split_visualization_without_plots_with_fingerprint(
            adata_obj, 
            batch_key="batch",
            label_column="labels", 
            plot_batch_labels=True,
            embedding_dir=embedding_dir,
            force_recompute=force_recompute
        ) #batch stores the dataset name.
        # Restore the original X_uce embedding
        # adata_obj.obsm['X_uce'] = original_x_uce
        missing_scvi = 'X_scVI' not in adata_obj.obsm
        missing_scanvi = 'X_scANVI' not in adata_obj.obsm
    
    elif missing_scanvi and not missing_scvi:
        # Only scANVI is missing
        print("X_scANVI embedding is missing. Computing scANVI embedding...")
        # temp_adata = preprocess_single_adata_for_scanvi(adata, batch_key="batch", save_output=False)
        temp_adata = preprocess_single_adata_for_scanvi_split_visualization(adata, batch_key="batch",label_column="labels", plot_batch_labels=True) #batch stores the dataset name.
        
        # Copy the scANVI embedding to the original object
        adata_obj.obsm['X_scANVI'] = temp_adata.obsm['X_scANVI']
        
    elif missing_scvi and not missing_scanvi:
        # Only scVI is missing
        print("X_scVI embedding is missing. Computing scVI embedding...")
        # temp_adata = process_single_adata_for_scvi(adata, batch_key="batch", save_output=False)
        temp_adata = process_single_adata_for_scvi_split_visualization(adata, batch_key="batch", plot_batch_labels=True,)
        # Copy the scVI embedding to the original object
        adata_obj.obsm['X_scVI'] = temp_adata.obsm['X_scVI']
    
    else:
        print("Both X_scVI and X_scANVI embeddings already exist.")
    
    # Verify all embeddings are present
    print(f"Final embeddings in adata.obsm: {list(adata_obj.obsm.keys())}")
    
    # Generate output filename with embedding suffixes
    if file_path:
        base_name, ext = os.path.splitext(file_path)
        # output_filename = f"{base_name}_X_scvi_X_scanvi_X_scGPT{ext}"
        output_filename = f"{base_name}_X_scvi_X_scanvi_X_scGPT_test{ext}"
        
    else:
        output_filename = "adata_with_embeddings_X_scvi_X_scanvi_X_uce.h5ad"
    
    # Save the final file with all embeddings
    adata_obj.write(output_filename)
    print(f"Final AnnData file with all embeddings saved to: {output_filename}")
    
    import sys
    import gc
    
    # Clear matplotlib figures
    if 'matplotlib.pyplot' in sys.modules:
        import matplotlib.pyplot as plt
        plt.close('all')
    
    # Clear CUDA cache if using GPU
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Remove heavy modules
    modules_to_remove = ['scvi', 'matplotlib', 'seaborn']
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
    
    # Force garbage collection
    gc.collect()
    
    return adata_obj







if __name__ == "__main__":
    # compute_embeddings_scvi_scanvi_scgpt("adata_concat_scGPT_baron_2016h_xin_2016.h5ad")
    # compute_embeddings_scvi_scanvi_scgpt("adata_concat_scGPT_baron_2016h_xin_2016_X_scvi_X_scanvi_X_scGPT.h5ad")
    compute_embeddings_scvi_scanvi_uce("F:/Thesis/chen_2017hrvatin_2018_uce_adata.h5ad")