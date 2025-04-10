import os
import scanpy as sc
import numpy as np

def convert_h5ad_to_pancreas_format(
    input_h5ad_path,
    output_h5ad_path,
    batch_key='batch_id',
    cell_type_key='labels',
    n_neighbors=10,
    log_normalize=True,
    plot_umap=True
):
    """
    Convert an existing H5AD file to match the pancreas data format.
    
    Parameters:
    -----------
    input_h5ad_path : str
        Path to the input H5AD file
    output_h5ad_path : str
        Path where the converted H5AD file will be saved
    batch_key : str
        Key in adata.obs containing batch information
    cell_type_key : str
        Key in adata.obs containing cell type labels
    n_neighbors : int
        Number of neighbors for scanpy's neighbors function
    log_normalize : bool
        Whether to apply log normalization to the data
    plot_umap : bool
        Whether to create and save UMAP plot
        
    Returns:
    --------
    adata : AnnData
        Converted AnnData object
    """
    # Load the existing H5AD file
    print(f"Loading H5AD file from: {input_h5ad_path}")
    adata = sc.read_h5ad(input_h5ad_path)
    print(f"Loaded AnnData object: {adata}")
    
    # Create a copy to avoid modifying the original data
    adata_new = adata.copy()
    
    # Rename observations to match expected format
    if batch_key in adata_new.obs and batch_key != 'batch':
        adata_new.obs['batch'] = adata_new.obs[batch_key]
    
    if cell_type_key in adata_new.obs and cell_type_key != 'celltype':
        adata_new.obs['celltype'] = adata_new.obs[cell_type_key]
    
    # Ensure batch is numeric
    if 'batch' in adata_new.obs and not pd.api.types.is_numeric_dtype(adata_new.obs['batch']):
        # Convert categorical or string batch to numeric
        batch_categories = adata_new.obs['batch'].astype('category').cat.categories
        batch_map = {cat: i+1 for i, cat in enumerate(batch_categories)}
        adata_new.obs['batch'] = adata_new.obs['batch'].map(batch_map).astype('int')
    
    # Log-normalize data if requested and not already normalized
    if log_normalize:
        # Check if data is already log-normalized
        data_max = adata_new.X.max()
        if isinstance(adata_new.X, np.ndarray):
            is_likely_normalized = data_max < 30  # Heuristic threshold
        else:  # For sparse matrices
            is_likely_normalized = data_max < 30
            
        if not is_likely_normalized:
            print("Log-normalizing data...")
            sc.pp.normalize_total(adata_new, target_sum=1e4)
            sc.pp.log1p(adata_new)
    
    # Compute neighbors and UMAP
    print("Computing neighbors and UMAP embedding...")
    sc.pp.neighbors(adata_new, n_neighbors=n_neighbors, metric='cosine')
    sc.tl.umap(adata_new)
    
    # Plot UMAP
    if plot_umap:
        output_dir = os.path.dirname(output_h5ad_path)
        if output_dir:  # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
        plot_name = os.path.splitext(os.path.basename(output_h5ad_path))[0]
        print(f"Creating UMAP plot: {plot_name}")
        sc.pl.umap(adata_new, color=['batch', 'celltype'], wspace=0.5, 
                   save=f"_{plot_name}.pdf")
    
    # Save converted AnnData object
    print(f"Saving converted H5AD file to: {output_h5ad_path}")
    adata_new.write(output_h5ad_path)
    
    return adata_new

# Example usage:
if __name__ == "__main__":
    import pandas as pd
    
    # Convert existing H5AD file to pancreas format
    input_path = "Datasets/baron_2016h.h5ad"
    output_path = "./realdata/pancreas/converted_data.h5ad"
    
    adata_converted = convert_h5ad_to_pancreas_format(
        input_h5ad_path=input_path,
        output_h5ad_path=output_path,
        batch_key='batch_id',  # Original batch key in your data
        cell_type_key='labels',  # Original cell type key in your data
        n_neighbors=10,
        log_normalize=True,
        plot_umap=True
    )