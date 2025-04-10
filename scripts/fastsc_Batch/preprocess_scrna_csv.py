import os
import numpy as np
import pandas as pd
import scanpy as sc

def process_pancreas_data(
    input_dir="./realdata/pancreas/raw/",
    output_dir="./realdata/pancreas/",
    include_mouse_data=False,
    n_neighbors=10,
    save_intermediate_files=False,
    plot_umap=True
):
    """
    Process pancreas single-cell RNA-seq data and create an H5AD file.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the raw data files
    output_dir : str
        Directory to save the output files
    include_mouse_data : bool
        Whether to include mouse data in the analysis
    n_neighbors : int
        Number of neighbors for scanpy's neighbors function
    save_intermediate_files : bool
        Whether to save intermediate CSV files
    plot_umap : bool
        Whether to create and save UMAP plot
        
    Returns:
    --------
    adata : AnnData
        Annotated data matrix with processed data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load human cell data
    human1 = pd.read_csv(os.path.join(input_dir, "GSM2230757_human1_umifm_counts.csv"), index_col=0)
    human2 = pd.read_csv(os.path.join(input_dir, "GSM2230758_human2_umifm_counts.csv"), index_col=0)
    human3 = pd.read_csv(os.path.join(input_dir, "GSM2230759_human3_umifm_counts.csv"), index_col=0)
    human4 = pd.read_csv(os.path.join(input_dir, "GSM2230760_human4_umifm_counts.csv"), index_col=0)

    # Drop unrelated columns
    human1 = human1.drop(["barcode"], axis=1)
    human2 = human2.drop(["barcode"], axis=1)
    human3 = human3.drop(["barcode"], axis=1)
    human4 = human4.drop(["barcode"], axis=1)
    
    # Add batch information
    human1["batch"] = 1
    human2["batch"] = 2
    human3["batch"] = 3
    human4["batch"] = 4
    
    if include_mouse_data:
        # Load mouse cell data
        mouse1 = pd.read_csv(os.path.join(input_dir, "GSM2230761_mouse1_umifm_counts.csv"), index_col=0)
        mouse2 = pd.read_csv(os.path.join(input_dir, "GSM2230762_mouse2_umifm_counts.csv"), index_col=0)
        
        # Drop unrelated columns
        mouse1 = mouse1.drop(["barcode", "PISD"], axis=1)
        mouse2 = mouse2.drop(["barcode", "PISD"], axis=1)
        
        # Normalize gene names
        mouse1.columns = [x.upper() if x != "assigned_cluster" else x for x in mouse1.columns]
        mouse2.columns = [x.upper() if x != "assigned_cluster" else x for x in mouse2.columns]
        
        # Add batch information
        mouse1["batch"] = 5
        mouse2["batch"] = 6
        
        # Merge all data
        res = pd.concat([human1, human2, human3, human4, mouse1, mouse2], axis=0).fillna(0)
    else:
        # Merge only human data
        res = pd.concat([human1, human2, human3, human4], axis=0).fillna(0)
    
    # Remove all-zero columns
    res = res.loc[:, (res != 0).any(axis=0)]

    # Data label split
    label = res["assigned_cluster"]
    batch = res["batch"]
    res = res.drop(["assigned_cluster", "batch"], axis=1)

    # Log-normalize data
    res = np.log1p(res)
    
    # Save intermediate files if requested
    if save_intermediate_files:
        res.to_csv(os.path.join(output_dir, "data.csv"))
        batch.to_csv(os.path.join(output_dir, "batch.csv"))
        label.to_csv(os.path.join(output_dir, "celltype.csv"))

    # Create AnnData object
    adata = sc.AnnData(res)
    adata.obs["batch"] = batch.values
    adata.obs["celltype"] = label.values
    
    print(adata)
    
    # Compute neighbors and UMAP
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=n_neighbors, metric='cosine')
    sc.tl.umap(adata)
    
    # Plot UMAP
    if plot_umap:
        sc.pl.umap(adata, color=["batch", "celltype"], wspace=0.5, 
                  save=f"_pancreas{'_with_mouse' if include_mouse_data else ''}.pdf")
    
    # Save AnnData object
    h5ad_filename = os.path.join(output_dir, f"data{'_with_mouse' if include_mouse_data else ''}.h5ad")
    adata.write(h5ad_filename)
    print(f"Data saved to {h5ad_filename}")
    
    return adata

# Example usage:
if __name__ == "__main__":
    # Process only human data
    adata_human = process_pancreas_data(
        input_dir="./realdata/pancreas/raw/",
        output_dir="./realdata/pancreas/",
        include_mouse_data=False,
        n_neighbors=10,
        save_intermediate_files=True,
        plot_umap=True
    )
    
    # Process human and mouse data
    adata_all = process_pancreas_data(
        input_dir="./realdata/pancreas/raw/",
        output_dir="./realdata/pancreas/",
        include_mouse_data=True,
        n_neighbors=10,
        save_intermediate_files=True,
        plot_umap=True
    )