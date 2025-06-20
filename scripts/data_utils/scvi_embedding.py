import anndata
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_for_scvi(file1, file2, label_column="labels", use_basename=True,
                                  batch_key="source", n_top_genes=2000, n_latent=30, qc_filtered = False):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    filters genes, prepares data for scVI, computes latent embeddings, and returns the two embedded AnnData objects.
    """

    # Load datasets
    adata1 = anndata.read_h5ad(file1)
    adata2 = anndata.read_h5ad(file2)       

    # Label source
    def extract_filename(path):
        return os.path.basename(path).rsplit('.h5ad', 1)[0]

    label1 = extract_filename(file1) if use_basename else file1
    label2 = extract_filename(file2) if use_basename else file2
    adata1.obs[batch_key] = pd.Categorical([label1] * adata1.n_obs)
    adata2.obs[batch_key] = pd.Categorical([label2] * adata2.n_obs)

    # Filter new data to match label column values - cell filtering
    if label_column in adata1.obs and label_column in adata2.obs:
        adata2 = adata2[adata2.obs[label_column].isin(adata1.obs[label_column])].copy()

    # Concatenate for joint preprocessing
    full = anndata.concat([adata1, adata2], join="outer", label=batch_key, keys=[label1, label2])

    # Use raw counts
    full.X = full.X.astype(int)
    full.raw = full.copy()
    
    #optional QC filtering
    if (qc_filtered):
        print("Applying qc filtering to remove cells with toofew genes and rarely expressed genes")
        sc.pp.filter_cells(full, min_genes=200)  # Remove cells with too few genes
        sc.pp.filter_genes(full, min_cells=3)    # Remove rarely expressed genes

    # Select highly variable genes
    sc.pp.highly_variable_genes(
        full,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        subset=True,
    )

    # Setup for scVI
    
    scvi._settings.ScviConfig(dl_num_workers=7)
    scvi.model.SCVI.setup_anndata(full, batch_key=batch_key)

    # Train scVI model
    model = scvi.model.SCVI(full, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    model.train(num_workers=7)

    # Store latent embeddings
    full.obsm["X_scVI"] = model.get_latent_representation()

    #Testing the integration
    sc.pp.neighbors(full, use_rep="X_scVI")
    sc.tl.umap(full)
    sc.pl.umap(full, color=batch_key)
    sc.pl.umap(full, color=label_column) #testing via the label_column by coloring base on the label_column


    # Split back the datasets
    adata1_out = full[full.obs[batch_key] == label1].copy()
    adata2_out = full[full.obs[batch_key] == label2].copy()
    
    # Save the processed datasets with _scvi suffix
    if (qc_filtered):
        output_file1 = os.path.join(os.path.dirname(file1), f"{label1}_qc_filtered_scvi.h5ad")
        output_file2 = os.path.join(os.path.dirname(file2), f"{label2}_qc_filtered_scvi.h5ad")
    else:
        output_file1 = os.path.join(os.path.dirname(file1), f"{label1}_scvi.h5ad")
        output_file2 = os.path.join(os.path.dirname(file2), f"{label2}_scvi.h5ad")
    
    print(f"Saving processed files to:\n{output_file1}\n{output_file2}")
    
    adata1_out.write(output_file1)
    adata2_out.write(output_file2)
    

    return adata1_out, adata2_out

import anndata
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def process_single_adata_for_scvi(file_path, batch_key=None, n_top_genes=2000, n_latent=30, 
                                  qc_filtered=False, use_basename=True, color_by=None, save_output=False):
    """
    Loads a single AnnData file, processes it for scVI, computes latent embeddings, 
    and returns the processed AnnData object with X_scVI embeddings.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
    batch_key : str, optional
        Column name in obs to use as batch key for scVI. If None, no batch correction is applied.
    n_top_genes : int
        Number of highly variable genes to select (default: 2000)
    n_latent : int
        Number of latent dimensions for scVI (default: 30)
    qc_filtered : bool
        Whether to apply quality control filtering (default: False)
    use_basename : bool
        Whether to use basename for output file naming (default: True)
    color_by : str or list, optional
        Column name(s) in obs to color UMAP plots by for visualization
    
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with X_scVI embeddings
    """
    
    # Load dataset
    if isinstance(file_path, str):
        adata = anndata.read_h5ad(file_path)
        file_path = file_path
    else:
        adata = file_path
        file_path = None
    print(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Use raw counts
    if hasattr(adata, 'raw') and adata.raw is not None:
        print("Using raw counts from adata.raw")
        adata.X = adata.raw.X.copy()
    
    adata.X = adata.X.astype(int)
    adata.raw = adata.copy()
    
    # Optional QC filtering
    if qc_filtered:
        print("Applying QC filtering to remove cells with too few genes and rarely expressed genes")
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        sc.pp.filter_cells(adata, min_genes=200)  # Remove cells with too few genes
        sc.pp.filter_genes(adata, min_cells=3)    # Remove rarely expressed genes
        
        print(f"After QC filtering: {adata.n_obs} cells ({n_cells_before - adata.n_obs} removed), "
              f"{adata.n_vars} genes ({n_genes_before - adata.n_vars} removed)")
    
    # Select highly variable genes
    print(f"Selecting {n_top_genes} highly variable genes")
    if batch_key and batch_key in adata.obs.columns:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            subset=True,
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            subset=True,
        )
    
    print(f"After gene filtering: {adata.n_vars} genes remain")
    
    # Setup for scVI
    if batch_key and batch_key in adata.obs.columns:
        print(f"Setting up scVI with batch key: {batch_key}")
        scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    else:
        print("Setting up scVI without batch correction")
        scvi.model.SCVI.setup_anndata(adata)
    
    # Train scVI model
    print("Training scVI model...")
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    model.train(num_workers=7)
    
    # Store latent embeddings
    print("Computing latent embeddings...")
    adata.obsm["X_scVI"] = model.get_latent_representation()
    
    # Compute UMAP for visualization
    print("Computing UMAP...")
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)
    
    # Visualization
    if batch_key and batch_key in adata.obs.columns:
        sc.pl.umap(adata, color=batch_key, title="UMAP colored by batch")
    
    if color_by:
        if isinstance(color_by, str):
            color_by = [color_by]
        for col in color_by:
            if col in adata.obs.columns:
                sc.pl.umap(adata, color=col, title=f"UMAP colored by {col}")
            else:
                print(f"Warning: Column '{col}' not found in adata.obs")
    
    # Save the processed dataset
    def extract_filename(path):
        return os.path.basename(path).rsplit('.h5ad', 1)[0]
    
    label = extract_filename(file_path) if use_basename else file_path
    
    if qc_filtered:
        output_file = os.path.join(os.path.dirname(file_path), f"{label}_qc_filtered_scvi.h5ad")
    else:
        output_file = os.path.join(os.path.dirname(file_path), f"{label}_scvi.h5ad")
    
    print(f"Saving processed file to: {output_file}")
    adata.write(output_file)
    
    if save_output:
        filename = os.path.basename(file_path).rsplit('.h5ad', 1)[0]
        print(f"Saving processed file to: {output_file}")
        adata.write(output_file)
    
    return adata

def process_single_adata_for_scvi_split_visualization(file_path, batch_key=None, n_top_genes=2000, n_latent=30, 
                                  qc_filtered=False, use_basename=True, color_by=None, save_output=False,
                                  plot_batch_labels=True, batch_values=None, figsize=(15, 6)):
    """
    Loads a single AnnData file, processes it for scVI, computes latent embeddings, 
    and returns the processed AnnData object with X_scVI embeddings.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
    batch_key : str, optional
        Column name in obs to use as batch key for scVI. If None, no batch correction is applied.
    n_top_genes : int
        Number of highly variable genes to select (default: 2000)
    n_latent : int
        Number of latent dimensions for scVI (default: 30)
    qc_filtered : bool
        Whether to apply quality control filtering (default: False)
    use_basename : bool
        Whether to use basename for output file naming (default: True)
    color_by : str or list, optional
        Column name(s) in obs to color UMAP plots by for visualization
    plot_batch_labels : bool
        Whether to create batch-specific label visualization (default: True)
    batch_values : list or None
        List of two batch values [batch1, batch2] for batch-specific plotting. If None, uses first two unique values
    figsize : tuple
        Figure size for batch-specific plots (width, height)
    
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with X_scVI embeddings
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.cm as cm
    
    # Load dataset
    if isinstance(file_path, str):
        adata = anndata.read_h5ad(file_path)
        file_path = file_path
    else:
        adata = file_path
        file_path = None
    print(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Use raw counts
    if hasattr(adata, 'raw') and adata.raw is not None:
        print("Using raw counts from adata.raw")
        adata.X = adata.raw.X.copy()
    
    adata.X = adata.X.astype(int)
    adata.raw = adata.copy()
    
    # Optional QC filtering
    if qc_filtered:
        print("Applying QC filtering to remove cells with too few genes and rarely expressed genes")
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        sc.pp.filter_cells(adata, min_genes=200)  # Remove cells with too few genes
        sc.pp.filter_genes(adata, min_cells=3)    # Remove rarely expressed genes
        
        print(f"After QC filtering: {adata.n_obs} cells ({n_cells_before - adata.n_obs} removed), "
              f"{adata.n_vars} genes ({n_genes_before - adata.n_vars} removed)")
    
    # Select highly variable genes
    print(f"Selecting {n_top_genes} highly variable genes")
    if batch_key and batch_key in adata.obs.columns:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            subset=True,
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            subset=True,
        )
    
    print(f"After gene filtering: {adata.n_vars} genes remain")
    
    # Setup for scVI
    if batch_key and batch_key in adata.obs.columns:
        print(f"Setting up scVI with batch key: {batch_key}")
        scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    else:
        print("Setting up scVI without batch correction")
        scvi.model.SCVI.setup_anndata(adata)
    
    # Train scVI model
    print("Training scVI model...")
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    model.train(num_workers=7)
    
    # Store latent embeddings
    print("Computing latent embeddings...")
    adata.obsm["X_scVI"] = model.get_latent_representation()
    
    # Compute UMAP for visualization
    print("Computing UMAP...")
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)
    
    # Enhanced visualization with batch-specific label plotting
    if plot_batch_labels and batch_key and batch_key in adata.obs.columns and "labels" in adata.obs.columns:
        print("Creating batch-specific label visualization...")
        
        # Check if we have at least 2 batches
        unique_batches = adata.obs[batch_key].unique()
        if len(unique_batches) >= 2:
            # Get batch values
            if batch_values is None:
                batch_values = unique_batches[:2]
                print(f"Using batch values for scvi visualization: {batch_values}")
            
            batch1, batch2 = batch_values
            
            # Get UMAP coordinates
            umap_coords = adata.obsm['X_umap']
            title_suffix = " (scVI Corrected)"
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Get unique labels and create color map
            unique_labels = adata.obs["labels"].unique()
            n_labels = len(unique_labels)
            
            # Use a colormap with distinct colors
            colors = cm.tab20(np.linspace(0, 1, n_labels)) if n_labels <= 20 else cm.hsv(np.linspace(0, 1, n_labels))
            label_color_map = dict(zip(unique_labels, colors))
            
            # Plot 1: All labels with normal colors
            for i, label in enumerate(unique_labels):
                mask = adata.obs["labels"] == label
                axes[0].scatter(
                    umap_coords[mask, 0], 
                    umap_coords[mask, 1],
                    c=[label_color_map[label]], 
                    label=label,
                    s=20,
                    alpha=0.7
                )
            
            axes[0].set_title(f'All Labels{title_suffix}')
            axes[0].set_xlabel('UMAP 1')
            axes[0].set_ylabel('UMAP 2')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Plot 2: Batch-specific coloring
            # First plot batch1 cells in grey
            batch1_mask = adata.obs[batch_key] == batch1
            if batch1_mask.sum() > 0:
                axes[1].scatter(
                    umap_coords[batch1_mask, 0], 
                    umap_coords[batch1_mask, 1],
                    c='lightgrey', 
                    label=f'{batch1} (all labels)',
                    s=20,
                    alpha=0.5
                )
            
            # Then plot batch2 cells with their label colors
            batch2_mask = adata.obs[batch_key] == batch2
            if batch2_mask.sum() > 0:
                batch2_data = adata.obs[batch2_mask]
                for label in unique_labels:
                    label_mask = batch2_data["labels"] == label
                    if label_mask.sum() > 0:
                        # Get indices in the full dataset
                        full_indices = batch2_data.index[label_mask]
                        full_mask = adata.obs.index.isin(full_indices)
                        
                        axes[1].scatter(
                            umap_coords[full_mask, 0], 
                            umap_coords[full_mask, 1],
                            c=[label_color_map[label]], 
                            label=f'{batch2}: {label}',
                            s=20,
                            alpha=0.7
                        )
            
            axes[1].set_title(f'{batch1} (Grey) vs {batch2} (Colored){title_suffix}')
            axes[1].set_xlabel('UMAP 1')
            axes[1].set_ylabel('UMAP 2')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            plt.show()
            
            # Print some statistics
            print(f"\nStatistics:")
            print(f"Total cells: {adata.n_obs}")
            print(f"Batch '{batch1}': {batch1_mask.sum()} cells")
            print(f"Batch '{batch2}': {batch2_mask.sum()} cells")
            print(f"Labels: {list(unique_labels)}")
            
            # Show label distribution per batch
            cross_tab = pd.crosstab(adata.obs[batch_key], adata.obs["labels"])
            print(f"\nLabel distribution per batch:")
            print(cross_tab)
        else:
            print(f"Warning: Need at least 2 batches for batch-specific visualization, found {len(unique_batches)}")
            # Fall back to simple batch visualization
            sc.pl.umap(adata, color=batch_key, title="UMAP colored by batch")
    else:
        # Original visualization logic
        if batch_key and batch_key in adata.obs.columns:
            sc.pl.umap(adata, color=batch_key, title="UMAP colored by batch")
        
        if color_by:
            if isinstance(color_by, str):
                color_by = [color_by]
            for col in color_by:
                if col in adata.obs.columns:
                    sc.pl.umap(adata, color=col, title=f"UMAP colored by {col}")
                else:
                    print(f"Warning: Column '{col}' not found in adata.obs")
    
    # Save the processed dataset
    def extract_filename(path):
        return os.path.basename(path).rsplit('.h5ad', 1)[0]
    
    if file_path:
        label = extract_filename(file_path) if use_basename else file_path
        
        if qc_filtered:
            output_file = os.path.join(os.path.dirname(file_path), f"{label}_qc_filtered_scvi.h5ad")
        else:
            output_file = os.path.join(os.path.dirname(file_path), f"{label}_scvi.h5ad")
        
        print(f"Saving processed file to: {output_file}")
        adata.write(output_file)
        
        if save_output:
            filename = os.path.basename(file_path).rsplit('.h5ad', 1)[0]
            print(f"Saving processed file to: {output_file}")
            adata.write(output_file)
    
    return adata