import anndata
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import multiprocessing
num_workers = multiprocessing.cpu_count()
from scripts.data_utils.test_save_embeddings import compute_or_load_embedding

def load_and_preprocess_for_scanvi(file1, file2, label_column="labels", use_basename=True,
                                   batch_key="source", n_top_genes=2000, n_latent=30):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    filters genes, prepares data for scANVI, computes latent embeddings, and returns the two embedded AnnData objects.
    
    This version uses scANVI for semi-supervised integration rather than scVI.
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

    # Filter new data to match label column values
    if label_column in adata1.obs and label_column in adata2.obs:
        adata2 = adata2[adata2.obs[label_column].isin(adata1.obs[label_column])].copy()

    # Concatenate for joint preprocessing
    full = anndata.concat([adata1, adata2], join="outer", label=batch_key, keys=[label1, label2])

    # Use raw counts
    full.X = full.X.astype(int)
    full.raw = full.copy()

    # Select highly variable genes
    sc.pp.highly_variable_genes(
        full,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        subset=True,
    )

    # Make sure the label column is categorical
    if label_column in full.obs:
        full.obs[label_column] = full.obs[label_column].astype('category')
    
    # IMPORTANT: Create a fresh copy of the AnnData object before setup
    # This avoids the "setup with a different model" error
    full_copy = full.copy()
    
    # Setup for scANVI directly - don't reuse AnnData that was set up with a different model
    scvi.model.SCANVI.setup_anndata(
        full_copy, 
        batch_key=batch_key,
        labels_key=label_column,
        unlabeled_category="Unknown"  # Category for unlabeled cells if any
    )

    # First train a scVI model as initialization for scANVI

    scvi.model.SCVI.setup_anndata(full_copy, batch_key=batch_key)
    vae = scvi.model.SCVI(full_copy, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    vae.train()
    
    # Initialize and train the scANVI model
    model = scvi.model.SCANVI.from_scvi_model(vae,adata =full_copy,labels_key=label_column, unlabeled_category="Unknown")
    model.train(max_epochs=20)  # Typically needs fewer epochs when initialized from scVI

    # Store latent embeddings
    full_copy.obsm["X_scANVI"] = model.get_latent_representation()

    # Testing the integration
    sc.pp.neighbors(full_copy, use_rep="X_scANVI")
    sc.tl.umap(full_copy)
    sc.pl.umap(full_copy, color=batch_key)
    sc.pl.umap(full_copy, color=label_column)  # Testing via the label_column

    # Split back the datasets
    adata1_out = full_copy[full_copy.obs[batch_key] == label1].copy()
    adata2_out = full_copy[full_copy.obs[batch_key] == label2].copy()
    
    # Save the processed datasets with _scvi suffix
    output_file1 = os.path.join(os.path.dirname(file1), f"{label1}_scanvi.h5ad")
    output_file2 = os.path.join(os.path.dirname(file2), f"{label2}_scanvi.h5ad")
    
    print(f"Saving processed files to:\n{output_file1}\n{output_file2}")
    
    adata1_out.write(output_file1)
    adata2_out.write(output_file2)

    return adata1_out, adata2_out

def preprocess_single_adata_for_scanvi(file_path, label_column="labels", batch_key=None, 
                                       n_top_genes=3000, n_latent=30, save_output=False):
    """
    Loads a single AnnData file, preprocesses it for scANVI, computes latent embeddings, 
    and returns the processed AnnData object with X_scANVI embeddings.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
    label_column : str, default "labels"
        Column name containing cell type labels
    batch_key : str, optional
        Column name for batch information. If None, no batch correction is applied
    n_top_genes : int, default 2000
        Number of highly variable genes to select
    n_latent : int, default 30
        Number of latent dimensions
    save_output : bool, default True
        Whether to save the processed file
        
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with X_scANVI embeddings
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
    adata.X = adata.X.astype(int)
    adata.raw = adata.copy()
    
    # Select highly variable genes
    if batch_key is not None:
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
    
    print(f"Selected {adata.n_vars} highly variable genes")
    
    # Make sure the label column is categorical
    if label_column in adata.obs:
        adata.obs[label_column] = adata.obs[label_column].astype('category')
        print(f"Found {len(adata.obs[label_column].cat.categories)} unique labels in {label_column}")
    else:
        raise ValueError(f"Label column '{label_column}' not found in adata.obs")
    
    # Create a fresh copy to avoid setup conflicts
    adata_copy = adata.copy()
    
    # Setup for scVI first (as initialization for scANVI)
    scvi._settings.ScviConfig(dl_num_workers=7)
    if batch_key is not None:
        scvi.model.SCVI.setup_anndata(adata_copy, batch_key=batch_key)
    else:
        scvi.model.SCVI.setup_anndata(adata_copy)
    
    # Train scVI model
    print("Training scVI model for initialization...")
    vae = scvi.model.SCVI(adata_copy, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    vae.train()
    
     # Store latent embeddings
    print("Computing latent embeddings...")
    adata_copy.obsm["X_scVI"] = vae.get_latent_representation()
    
    # Setup and train scANVI model
    print("Setting up and training scANVI model...")
    
    # Setup scANVI on the same data
    if batch_key is not None:
        scvi.model.SCANVI.setup_anndata(
            adata_copy, 
            batch_key=batch_key,
            labels_key=label_column,
            unlabeled_category="Unknown"
        )
    else:
        scvi.model.SCANVI.setup_anndata(
            adata_copy, 
            labels_key=label_column,
            unlabeled_category="Unknown"
        )
    
    # Initialize scANVI from the trained scVI model
    model = scvi.model.SCANVI.from_scvi_model(
        vae, 
        adata=adata_copy, 
        labels_key=label_column, 
        unlabeled_category="Unknown"
    )
    
    # Train scANVI model
    model.train(max_epochs=200, early_stopping=True, early_stopping_patience=10)
    
    # Store latent embeddings
    print("Computing latent embeddings...")
    adata_copy.obsm["X_scANVI"] = model.get_latent_representation()
    
    # Compute neighborhood graph and UMAP for visualization
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata_copy, use_rep="X_scANVI")
    sc.tl.umap(adata_copy)
    
    # Generate visualization plots
    if batch_key is not None:
        sc.pl.umap(adata_copy, color=batch_key, title="scANVI integration - Batch")
    sc.pl.umap(adata_copy, color=label_column, title=f"scANVI integration - {label_column}")
    
    # Save the processed dataset
    if save_output:
        filename = os.path.basename(file_path).rsplit('.h5ad', 1)[0]
        output_file = os.path.join(os.path.dirname(file_path), f"{filename}_scanvi.h5ad")
        print(f"Saving processed file to: {output_file}")
        adata_copy.write(output_file)
    
    return adata_copy


def preprocess_single_adata_for_scanvi_split_visualization(file_path, label_column="labels", batch_key=None, 
                                       n_top_genes=3000, n_latent=30, save_output=False,
                                       plot_batch_labels=True, batch_values=None, figsize=(15, 6)):
    """
    Loads a single AnnData file, preprocesses it for scANVI, computes latent embeddings, 
    and returns the processed AnnData object with X_scANVI embeddings.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
    label_column : str, default "labels"
        Column name containing cell type labels
    batch_key : str, optional
        Column name for batch information. If None, no batch correction is applied
    n_top_genes : int, default 3000
        Number of highly variable genes to select
    n_latent : int, default 30
        Number of latent dimensions
    save_output : bool, default False
        Whether to save the processed file
    plot_batch_labels : bool, default True
        Whether to create batch-specific label visualization
    batch_values : list or None
        List of two batch values [batch1, batch2] for batch-specific plotting. If None, uses first two unique values
    figsize : tuple, default (15, 6)
        Figure size for batch-specific plots (width, height)
        
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with X_scANVI embeddings and preserved .uns data
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
    
    # PRESERVE ORIGINAL .uns DATA BEFORE ANY PROCESSING
    original_uns = adata.uns.copy()
    print(f"📦 Preserved original .uns with keys: {list(original_uns.keys())}")
    
    # Create parameter dictionary for embedding cache
    embedding_params = {
        'n_top_genes': n_top_genes,
        'n_latent': n_latent,
        'batch_key': batch_key if batch_key else 'none',
        'label_column': label_column
    }
    
    # Use raw counts
    adata.X = adata.X.astype(int)
    
    # PRESERVE .uns WHEN SETTING RAW
    adata.raw = adata.copy()
    # Restore .uns after raw assignment
    adata.uns = original_uns.copy()
    
    # Select highly variable genes
    if batch_key is not None:
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
    
    print(f"Selected {adata.n_vars} highly variable genes")
    
    # RESTORE .uns AFTER GENE FILTERING (subset=True can affect .uns)
    adata.uns.update(original_uns)
    print(f"📦 Restored .uns after gene filtering: {list(adata.uns.keys())}")
    
    # Make sure the label column is categorical
    if label_column in adata.obs:
        adata.obs[label_column] = adata.obs[label_column].astype('category')
        print(f"Found {len(adata.obs[label_column].cat.categories)} unique labels in {label_column}")
    else:
        raise ValueError(f"Label column '{label_column}' not found in adata.obs")
    
    # Create a fresh copy to avoid setup conflicts
    adata_copy = adata.copy()
    
    # PRESERVE .uns IN THE COPY
    adata_copy.uns = original_uns.copy()
    print(f"📦 Preserved .uns in adata_copy: {list(adata_copy.uns.keys())}")
    
    # Setup for scVI first (as initialization for scANVI)
    scvi._settings.ScviConfig(dl_num_workers=7)
    
    # BACKUP .uns BEFORE scVI SETUP (scVI setup modifies .uns extensively)
    uns_backup_before_scvi = adata_copy.uns.copy()
    
    if batch_key is not None:
        scvi.model.SCVI.setup_anndata(adata_copy, batch_key=batch_key)
    else:
        scvi.model.SCVI.setup_anndata(adata_copy)
    
    # MERGE scVI-specific .uns data with original data
    scvi_uns_keys = set(adata_copy.uns.keys()) - set(uns_backup_before_scvi.keys())
    print(f"📦 scVI added these .uns keys: {scvi_uns_keys}")
    
    # Keep scVI-specific keys but restore original data
    merged_uns = uns_backup_before_scvi.copy()
    for key in scvi_uns_keys:
        merged_uns[f"scvi_{key}"] = adata_copy.uns[key]
    
    adata_copy.uns = merged_uns
    
    # Train scVI model
    print("Training scVI model for initialization...")
    vae = scvi.model.SCVI(adata_copy, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    vae.train()
    
    # Store latent embeddings
    print("Computing latent embeddings...")
    adata_copy.obsm["X_scVI"] = vae.get_latent_representation()
    
    # Setup and train scANVI model
    print("Setting up and training scANVI model...")
    
    # BACKUP .uns BEFORE scANVI SETUP
    uns_backup_before_scanvi = adata_copy.uns.copy()
    
    # Setup scANVI on the same data
    if batch_key is not None:
        scvi.model.SCANVI.setup_anndata(
            adata_copy, 
            batch_key=batch_key,
            labels_key=label_column,
            unlabeled_category="Unknown"
        )
    else:
        scvi.model.SCANVI.setup_anndata(
            adata_copy, 
            labels_key=label_column,
            unlabeled_category="Unknown"
        )
    
    # MERGE scANVI-specific .uns data with existing data
    scanvi_uns_keys = set(adata_copy.uns.keys()) - set(uns_backup_before_scanvi.keys())
    print(f"📦 scANVI added these .uns keys: {scanvi_uns_keys}")
    
    # Keep scANVI-specific keys but restore previous data
    final_uns = uns_backup_before_scanvi.copy()
    for key in scanvi_uns_keys:
        final_uns[f"scanvi_{key}"] = adata_copy.uns[key]
    
    adata_copy.uns = final_uns
    
    # Initialize scANVI from the trained scVI model
    model = scvi.model.SCANVI.from_scvi_model(
        vae, 
        adata=adata_copy, 
        labels_key=label_column, 
        unlabeled_category="Unknown"
    )
    
    # Train scANVI model
    model.train(max_epochs=200, early_stopping=True, early_stopping_patience=10)
    
    # Store latent embeddings
    print("Computing latent embeddings...")
    adata_copy.obsm["X_scANVI"] = model.get_latent_representation()
    
    # Compute neighborhood graph and UMAP for visualization
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata_copy, use_rep="X_scANVI")
    sc.tl.umap(adata_copy)
    
    # Enhanced visualization with batch-specific label plotting
    if plot_batch_labels and batch_key is not None and batch_key in adata_copy.obs.columns and label_column in adata_copy.obs.columns:
        print("Creating batch-specific label visualization...")
        
        # Check if we have at least 2 batches
        unique_batches = adata_copy.obs[batch_key].unique()
        if len(unique_batches) >= 2:
            # Get batch values
            if batch_values is None:
                batch_values = unique_batches[:2]
                print(f"Using batch values: {batch_values}")
            
            batch1, batch2 = batch_values
            
            # Get UMAP coordinates
            umap_coords = adata_copy.obsm['X_umap']
            title_suffix = " (scANVI Corrected)"
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Get unique labels and create color map
            unique_labels = adata_copy.obs[label_column].unique()
            n_labels = len(unique_labels)
            
            # Use a colormap with distinct colors
            colors = cm.tab20(np.linspace(0, 1, n_labels)) if n_labels <= 20 else cm.hsv(np.linspace(0, 1, n_labels))
            label_color_map = dict(zip(unique_labels, colors))
            
            # Plot 1: Only batch1 cells with their label colors
            print(f"Plotting batch1-specific visualization for {batch1}...")
            batch1_mask = adata_copy.obs[batch_key] == batch1
            if batch1_mask.sum() > 0:
                batch1_data = adata_copy.obs[batch1_mask]
                for label in unique_labels:
                    label_mask = batch1_data[label_column] == label
                    if label_mask.sum() > 0:
                        # Get indices in the full dataset
                        full_indices = batch1_data.index[label_mask]
                        full_mask = adata_copy.obs.index.isin(full_indices)
                        
                        axes[0].scatter(
                            umap_coords[full_mask, 0], 
                            umap_coords[full_mask, 1],
                            c=[label_color_map[label]], 
                            label=f'{label}',
                            s=20,
                            alpha=0.7
                        )

            axes[0].set_title(f'{batch1} Labels{title_suffix}')
            axes[0].set_xlabel('UMAP 1')
            axes[0].set_ylabel('UMAP 2')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Plot 2: Batch-specific coloring
            # First plot batch1 cells in grey
            print(f"Plotting batch-specific visualization Plot 2 for {batch1} and {batch2}...")
            print(f"Batch 1: {batch1}, Batch 2: {batch2}")
            batch1_mask = adata_copy.obs[batch_key] == batch1
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
            batch2_mask = adata_copy.obs[batch_key] == batch2
            if batch2_mask.sum() > 0:
                batch2_data = adata_copy.obs[batch2_mask]
                for label in unique_labels:
                    label_mask = batch2_data[label_column] == label
                    if label_mask.sum() > 0:
                        # Get indices in the full dataset
                        full_indices = batch2_data.index[label_mask]
                        full_mask = adata_copy.obs.index.isin(full_indices)
                        
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
            print(f"Total cells: {adata_copy.n_obs}")
            print(f"Batch '{batch1}': {batch1_mask.sum()} cells")
            print(f"Batch '{batch2}': {batch2_mask.sum()} cells")
            print(f"Labels: {list(unique_labels)}")
            
            # Show label distribution per batch
            cross_tab = pd.crosstab(adata_copy.obs[batch_key], adata_copy.obs[label_column])
            print(f"\nLabel distribution per batch:")
            print(cross_tab)
            
            # Additionally, create a classification report if we have predictions
            if hasattr(model, 'predict') and batch_key is not None:
                print("\nGenerating scANVI predictions...")
                predictions = model.predict()
                adata_copy.obs[f'{label_column}_predicted'] = predictions
                
                # Show prediction accuracy per batch
                from sklearn.metrics import classification_report
                
                for batch in [batch1, batch2]:
                    batch_mask = adata_copy.obs[batch_key] == batch
                    if batch_mask.sum() > 0:
                        y_true = adata_copy.obs[batch_mask][label_column]
                        y_pred = adata_copy.obs[batch_mask][f'{label_column}_predicted']
                        
                        print(f"\nClassification report for {batch}:")
                        print(classification_report(y_true, y_pred, zero_division=0))
        else:
            print(f"Warning: Need at least 2 batches for batch-specific visualization, found {len(unique_batches)}")
            # Fall back to simple visualization
            if batch_key is not None:
                sc.pl.umap(adata_copy, color=batch_key, title="scANVI integration - Batch")
            sc.pl.umap(adata_copy, color=label_column, title=f"scANVI integration - {label_column}")
    else:
        # Original visualization logic
        if batch_key is not None:
            sc.pl.umap(adata_copy, color=batch_key, title="scANVI integration - Batch")
        sc.pl.umap(adata_copy, color=label_column, title=f"scANVI integration - {label_column}")
    
    # FINAL VERIFICATION AND RESTORATION OF .uns
    print(f"📦 Final .uns keys before return: {list(adata_copy.uns.keys())}")
    
    # Ensure we still have the most important original .uns data
    for key, value in original_uns.items():
        if key not in adata_copy.uns:
            adata_copy.uns[key] = value
            print(f"📦 Restored missing original .uns key: {key}")
    
    print(f"📦 Final .uns keys after restoration: {list(adata_copy.uns.keys())}")
    
    # Save the processed dataset
    if save_output and file_path:
        filename = os.path.basename(file_path).rsplit('.h5ad', 1)[0]
        output_file = os.path.join(os.path.dirname(file_path), f"{filename}_scanvi.h5ad")
        print(f"Saving processed file to: {output_file}")
        adata_copy.write(output_file)
    
    return adata_copy




def preprocess_single_adata_for_scanvi_split_visualization_without_plots(file_path, label_column="labels", batch_key=None, 
                                       n_top_genes=3000, n_latent=30, save_output=False,
                                       plot_batch_labels=True, batch_values=None, figsize=(15, 6),
                                       embedding_dir="embeddings", force_recompute=False):
    """
    Loads a single AnnData file, preprocesses it for scANVI, computes latent embeddings, 
    and returns the processed AnnData object with X_scANVI embeddings.
    Now supports caching of embeddings to avoid recomputation and preserves uns data.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
    label_column : str, default "labels"
        Column name containing cell type labels
    batch_key : str, optional
        Column name for batch information. If None, no batch correction is applied
    n_top_genes : int, default 3000
        Number of highly variable genes to select
    n_latent : int, default 30
        Number of latent dimensions
    save_output : bool, default False
        Whether to save the processed file
    plot_batch_labels : bool, default True
        Whether to create batch-specific label visualization
    batch_values : list or None
        List of two batch values [batch1, batch2] for batch-specific plotting. If None, uses first two unique values
    figsize : tuple, default (15, 6)
        Figure size for batch-specific plots (width, height)
    embedding_dir : str, default "embeddings"
        Directory to save/load embeddings
    force_recompute : bool, default False
        If True, always recompute embeddings even if cached versions exist
        
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with X_scANVI embeddings and preserved uns data
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.cm as cm
    
    # Load dataset
    if isinstance(file_path, str):
        adata = anndata.read_h5ad(file_path)
        dataset_name = os.path.basename(file_path).rsplit('.h5ad', 1)[0]
        file_path_str = file_path
    else:
        adata = file_path
        dataset_name = "anndata_object"
        file_path_str = None
    
    print(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Store original uns data to preserve it
    original_uns = adata.uns.copy() if adata.uns else {}
    
    # Create parameter dictionary for embedding cache
    embedding_params = {
        'n_top_genes': n_top_genes,
        'n_latent': n_latent,
        'batch_key': batch_key if batch_key else 'none',
        'label_column': label_column
    }
    
    # Define embedding computation functions
    def compute_scvi_embedding(adata_input, **params):
        """Compute scVI embedding"""
        # Create a working copy to avoid modifying the original
        adata_work = adata_input.copy()
        
        # Use raw counts
        adata_work.X = adata_work.X.astype(int)
        adata_work.raw = adata_work.copy()
        
        # Select highly variable genes
        if params['batch_key'] != 'none':
            sc.pp.highly_variable_genes(
                adata_work,
                flavor="seurat_v3",
                n_top_genes=params['n_top_genes'],
                batch_key=params['batch_key'],
                subset=True,
            )
        else:
            sc.pp.highly_variable_genes(
                adata_work,
                flavor="seurat_v3",
                n_top_genes=params['n_top_genes'],
                subset=True,
            )
        
        print(f"Selected {adata_work.n_vars} highly variable genes")
        
        # Setup for scVI
        scvi._settings.ScviConfig(dl_num_workers=7)
        if params['batch_key'] != 'none':
            scvi.model.SCVI.setup_anndata(adata_work, batch_key=params['batch_key'])
        else:
            scvi.model.SCVI.setup_anndata(adata_work)
        
        # Train scVI model
        print("Training scVI model for initialization...")
        vae = scvi.model.SCVI(adata_work, n_layers=2, n_latent=params['n_latent'], gene_likelihood="nb")
        vae.train()
        
        # Get embeddings
        scvi_embedding = vae.get_latent_representation()
        return {'embedding': scvi_embedding, 'model': vae, 'adata': adata_work}
    
    def compute_scanvi_embedding(adata_input, scvi_result=None, **params):
        """Compute scANVI embedding, optionally using pre-trained scVI model"""
        if scvi_result is None:
            # Need to compute scVI first
            scvi_result = compute_scvi_embedding(adata_input, **params)
        
        adata_processed = scvi_result['adata']
        vae = scvi_result['model']
        
        # Make sure the label column is categorical
        if params['label_column'] in adata_processed.obs:
            adata_processed.obs[params['label_column']] = adata_processed.obs[params['label_column']].astype('category')
            print(f"Found {len(adata_processed.obs[params['label_column']].cat.categories)} unique labels in {params['label_column']}")
        else:
            raise ValueError(f"Label column '{params['label_column']}' not found in adata.obs")
        
        # Setup scANVI
        print("Setting up and training scANVI model...")
        if params['batch_key'] != 'none':
            scvi.model.SCANVI.setup_anndata(
                adata_processed, 
                batch_key=params['batch_key'],
                labels_key=params['label_column'],
                unlabeled_category="Unknown"
            )
        else:
            scvi.model.SCANVI.setup_anndata(
                adata_processed, 
                labels_key=params['label_column'],
                unlabeled_category="Unknown"
            )
        
        # Initialize scANVI from the trained scVI model
        model = scvi.model.SCANVI.from_scvi_model(
            vae, 
            adata=adata_processed, 
            labels_key=params['label_column'], 
            unlabeled_category="Unknown"
        )
        
        # Train scANVI model
        model.train(max_epochs=200, early_stopping=True, early_stopping_patience=10)
        
        # Get embeddings
        scanvi_embedding = model.get_latent_representation()
        return {'embedding': scanvi_embedding, 'model': model, 'adata': adata_processed, 'scvi_embedding': scvi_result['embedding']}
    
    # Try to load existing embeddings or compute new ones
    scvi_result = compute_or_load_embedding(
        adata=adata,
        dataset_name=dataset_name,
        embedding_type="scVI",
        embedding_function=compute_scvi_embedding,
        embedding_dir=embedding_dir,
        force_recompute=force_recompute,
        **embedding_params
    )
    
    scanvi_result = compute_or_load_embedding(
        adata=adata,
        dataset_name=dataset_name,
        embedding_type="scANVI",
        embedding_function=lambda adata_input, **params: compute_scanvi_embedding(adata_input, scvi_result, **params),
        embedding_dir=embedding_dir,
        force_recompute=force_recompute,
        **embedding_params
    )
    
    # Extract results and preserve uns data
    if isinstance(scanvi_result, dict):
        adata_copy = scanvi_result['adata']
        adata_copy.obsm["X_scANVI"] = scanvi_result['embedding']
        if 'scvi_embedding' in scanvi_result:
            adata_copy.obsm["X_scVI"] = scanvi_result['scvi_embedding']
    else:
        # If loaded from cache, scanvi_result is just the embedding
        # Create a copy from the original data to preserve structure
        adata_copy = adata.copy()
        adata_copy.obsm["X_scANVI"] = scanvi_result
        
        # Also add scVI embedding if available
        if isinstance(scvi_result, dict):
            adata_copy.obsm["X_scVI"] = scvi_result['embedding']
        else:
            adata_copy.obsm["X_scVI"] = scvi_result
    
    # Restore original uns data
    adata_copy.uns = original_uns.copy()
    
    # Compute neighborhood graph and UMAP for visualization
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata_copy, use_rep="X_scANVI")
    sc.tl.umap(adata_copy)
    
    # Generate visualization plots
    if batch_key is not None:
        sc.pl.umap(adata_copy, color=batch_key, title="scANVI integration - Batch")
    sc.pl.umap(adata_copy, color=label_column, title=f"scANVI integration - {label_column}")
    
    # Additional batch-specific visualization if requested
    if plot_batch_labels and batch_key is not None:
        create_batch_specific_plots(adata_copy, batch_key, label_column, batch_values, figsize)
    
    # Save the processed dataset
    if save_output and file_path_str:
        filename = os.path.basename(file_path_str).rsplit('.h5ad', 1)[0]
        output_file = os.path.join(os.path.dirname(file_path_str), f"{filename}_scanvi.h5ad")
        print(f"Saving processed file to: {output_file}")
        adata_copy.write(output_file)
    
    return adata_copy



def preprocess_single_adata_for_scanvi_split_visualization_without_plots_with_fingerprint(
    file_path, label_column="labels", batch_key=None, 
    n_top_genes=3000, n_latent=30, save_output=False,
    plot_batch_labels=True, batch_values=None, figsize=(15, 6),
    embedding_dir="embeddings", force_recompute=False, 
    fingerprint_column='source'  # New parameter
):
    """
    Loads a single AnnData file, preprocesses it for scANVI, computes latent embeddings, 
    and returns the processed AnnData object with X_scANVI embeddings.
    Now supports caching of embeddings with dataset fingerprinting to handle dataset changes.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file
    label_column : str, default "labels"
        Column name containing cell type labels
    batch_key : str, optional
        Column name for batch information. If None, no batch correction is applied
    n_top_genes : int, default 3000
        Number of highly variable genes to select
    n_latent : int, default 30
        Number of latent dimensions
    save_output : bool, default False
        Whether to save the processed file
    plot_batch_labels : bool, default True
        Whether to create batch-specific label visualization
    batch_values : list or None
        List of two batch values [batch1, batch2] for batch-specific plotting. If None, uses first two unique values
    figsize : tuple, default (15, 6)
        Figure size for batch-specific plots (width, height)
    embedding_dir : str, default "embeddings"
        Directory to save/load embeddings
    force_recompute : bool, default False
        If True, always recompute embeddings even if cached versions exist
    fingerprint_column : str, default 'source'
        Column name to use for dataset fingerprinting to detect dataset changes
        
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with X_scANVI embeddings and preserved uns data
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.cm as cm
    
    # Load dataset
    if isinstance(file_path, str):
        adata = anndata.read_h5ad(file_path)
        dataset_name = os.path.basename(file_path).rsplit('.h5ad', 1)[0]
        file_path_str = file_path
    else:
        adata = file_path
        dataset_name = "anndata_object"
        file_path_str = None
    
    print(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Check dataset fingerprint status
    cache_status = check_embedding_cache_status(
        adata, dataset_name, "scANVI", embedding_dir, 
        fingerprint_column, n_top_genes=n_top_genes, 
        n_latent=n_latent, batch_key=batch_key if batch_key else 'none',
        label_column=label_column
    )
    print(f"Cache status: {'Found' if cache_status['cache_exists'] else 'Not found'}")
    if fingerprint_column in adata.obs.columns:
        print(f"Dataset fingerprint based on {fingerprint_column}: {cache_status['dataset_fingerprint']}")
        print(f"Unique {fingerprint_column} values: {cache_status['unique_values']}")
    
    # Store original uns data to preserve it
    original_uns = adata.uns.copy() if adata.uns else {}
    
    # Create parameter dictionary for embedding cache
    embedding_params = {
        'n_top_genes': n_top_genes,
        'n_latent': n_latent,
        'batch_key': batch_key if batch_key else 'none',
        'label_column': label_column
    }
    
    # Define embedding computation functions (same as before)
    def compute_scvi_embedding(adata_input, **params):
        """Compute scVI embedding"""
        # Create a working copy to avoid modifying the original
        adata_work = adata_input.copy()
        
        # Use raw counts
        adata_work.X = adata_work.X.astype(int)
        adata_work.raw = adata_work.copy()
        
        # Select highly variable genes
        if params['batch_key'] != 'none':
            sc.pp.highly_variable_genes(
                adata_work,
                flavor="seurat_v3",
                n_top_genes=params['n_top_genes'],
                batch_key=params['batch_key'],
                subset=True,
            )
        else:
            sc.pp.highly_variable_genes(
                adata_work,
                flavor="seurat_v3",
                n_top_genes=params['n_top_genes'],
                subset=True,
            )
        
        print(f"Selected {adata_work.n_vars} highly variable genes")
        
        # Setup for scVI
        scvi._settings.ScviConfig(dl_num_workers=7)
        if params['batch_key'] != 'none':
            scvi.model.SCVI.setup_anndata(adata_work, batch_key=params['batch_key'])
        else:
            scvi.model.SCVI.setup_anndata(adata_work)
        
        # Train scVI model
        print("Training scVI model for initialization...")
        vae = scvi.model.SCVI(adata_work, n_layers=2, n_latent=params['n_latent'], gene_likelihood="nb")
        vae.train()
        
        # Get embeddings
        scvi_embedding = vae.get_latent_representation()
        return {'embedding': scvi_embedding, 'model': vae, 'adata': adata_work}
    
    def compute_scanvi_embedding(adata_input, scvi_result=None, **params):
        """Compute scANVI embedding, optionally using pre-trained scVI model"""
        if scvi_result is None:
            # Need to compute scVI first
            scvi_result = compute_scvi_embedding(adata_input, **params)
        
        adata_processed = scvi_result['adata']
        vae = scvi_result['model']
        
        # Make sure the label column is categorical
        if params['label_column'] in adata_processed.obs:
            adata_processed.obs[params['label_column']] = adata_processed.obs[params['label_column']].astype('category')
            print(f"Found {len(adata_processed.obs[params['label_column']].cat.categories)} unique labels in {params['label_column']}")
        else:
            raise ValueError(f"Label column '{params['label_column']}' not found in adata.obs")
        
        # Setup scANVI
        print("Setting up and training scANVI model...")
        if params['batch_key'] != 'none':
            scvi.model.SCANVI.setup_anndata(
                adata_processed, 
                batch_key=params['batch_key'],
                labels_key=params['label_column'],
                unlabeled_category="Unknown"
            )
        else:
            scvi.model.SCANVI.setup_anndata(
                adata_processed, 
                labels_key=params['label_column'],
                unlabeled_category="Unknown"
            )
        
        # Initialize scANVI from the trained scVI model
        model = scvi.model.SCANVI.from_scvi_model(
            vae, 
            adata=adata_processed, 
            labels_key=params['label_column'], 
            unlabeled_category="Unknown"
        )
        
        # Train scANVI model
        model.train(max_epochs=200, early_stopping=True, early_stopping_patience=10)
        
        # Get embeddings
        scanvi_embedding = model.get_latent_representation()
        return {'embedding': scanvi_embedding, 'model': model, 'adata': adata_processed, 'scvi_embedding': scvi_result['embedding']}
    
    # Try to load existing embeddings or compute new ones (WITH FINGERPRINTING)
    scvi_result = compute_or_load_embedding(
        adata=adata,
        dataset_name=dataset_name,
        embedding_type="scVI",
        embedding_function=compute_scvi_embedding,
        embedding_dir=embedding_dir,
        force_recompute=force_recompute,
        fingerprint_column=fingerprint_column,  # NEW: Add fingerprint column
        **embedding_params
    )
    
    scanvi_result = compute_or_load_embedding(
        adata=adata,
        dataset_name=dataset_name,
        embedding_type="scANVI",
        embedding_function=lambda adata_input, **params: compute_scanvi_embedding(adata_input, scvi_result, **params),
        embedding_dir=embedding_dir,
        force_recompute=force_recompute,
        fingerprint_column=fingerprint_column,  # NEW: Add fingerprint column
        **embedding_params
    )
    
    # Extract results and preserve uns data (same as before)
    if isinstance(scanvi_result, dict):
        adata_copy = scanvi_result['adata']
        adata_copy.obsm["X_scANVI"] = scanvi_result['embedding']
        if 'scvi_embedding' in scanvi_result:
            adata_copy.obsm["X_scVI"] = scanvi_result['scvi_embedding']
    else:
        # If loaded from cache, scanvi_result is just the embedding
        # Create a copy from the original data to preserve structure
        adata_copy = adata.copy()
        adata_copy.obsm["X_scANVI"] = scanvi_result
        
        # Also add scVI embedding if available
        if isinstance(scvi_result, dict):
            adata_copy.obsm["X_scVI"] = scvi_result['embedding']
        else:
            adata_copy.obsm["X_scVI"] = scvi_result
    
    # Restore original uns data
    adata_copy.uns = original_uns.copy()
    
    # Compute neighborhood graph and UMAP for visualization
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata_copy, use_rep="X_scANVI")
    sc.tl.umap(adata_copy)
    
    # Generate visualization plots
    if batch_key is not None:
        sc.pl.umap(adata_copy, color=batch_key, title="scANVI integration - Batch")
    sc.pl.umap(adata_copy, color=label_column, title=f"scANVI integration - {label_column}")
    
    # Additional batch-specific visualization if requested
    if plot_batch_labels and batch_key is not None:
        create_batch_specific_plots(adata_copy, batch_key, label_column, batch_values, figsize)
    
    # Save the processed dataset
    if save_output and file_path_str:
        filename = os.path.basename(file_path_str).rsplit('.h5ad', 1)[0]
        output_file = os.path.join(os.path.dirname(file_path_str), f"{filename}_scanvi.h5ad")
        print(f"Saving processed file to: {output_file}")
        adata_copy.write(output_file)
    
    return adata_copy

def create_batch_specific_plots(adata, batch_key, label_column, batch_values=None, figsize=(15, 6)):
    """Create batch-specific visualization plots"""
    import matplotlib.pyplot as plt
    
    # Get unique batch values
    unique_batches = adata.obs[batch_key].unique()
    
    if batch_values is None:
        batch_values = unique_batches[:2] if len(unique_batches) >= 2 else unique_batches
    
    # Create batch-specific plots
    fig, axes = plt.subplots(1, len(batch_values), figsize=figsize)
    if len(batch_values) == 1:
        axes = [axes]
    
    for i, batch_val in enumerate(batch_values):
        batch_mask = adata.obs[batch_key] == batch_val
        batch_adata = adata[batch_mask]
        
        sc.pl.umap(batch_adata, color=label_column, 
                  title=f"{label_column} - {batch_val}",
                  ax=axes[i], show=False)
    
    plt.tight_layout()
    plt.show()
