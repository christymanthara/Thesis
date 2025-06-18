import anndata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import scripts.utils
import os
from sklearn.model_selection import cross_val_score
from .knn_plot_table import create_results_table
from . import utils

from .compute_tsne_embeddings import compute_tsne_embedding_pavlin
from .compute_tsne_embeddings import compute_tsne_embedding
from .data_utils.silhouette_score import compute_silhouette_scores

def compute_knn_tsne_all(file_path, reference_file=None, skip_preprocessing=False, n_jobs=1,split_by_source=True):
    """
    Compute KNN and t-SNE for all embeddings in an AnnData file and save each plot as a separate PDF.
    
    Parameters:
    -----------
    file_path : str
        Path to the AnnData file to process
    reference_file : str, optional
        Path to reference AnnData file for KNN training. If None, uses the same file for both training and testing
    skip_preprocessing : bool, default False
        Whether to skip preprocessing and load data directly
    n_jobs : int, default 1
        Number of threads to use for t-SNE computation    
    split_by_source : bool, default False
        Whether to split the data based on obs.source values              
    """   
    
    # Dictionary to store results for table generation
     # Dictionary to store results for table generation     
    results_table = {}          
    
    # Handle input type - could be file path string or AnnData object
    if isinstance(file_path, str):
        file_name = file_path  # Store original path for logging
        # Load the main data file
        if skip_preprocessing:
            print(f"Loading {file_name} directly (skipping preprocessing)")
            adata = anndata.read_h5ad(file_path)
        else:
            print(f"Loading and preprocessing {file_name}")
            adata = anndata.read_h5ad(file_path)
            # Add preprocessing code here if needed
    else:
        # Input is already an AnnData object
        adata = file_path
        if not skip_preprocessing:
            print("Preprocessing existing AnnData object")
            # Add preprocessing code here if needed
        else:
            print("Using existing AnnData object directly (skipping preprocessing)")      
        
    # Initialize metadata variables
    main_source = None
    ref_source = None
    main_tissue = "unknown"
    ref_tissue = "unknown"
    main_organism = "unknown"
    ref_organism = "unknown"    
    
    # Handle data splitting based on source
    if split_by_source and 'source' in adata.obs.columns:
        # Get unique source values
        source_values = adata.obs['source'].unique()
        print(f"Found source values: {source_values}")
        
        if len(source_values) == 2:
            # Split into two datasets based on source
            source1, source2 = source_values
            print(f"Splitting data into {source1} and {source2}")
            # Create separate AnnData objects for each source   
            print(f"Using {source1} as main data and {source2} as reference")
            adata_source1 = adata[adata.obs['source'] == source1].copy()
            adata_source2 = adata[adata.obs['source'] == source2].copy()
            
            print(f"Split data: {source1} ({adata_source1.n_obs} cells), {source2} ({adata_source2.n_obs} cells)")
            
            # Store the source names for later use
            main_source = source1
            ref_source = source2
            
            # Extract tissue and organism information
            # For main source (adata_source1)
            if 'tissue' in adata_source1.uns.keys():
                main_tissue = adata_source1.uns['tissue']
                print(f"Found tissue for {main_source}: {main_tissue}")
            else:
                print(f"No tissue key found in uns for {main_source}, using 'unknown'")
                
            if 'organism' in adata_source1.uns.keys():
                main_organism = adata_source1.uns['organism']
                print(f"Found organism for {main_source}: {main_organism}")
            else:
                print(f"No organism key found in uns for {main_source}, using 'unknown'")
            
            # For reference source (adata_source2)
            if 'tissue' in adata_source2.uns.keys():
                ref_tissue = adata_source2.uns['tissue']
                print(f"Found tissue for {ref_source}: {ref_tissue}")
            else:
                print(f"No tissue key found in uns for {ref_source}, using 'unknown'")
                
            if 'organism' in adata_source2.uns.keys():
                ref_organism = adata_source2.uns['organism']
                print(f"Found organism for {ref_source}: {ref_organism}")
            else:
                print(f"No organism key found in uns for {ref_source}, using 'unknown'")
            
            
            # Assign one as main adata and other as reference
            adata = adata_source1
            ref_adata = adata_source2
            
        else:
            print(f"Warning: Expected 2 source values, found {len(source_values)}. Using original data.")
            # Fall back to original logic
            if reference_file is not None:
                if skip_preprocessing:
                    print(f"Loading reference file {reference_file} directly")
                    ref_adata = anndata.read_h5ad(reference_file)
                else:
                    print(f"Loading and preprocessing reference file {reference_file}")
                    ref_adata = anndata.read_h5ad(reference_file)
            else:
                print("Using the same file as reference")
                ref_adata = adata.copy()
    else:
        # Original logic when not splitting by source
        if reference_file is not None:
            if skip_preprocessing:
                print(f"Loading reference file {reference_file} directly")
                ref_adata = anndata.read_h5ad(reference_file)
            else:
                print(f"Loading and preprocessing reference file {reference_file}")
                ref_adata = anndata.read_h5ad(reference_file)
            
            # Extract metadata from separate files    
            if 'tissue' in adata.uns.keys():
                main_tissue = adata.uns['tissue']
                print(f"Found tissue for main data: {main_tissue}")
            else:
                print(f"No tissue key found in uns for main data, using 'unknown'")
                
            if 'organism' in adata.uns.keys():
                main_organism = adata.uns['organism']
                print(f"Found organism for main data: {main_organism}")
            else:
                print(f"No organism key found in uns for main data, using 'unknown'")
                
            if 'tissue' in ref_adata.uns.keys():
                ref_tissue = ref_adata.uns['tissue']
                print(f"Found tissue for reference data: {ref_tissue}")
            else:
                print(f"No tissue key found in uns for reference data, using 'unknown'")
                
            if 'organism' in ref_adata.uns.keys():
                ref_organism = ref_adata.uns['organism']
                print(f"Found organism for reference data: {ref_organism}")
            else:
                print(f"No organism key found in uns for reference data, using 'unknown'")
            
        else:
            print("Using the same file as input for the  reference")
            ref_adata = adata.copy()
            # Use same metadata for both since it's the same file
            if 'tissue' in adata.uns.keys():
                main_tissue = ref_tissue = adata.uns['tissue']
                print(f"Found tissue: {main_tissue}")
            else:
                print(f"No tissue key found in uns, using 'unknown'")
                
            if 'organism' in adata.uns.keys():
                main_organism = ref_organism = adata.uns['organism']
                print(f"Found organism: {main_organism}")
            else:
                print(f"No organism key found in uns, using 'unknown'")
    
    # Get base filename for output
    if isinstance(file_path, str):
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
    else:
        base_filename_ref = ref_adata.obs["batch"].unique()[0] if 'batch' in ref_adata.obs.columns else "merged_data1"
        adata_name = adata.obs["batch"].unique()[0] if 'batch' in adata.obs.columns else "merged_data2"
        base_filename = f"{adata_name}_{base_filename_ref}"
    print(f"Base filename for output: {base_filename}")
    # Get all embedding keys from obsm
    embedding_keys = [key for key in adata.obsm.keys() if key.startswith('X_')]

    # Check if original_X exists in layers and add it to processing list
    if 'original_X' in adata.layers.keys():
        embedding_keys.append('original_X')  # Add as special case
        print("Found original_X in layers, adding to processing list")

    if not embedding_keys:
        print("No embeddings found in obsm (looking for keys starting with 'X_')")
        return

    print(f"Found embeddings: {embedding_keys}")

    # Process each embedding
    for embedding_key in embedding_keys:
        print(f"\nProcessing embedding: {embedding_key}")
        
        # Handle original_X differently
        if embedding_key == 'original_X':
            # Check if original_X exists in both files' layers
            if 'original_X' not in adata.layers.keys():
                print(f"original_X not found in main file layers, skipping...")
                continue
            if 'original_X' not in ref_adata.layers.keys():
                print(f"original_X not found in reference file layers, skipping...")
                continue
            
            print("Extracting original_X data from layers...")
            try:
                # Temporarily add original_X data to obsm for processing
                adata.obsm['X_original_temp'] = adata.layers['original_X']
                ref_adata.obsm['X_original_temp'] = ref_adata.layers['original_X']
                
                # Set the key we'll use for processing
                processing_key = 'X_original_temp'
                
            except Exception as e:
                print(f"Error extracting original_X data: {str(e)}")
                continue
        else:
            # For regular embeddings in obsm
            # Check if embedding exists in both files
            if embedding_key not in adata.obsm:
                print(f"Embedding {embedding_key} not found in main file, skipping...")
                continue
            if embedding_key not in ref_adata.obsm:
                print(f"Embedding {embedding_key} not found in reference file, skipping...")
                continue
            
            processing_key = embedding_key
        
        # Create copies for processing
        adata_copy = adata.copy()
        ref_copy = ref_adata.copy()
        
        # Compute t-SNE embeddings
        tsne_key = f"{embedding_key.lower()}_tsne"
        print(f"Computing t-SNE embeddings from {embedding_key}...")
        
        # Skip t-SNE computation for UMAP since it's already a 2D visualization
        if 'umap' in embedding_key.lower():
            print(f"Skipping t-SNE computation for {embedding_key} (already 2D visualization)")
            # Use UMAP directly as the "t-SNE" visualization
            adata_tsne = adata_copy.copy()
            ref_tsne = ref_copy.copy()
            tsne_key = processing_key  
        else:
            try:
                # Check if this is a PCA embedding to use Pavlin's method
                if 'X_pca' in embedding_key.lower():
                    print(f"Using Pavlin's method for PCA embedding: {embedding_key}")
                    # Use Pavlin's method for PCA embeddings
                    adata_tsne = compute_tsne_embedding_pavlin(adata_copy, embedding_key=processing_key, 
                                                             output_key=tsne_key, n_jobs=n_jobs)
                    ref_tsne = compute_tsne_embedding_pavlin(ref_copy, embedding_key=processing_key, 
                                                           output_key=tsne_key, n_jobs=n_jobs)
                else:
                    # Use default method for all other embeddings
                    adata_tsne = compute_tsne_embedding(adata_copy, embedding_key=processing_key, 
                                                      output_key=tsne_key, n_jobs=n_jobs)
                    ref_tsne = compute_tsne_embedding(ref_copy, embedding_key=processing_key, 
                                                    output_key=tsne_key, n_jobs=n_jobs)
            except Exception as e:
                print(f"Error computing t-SNE for {embedding_key}: {str(e)}")
                print("Trying with n_jobs=1...")
                # Clean up temporary keys if they exist
                try:
                    # Fallback to single thread
                    if 'X_pca' in embedding_key.lower():
                        print(f"Retrying with Pavlin's method and n_jobs=1 for: {embedding_key}")
                        adata_tsne = compute_tsne_embedding_pavlin(adata_copy, embedding_key=processing_key, 
                                                                 output_key=tsne_key, n_jobs=1)
                        ref_tsne = compute_tsne_embedding_pavlin(ref_copy, embedding_key=processing_key, 
                                                               output_key=tsne_key, n_jobs=1)
                    else:
                        adata_tsne = compute_tsne_embedding(adata_copy, embedding_key=processing_key, 
                                                          output_key=tsne_key, n_jobs=1)
                        ref_tsne = compute_tsne_embedding(ref_copy, embedding_key=processing_key, 
                                                        output_key=tsne_key, n_jobs=1)
                except Exception as e2:
                    print(f"Error computing t-SNE for {embedding_key} even with n_jobs=1: {str(e2)}")
                    # Clean up temporary keys if they exist
                    if embedding_key == 'original_X':
                        if 'X_original_temp' in adata.obsm:
                            del adata.obsm['X_original_temp']
                        if 'X_original_temp' in ref_adata.obsm:
                            del ref_adata.obsm['X_original_temp']
                    continue
        
        try:
            # Run KNN classification using original embeddings
            print(f"Running KNN classification using {embedding_key} embeddings...")
            
            # Determine metric based on embedding type
            if 'scGPT' in embedding_key or 'scgpt' in embedding_key:
                knn_orig = KNeighborsClassifier(n_neighbors=10, metric='cosine')
                
            elif 'tsne_pavlin' in embedding_key:
                knn_orig = KNeighborsClassifier(n_neighbors=10, metric='cosine') #as per literature, pavlins method works well with cosine similarity
            elif 'scVI' in embedding_key or 'scANVI' in embedding_key:
                # scVI/scANVI embeddings work well with cosine similarity
                knn_orig = KNeighborsClassifier(n_neighbors=10, metric='cosine')
            else:
                knn_orig = KNeighborsClassifier(n_neighbors=10)
            
            # KNN training should filter to reference only
            
            if split_by_source and 'source' in ref_tsne.obs.columns:
                reference_mask = ref_tsne.obs["source"] == ref_source  # Use stored ref_source
            else:
                reference_mask = ref_tsne.obs["source"] == "baron_2016h"  # Fallback
            
            # Fix the query mask similarly
            if split_by_source and 'source' in adata_tsne.obs.columns:
                query_mask = adata_tsne.obs["source"] == main_source  # Use stored main_source
            else:
                query_mask = adata_tsne.obs["source"] == "xin_2016"  # Fallback
                
            # print query and reference masks for debugging
            print(f"Reference source: {ref_source}, Query source: {main_source}")
            
            # Debug prints to verify masks
            print(f"Reference mask sum: {reference_mask.sum()}")
            print(f"Query mask sum: {query_mask.sum()}")
            
            if reference_mask.sum() == 0:
                print(f"Warning: No reference samples found! Reference source: {ref_source}")
                print(f"Available sources in ref_tsne: {ref_tsne.obs['source'].unique()}")
                continue
                
            if query_mask.sum() == 0:
                print(f"Warning: No query samples found! Query source: {main_source}")
                print(f"Available sources in adata_tsne: {adata_tsne.obs['source'].unique()}")
                continue
        
            #Cross validation to get a more robust estimate of accuracy
            print(f"Cross-validating KNN on reference embeddings for {embedding_key}...")
            # Extract reference data for cross-validation
            reference_embeddings = ref_tsne.obsm[processing_key][reference_mask]
            reference_labels = ref_tsne.obs["labels"][reference_mask].values.astype(str)
            print(f"Refference labels unique values: {np.unique(reference_labels)}")

            # Cross-validation on reference data
            print(f"Running cross-validation on reference data for {embedding_key}...")
            cv_scores_orig = cross_val_score(knn_orig, reference_embeddings, reference_labels, cv=5)
            reference_cv_accuracy = cv_scores_orig.mean()
            reference_cv_std = cv_scores_orig.std()
            print(f"Reference CV accuracy using {embedding_key}: {reference_cv_accuracy:.4f} ± {reference_cv_std:.4f}")

            # Train final model on full reference data
            knn_orig.fit(reference_embeddings, reference_labels)

            # Evaluate only on query data
            query_embeddings = adata_tsne.obsm[processing_key][query_mask]
            query_labels = adata_tsne.obs["labels"][query_mask].values.astype(str)
            print(f"Query labels unique values: {np.unique(query_labels)}")
            query_orig_accuracy = accuracy_score(knn_orig.predict(query_embeddings), query_labels)
            print(f"Query accuracy using {embedding_key}: {query_orig_accuracy:.4f}")
            
            #For cross-validation on t-SNE embeddings
            print(f"Cross-validating KNN on t-SNE embeddings for {tsne_key}...")
            print(f"Running KNN classification using {tsne_key} embeddings...")
            knn_tsne = KNeighborsClassifier(n_neighbors=10)

            # Cross-validation on reference t-SNE data
            reference_tsne_embeddings = ref_tsne.obsm[tsne_key][reference_mask]
            cv_scores_tsne = cross_val_score(knn_tsne, reference_tsne_embeddings, reference_labels, cv=5)
            reference_tsne_cv_accuracy = cv_scores_tsne.mean()
            reference_tsne_cv_std = cv_scores_tsne.std()
            print(f"Reference CV accuracy using {tsne_key}: {reference_tsne_cv_accuracy:.4f} ± {reference_tsne_cv_std:.4f}")

            # Train final model and evaluate on query
            knn_tsne.fit(reference_tsne_embeddings, reference_labels)
            query_tsne_embeddings = adata_tsne.obsm[tsne_key][query_mask]
            query_tsne_accuracy = accuracy_score(knn_tsne.predict(query_tsne_embeddings), query_labels)
            print(f"Query accuracy using {tsne_key}: {query_tsne_accuracy:.4f}")
            
            #---------------------------------------------------------------------------------
            # Compute silhouette scores
            silhouette_results = compute_silhouette_scores(
                reference_embeddings=reference_embeddings,
                reference_labels=reference_labels,
                query_embeddings=query_embeddings, 
                query_labels=query_labels,
                reference_tsne_embeddings=reference_tsne_embeddings,
                query_tsne_embeddings=query_tsne_embeddings,
                embedding_key=embedding_key,
                tsne_key=tsne_key
            )
            
            # Extract silhouette scores from results
            reference_sil_score = silhouette_results['reference_sil_score']
            query_sil_score = silhouette_results['query_sil_score']
            reference_tsne_sil_score = silhouette_results['reference_tsne_sil_score']
            query_tsne_sil_score = silhouette_results['query_tsne_sil_score']
            
            #----------------------------------------------------------------------------------
            # Compute batch effect reduction metrics
            print(f"Computing batch integration metrics for {embedding_key}...")
            
            # Prepare combined data for batch effect analysis
            combined_embeddings = np.vstack([reference_embeddings, query_embeddings])
            combined_labels = np.concatenate([reference_labels, query_labels])
            combined_batches = np.concatenate([
                np.full(len(reference_labels), ref_source),
                np.full(len(query_labels), main_source)
            ])
            
            # Compute batch integration metrics
            batch_metrics = compute_batch_effect_metrics(
                embeddings=combined_embeddings,
                cell_types=combined_labels, 
                batch_labels=combined_batches,
                embedding_name=embedding_key
            )
            
            # Also compute for t-SNE embeddings
            combined_tsne_embeddings = np.vstack([reference_tsne_embeddings, query_tsne_embeddings])
            
            batch_metrics_tsne = compute_batch_effect_metrics(
                embeddings=combined_tsne_embeddings,
                cell_types=combined_labels,
                batch_labels=combined_batches, 
                embedding_name=tsne_key
            )
            
            # Extract key metrics
            integration_score = batch_metrics['integration_score']
            batch_silhouette = batch_metrics['batch_silhouette']
            celltype_silhouette = batch_metrics['celltype_silhouette']
            mixing_entropy = batch_metrics['batch_mixing_entropy']
            
            integration_score_tsne = batch_metrics_tsne['integration_score']
            batch_silhouette_tsne = batch_metrics_tsne['batch_silhouette']
            
            #---------------------------------------------------------------------------------------
            
            embedding_clean = embedding_key.replace('X_', '')

            # Special display name for original_X
            if embedding_key == 'original_X':
                display_name = 'RAW data (original_X)'
            else:
                display_name = embedding_clean

            # results_table[f"{display_name}"] = {
            #     'Reference CV': f"{reference_cv_accuracy:.3f}±{reference_cv_std:.3f}",
            #     'Query Transfer': f"{query_orig_accuracy:.3f}",
            #     'Ref Silhouette': f"{reference_sil_score:.3f}",
            #     'Query Silhouette': f"{query_sil_score:.3f}",
            #     'Ref t-SNE Sil': f"{reference_tsne_sil_score:.3f}",
            #     'Query t-SNE Sil': f"{query_tsne_sil_score:.3f}"
            # }
            
            results_table[f"{display_name}"] = {
                'Reference CV': f"{reference_cv_accuracy:.3f}±{reference_cv_std:.3f}",
                'Query Transfer': f"{query_orig_accuracy:.3f}",
                'Integration Score': f"{integration_score:.3f}",
                'Batch Mixing': f"{mixing_entropy:.3f}",
                'Batch Silhouette': f"{batch_silhouette:.3f}",
                'CellType Silhouette': f"{celltype_silhouette:.3f}",
                't-SNE Integration': f"{integration_score_tsne:.3f}"
            }
            
            # Get colors and cell order
            # Get colors from reference data only
            # adata_ref = ref_tsne[ref_tsne.obs["source"] == "baron_2016h"].copy()
            adata_ref = ref_tsne[reference_mask].copy()
            colors = utils.get_colors_for(adata_ref)
            cell_order = list(colors.keys())
            num_cell_types = len(np.unique(ref_tsne.obs["labels"]))
            
            # Create plot
            fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
            
            # Add overall title
            ref_name = os.path.splitext(os.path.basename(reference_file))[0] if reference_file else base_filename
            fig.suptitle(f"{base_filename}_{ref_name}_{embedding_key}", fontsize=14, y=0.95)
            
            # Plot reference embedding (t-SNE or UMAP)
            viz_method = "UMAP" if 'umap' in embedding_key.lower() else "t-SNE"
            
            
            reference_mask = ref_tsne.obs["source"] == ref_source
            
            # Plot reference points in colors on the left
            # Plot reference points in colors on the left
            ref_coords = ref_tsne.obsm[tsne_key][reference_mask]
            ref_labels = ref_tsne.obs["labels"][reference_mask]
            
            utils.plot(ref_coords, ref_labels, ax=ax[0], 
                    title=f"Reference embedding ({viz_method} from {embedding_key})", 
                    colors=colors, s=3, label_order=cell_order,
                    legend_kwargs=dict(loc="upper center", bbox_to_anchor=(0.5, 0.05), 
                                        bbox_transform=fig.transFigure, labelspacing=1, 
                                        ncol=num_cell_types // 2 + 1))
        
            
            # Plot transformed samples on the right
            # Create a mask for the reference points to plot them in black and white
            colors_bw = {1: "#666666"}
            
            #First plot the reference points in black and white
            utils.plot(ref_coords, np.ones(len(ref_coords)), ax=ax[1], 
                    colors=colors_bw, alpha=0.05, s=3, draw_legend=False)
            
            
            query_mask = adata_tsne.obs["source"] == main_source
            # Then plot the transformed samples with colors
            query_coords = adata_tsne.obsm[tsne_key][query_mask]
            query_labels = adata_tsne.obs["labels"][query_mask]
            
            utils.plot(query_coords, query_labels, ax=ax[1], colors=colors, 
                    draw_legend=False, s=6, label_order=cell_order, alpha=0.7)
            # Right plot title  - they are creating visualization issues
            # ax[1].set_title(f"Reference {reference_mask} (gray) +  Query {query_mask} (colored) ({viz_method} from {embedding_key})")
            
            # Set equal axis for all plots
            for ax_ in ax.ravel(): 
                ax_.axis("equal")
            
            # Determine coordinate range from visualization data
            if len(ref_coords) > 0 and len(query_coords) > 0:
                all_coords = np.vstack([ref_coords, query_coords])
                coord_min = all_coords.min()
                coord_max = all_coords.max()
                coord_range = coord_min - 1, coord_max + 1
                
                for ax_ in ax.ravel():
                    ax_.set_xlim(*coord_range), ax_.set_ylim(*coord_range)
                    
            else:
                print(f"Warning: Empty coordinate arrays - ref_coords: {len(ref_coords)}, query_coords: {len(query_coords)}")
                print(f"Skipping coordinate range setting for {embedding_key}")
            
            # Add subplot labels
            for ax_, letter in zip(ax, string.ascii_lowercase): 
                plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=15, fontweight="bold")
            
            # Add KNN accuracy text to the figure
            embedding_clean = embedding_key.replace('X_', '').replace('_', '')
            viz_label = "UMAP" if 'umap' in embedding_key.lower() else "t-SNE"
            
            
            # fig.text(0.5, 0.10, 
            #     f"Reference CV - {embedding_clean}: {reference_cv_accuracy:.3f}±{reference_cv_std:.3f}  |  "
            #     f"Query Transfer - {embedding_clean}: {query_orig_accuracy:.3f}\n"
            #     f"Ref Sil: {reference_sil_score:.3f}  |  Query Sil: {query_sil_score:.3f}  |  "
            #     f"t-SNE Sil: {reference_tsne_sil_score:.3f}/{query_tsne_sil_score:.3f}", 
            #     ha='center', fontsize=9, va='center')
            
            # Update the plot text to include batch integration info
            fig.text(0.5, 0.10, 
                    f"Reference CV: {reference_cv_accuracy:.3f}±{reference_cv_std:.3f}  |  "
                    f"Query Transfer: {query_orig_accuracy:.3f}\n"
                    f"Integration Score: {integration_score:.3f}  |  "
                    f"Batch Sil: {batch_silhouette:.3f} (↓)  |  "
                    f"CellType Sil: {celltype_silhouette:.3f} (↑)", 
                    ha='center', fontsize=9, va='center')
            
            # Generate output filename
            embedding_name = embedding_key.replace('X_', '').lower()
            output_pdf = f"tsne_plot_cross_validated_final_{base_filename}_{embedding_name}.pdf"
            
            if embedding_key == 'X_pca':
                print("Using Pavlin's method for PCA embedding and plotting the results")
                output_pdf = f"tsne_plot_cross_validated_final_{base_filename}_pavlin_plot.pdf"
            
            # Save plot
            plt.savefig(output_pdf, dpi=600, bbox_inches="tight", transparent=True)
            print(f"Saved plot as {output_pdf}")
            plt.close()
            
        except Exception as e:
            print(f"Error processing embedding {embedding_key}: {str(e)}")
            continue
        
        finally:
        # Clean up temporary keys
            if embedding_key == 'original_X':
                if 'X_original_temp' in adata.obsm:
                    del adata.obsm['X_original_temp']
                if 'X_original_temp' in ref_adata.obsm:
                    del ref_adata.obsm['X_original_temp']
        # Generate summary table
     # Generate summary table with metadata
    if results_table:
        # Create metadata dictionary
        metadata = {
            'main_source': main_source,
            'ref_source': ref_source,
            'main_tissue': main_tissue,
            'ref_tissue': ref_tissue,
            'main_organism': main_organism,
            'ref_organism': ref_organism
        }
        create_results_table(results_table, main_source, ref_source, base_filename, reference_file, metadata)
        print("Results table created successfully.")
    print(f"\nCompleted processing all embeddings for {file_path}")
    


# Example usage
if __name__ == "__main__":
    # Process single file (uses same file as reference)
    compute_knn_tsne_all("/shared/home/christy.jo.manthara/batch-effect-analysis/output/baron_2016h_xin_2016_preprocessed_with_original_X_uce_adata_X_scvi_X_scanvi_X_uce_test.h5ad", skip_preprocessing=True, n_jobs=1)
    