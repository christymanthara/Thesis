import anndata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import utils
import os
from sklearn.model_selection import cross_val_score
from knn_plot_table import create_results_table

from compute_tsne_embeddings import compute_tsne_embedding_pavlin
from compute_tsne_embeddings import compute_tsne_embedding

def compute_knn_tsne_all(file_path, reference_file=None, skip_preprocessing=False, n_jobs=1):
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
    
    """
    
    # Dictionary to store results for table generation
    results_table = {}
    
    # Load the main data file
    if skip_preprocessing:
        print(f"Loading {file_path} directly (skipping preprocessing)")
        adata = anndata.read_h5ad(file_path)
    else:
        print(f"Loading and preprocessing {file_path}")
        # Add preprocessing code here if needed
        adata = anndata.read_h5ad(file_path)
    
    # Load reference file if provided, otherwise use the same file
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
    
    # Get base filename for output
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
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
                if 'pca' in embedding_key.lower():
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
                    if 'pca' in embedding_key.lower():
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
            
            reference_mask = ref_tsne.obs["source"] == "baron_2016h"
        
            #Cross validation to get a more robust estimate of accuracy
            print(f"Cross-validating KNN on reference embeddings for {embedding_key}...")
            # Extract reference data for cross-validation
            # reference_embeddings = ref_tsne.obsm[embedding_key][reference_mask]
            reference_embeddings = ref_tsne.obsm[processing_key][reference_mask]
            reference_labels = ref_tsne.obs["labels"][reference_mask].values.astype(str)

            # Cross-validation on reference data
            print(f"Running cross-validation on reference data for {embedding_key}...")
            cv_scores_orig = cross_val_score(knn_orig, reference_embeddings, reference_labels, cv=5)
            reference_cv_accuracy = cv_scores_orig.mean()
            reference_cv_std = cv_scores_orig.std()
            print(f"Reference CV accuracy using {embedding_key}: {reference_cv_accuracy:.4f} ± {reference_cv_std:.4f}")

            # Train final model on full reference data
            knn_orig.fit(reference_embeddings, reference_labels)

            # Evaluate only on query data
            query_mask = adata_tsne.obs["source"] == "xin_2016"
            # query_embeddings = adata_tsne.obsm[embedding_key][query_mask]
            query_embeddings = adata_tsne.obsm[processing_key][query_mask]
            query_labels = adata_tsne.obs["labels"][query_mask].values.astype(str)
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
            
        
        
            embedding_clean = embedding_key.replace('X_', '')

            # Special display name for original_X
            if embedding_key == 'original_X':
                display_name = 'RAW data (original_X)'
            else:
                display_name = embedding_clean

            results_table[f"{display_name}"] = {
                'Reference CV': f"{reference_cv_accuracy:.3f}±{reference_cv_std:.3f}",
                'Query Transfer': f"{query_orig_accuracy:.3f}"
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
            
            
            reference_mask = ref_tsne.obs["source"] == "baron_2016h"
            
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
            
            
            query_mask = adata_tsne.obs["source"] == "xin_2016"
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
            # tsne_min = min(ref_tsne.obsm[tsne_key].min(), adata_tsne.obsm[tsne_key].min())
            # tsne_max = max(ref_tsne.obsm[tsne_key].max(), adata_tsne.obsm[tsne_key].max())
            # coord_range = tsne_min - 1, tsne_max + 1
            all_coords = np.vstack([ref_coords, query_coords])
            coord_min = all_coords.min()
            coord_max = all_coords.max()
            coord_range = coord_min - 1, coord_max + 1
            
            for ax_ in ax.ravel():
                ax_.set_xlim(*coord_range), ax_.set_ylim(*coord_range)
            
            # Add subplot labels
            for ax_, letter in zip(ax, string.ascii_lowercase): 
                plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=15, fontweight="bold")
            
            # Add KNN accuracy text to the figure
            embedding_clean = embedding_key.replace('X_', '').replace('_', '')
            viz_label = "UMAP" if 'umap' in embedding_key.lower() else "t-SNE"
            # fig.text(0.5, 0.10, 
            #         f"Reference CV - {embedding_clean}: {reference_cv_accuracy:.3f}±{reference_cv_std:.3f}  |  "
            #         # f"{viz_label}: {reference_tsne_cv_accuracy:.3f}±{reference_tsne_cv_std:.3f}\n"
            #         f"Query Transfer - {embedding_clean}: {query_orig_accuracy:.3f}  |  "
            #         # f"{viz_label}: {query_tsne_accuracy:.3f}", 
            #         ha='center', fontsize=10)
            
            fig.text(0.5, 0.10, 
                    f"Reference CV - {embedding_clean}: {reference_cv_accuracy:.3f}±{reference_cv_std:.3f}  |  "
                    f"Query Transfer - {embedding_clean}: {query_orig_accuracy:.3f}", 
                    ha='center', fontsize=10)
            
            
            # Generate output filename
            embedding_name = embedding_key.replace('X_', '').lower()
            output_pdf = f"tsne_plot_cross_validated_final_{base_filename}_{embedding_name}.pdf"
            
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
    if results_table:
        create_results_table(results_table, base_filename, reference_file)
    
    
    print(f"\nCompleted processing all embeddings for {file_path}")
    


# Example usage
if __name__ == "__main__":
    # Process single file (uses same file as reference)
    compute_knn_tsne_all("/shared/home/christy.jo.manthara/batch-effect-analysis/output/baron_2016h_xin_2016_preprocessed_with_original_X_uce_adata_X_scvi_X_scanvi_X_uce_test.h5ad", skip_preprocessing=True, n_jobs=1)
    
    # Process with separate reference file
    # compute_knn_tsne_all("xin_2016_scGPT.h5ad", reference_file="baron_2016h_scGPT.h5ad", skip_preprocessing=True)