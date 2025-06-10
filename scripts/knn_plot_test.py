    import anndata
    import numpy as np
    import matplotlib.pyplot as plt
    import string
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import utils
    import os

    from compute_tsne_embeddings import compute_tsne_embedding

    def compute_knn_tsne_all(file_path, reference_file=None, skip_preprocessing=False):
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
        """
        
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
        
        if not embedding_keys:
            print("No embeddings found in obsm (looking for keys starting with 'X_')")
            return
        
        print(f"Found embeddings: {embedding_keys}")
        
        # Process each embedding
        for embedding_key in embedding_keys:
            print(f"\nProcessing embedding: {embedding_key}")
            
            # Check if embedding exists in both files
            if embedding_key not in adata.obsm:
                print(f"Embedding {embedding_key} not found in main file, skipping...")
                continue
            if embedding_key not in ref_adata.obsm:
                print(f"Embedding {embedding_key} not found in reference file, skipping...")
                continue
            
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
                tsne_key = embedding_key  # Use the original UMAP embedding
            else:
                try:
                    # Compute t-SNE for both datasets
                    adata_tsne = compute_tsne_embedding(adata_copy, embedding_key=embedding_key, output_key=tsne_key)
                    ref_tsne = compute_tsne_embedding(ref_copy, embedding_key=embedding_key, output_key=tsne_key)
                except Exception as e:
                    print(f"Error computing t-SNE for {embedding_key}: {str(e)}")
                    continue
            
            try:
                # Run KNN classification using original embeddings
                print(f"Running KNN classification using {embedding_key} embeddings...")
                
                # Determine metric based on embedding type
                if 'scGPT' in embedding_key or 'scgpt' in embedding_key:
                    knn_orig = KNeighborsClassifier(n_neighbors=10, metric='cosine')
                elif 'scVI' in embedding_key or 'scANVI' in embedding_key:
                    # scVI/scANVI embeddings work well with cosine similarity
                    knn_orig = KNeighborsClassifier(n_neighbors=10, metric='cosine')
                else:
                    knn_orig = KNeighborsClassifier(n_neighbors=10)
                
                knn_orig.fit(ref_tsne.obsm[embedding_key], ref_tsne.obs["labels"].values.astype(str))
                orig_accuracy = accuracy_score(knn_orig.predict(adata_tsne.obsm[embedding_key]), 
                                            adata_tsne.obs["labels"].values.astype(str))
                print(f"KNN accuracy using {embedding_key} embeddings: {orig_accuracy:.4f}")
                
                # Run KNN classification using t-SNE embeddings
                print(f"Running KNN classification using {tsne_key} embeddings...")
                knn_tsne = KNeighborsClassifier(n_neighbors=10)
                knn_tsne.fit(ref_tsne.obsm[tsne_key], ref_tsne.obs["labels"].values.astype(str))
                tsne_accuracy = accuracy_score(knn_tsne.predict(adata_tsne.obsm[tsne_key]), 
                                            adata_tsne.obs["labels"].values.astype(str))
                print(f"KNN accuracy using {tsne_key} embeddings: {tsne_accuracy:.4f}")
                
                # Get colors and cell order
                colors = utils.get_colors_for(ref_tsne)
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
                utils.plot(ref_tsne.obsm[tsne_key][reference_mask], ref_tsne.obs["labels"], ax=ax[0], 
                        title=f"Reference embedding ({viz_method} from {embedding_key})", 
                        colors=colors, s=3, label_order=cell_order,
                        legend_kwargs=dict(loc="upper center", bbox_to_anchor=(0.5, 0.05), 
                                            bbox_transform=fig.transFigure, labelspacing=1, 
                                            ncol=num_cell_types // 2 + 1))
                # Left plot title
                ax[0].set_title(f"Reference embedding - {reference_mask} only ({viz_method} from {embedding_key})")

                
                # Plot transformed samples on the right
                # Create a mask for the reference points to plot them in black and white
                colors_bw = {1: "#666666"}
                
                #First plot the reference points in black and white
                utils.plot(ref_tsne.obsm[tsne_key][reference_mask], np.ones_like(ref_tsne.obs["labels"]), ax=ax[1], 
                        colors=colors_bw, alpha=0.05, s=3, draw_legend=False)
                
                
                query_mask = adata_tsne.obs["source"] == "xin_2016"
                # Then plot the transformed samples with colors
                utils.plot(adata_tsne.obsm[tsne_key][query_mask], adata_tsne.obs["labels"], ax=ax[1], colors=colors, 
                        draw_legend=False, s=6, label_order=cell_order, alpha=0.7)
                # Right plot title  
                ax[1].set_title(f"{reference_mask} (gray) + {query_mask} (colored) ({viz_method} from {embedding_key})")
                
                # Set equal axis for all plots
                for ax_ in ax.ravel(): 
                    ax_.axis("equal")
                
                # Determine coordinate range from visualization data
                tsne_min = min(ref_tsne.obsm[tsne_key].min(), adata_tsne.obsm[tsne_key].min())
                tsne_max = max(ref_tsne.obsm[tsne_key].max(), adata_tsne.obsm[tsne_key].max())
                coord_range = tsne_min - 1, tsne_max + 1
                
                for ax_ in ax.ravel():
                    ax_.set_xlim(*coord_range), ax_.set_ylim(*coord_range)
                
                # Add subplot labels
                for ax_, letter in zip(ax, string.ascii_lowercase): 
                    plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=15, fontweight="bold")
                
                # Add KNN accuracy text to the figure
                embedding_clean = embedding_key.replace('X_', '').replace('_', '')
                viz_label = "UMAP" if 'umap' in embedding_key.lower() else "t-SNE"
                fig.text(0.5, 0.10, f"KNN({embedding_clean}): {orig_accuracy:.4f}    |    KNN({viz_label}): {tsne_accuracy:.4f}", 
                        ha='center', fontsize=12)
                
                # Generate output filename
                embedding_name = embedding_key.replace('X_', '').lower()
                output_pdf = f"tsne_plot_{base_filename}_{embedding_name}.pdf"
                
                # Save plot
                plt.savefig(output_pdf, dpi=600, bbox_inches="tight", transparent=True)
                print(f"Saved plot as {output_pdf}")
                plt.close()
                
            except Exception as e:
                print(f"Error processing embedding {embedding_key}: {str(e)}")
                continue
        
        print(f"\nCompleted processing all embeddings for {file_path}")

    # Example usage
    if __name__ == "__main__":
        # Process single file (uses same file as reference)
        compute_knn_tsne_all("/shared/home/christy.jo.manthara/batch-effect-analysis/output/baron_2016h_xin_2016_preprocessed_uce_adata_X_scvi_X_scanvi_X_uce_test.h5ad", skip_preprocessing=True)
        
        # Process with separate reference file
        # compute_knn_tsne_all("xin_2016_scGPT.h5ad", reference_file="baron_2016h_scGPT.h5ad", skip_preprocessing=True)