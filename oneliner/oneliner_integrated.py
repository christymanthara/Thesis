from scripts.knn_plot_test_simple_re2 import compute_knn_tsne_simple
from scripts.pavlin_plot_on_top_anndata import transform_tsne_single
from scripts.pavlin_preprocess_anndata import process_single_anndata
from scripts.knn_plot_table import create_results_table
from scripts.data_utils import preprocessing
# from batch-effect-analysis import UCE.compute_uce_embeddings
from scripts.compute_embeddings_scvi_scanvi_uce import compute_embeddings_scvi_scanvi_uce as ce

from scripts.test_harmony_cells import run_harmony_correction_simple, run_harmony_correction
from scripts import knn_plot_test_simple

def run_integrated_analysis_with_existing_function(file_path, reference_file=None, 
                                                   transform_ref_path=None, transform_new_path=None):
    """
    Run both pipelines and let your existing create_results_table function handle everything.
    Your function already accumulates results and creates comprehensive tables!
    """
    
    print("=== Running First Pipeline Oneliner===")
    # Run first pipeline (modify to return results_table and metadata)
    
    combined_data = preprocessing.load_and_preprocess_multi_embedder_individual_pca(
        
    file1="F:/Thesis/Datasets/baron_2016h.h5ad", 
    # file2="F:/Thesis/muraro_transformed.h5ad",
    file2="F:/Thesis/Datasets/xin_2016.h5ad",
    save=False,          # Saves all files
    split_output=False   # Getting the combined AnnData object
    )
    
    print(f"📦 Available keys in uns after step1 (Unstructured Data):")
    print(list(combined_data.uns.keys()))
    
    # 2.Adding embeddings using scVI, scANVI, and UCE
    embedded_adata = ce(combined_data)
    
    print(f"📦 Available keys in uns after step2 (Unstructured Data):")
    print(list(embedded_adata.uns.keys()))
    
    # 2.5 Add harmony embedding
    adata_harmony = run_harmony_correction_simple(embedded_adata, batch_key="source")
    
    print(f"📦 Available keys in uns after step2.5 (Unstructured Data):")
    print(list(adata_harmony.uns.keys()))
    
    results1, metadata1 = compute_knn_tsne_simple(adata_harmony)
    
    if transform_ref_path and transform_new_path:
        print("\n=== Running Second Pipeline Pavlins===")
        # Run second pipeline 
        
        print("Running the first step:embedding the reference data")
        
        transformed_ref_adata = process_single_anndata(transform_ref_path) #
        print("Running the second step:transforming the new data onto the reference embedding, plotting and saving")
        transformed_data, results2 = transform_tsne_single(transformed_ref_adata, transform_new_path)
        
        print("\n=== Integrating Results ===")
        # Merge the results tables
        integrated_results = merge_results_tables(results1, results2)
        
        # Use your existing create_results_table function!
        # It will automatically accumulate these results with previous runs
        source1 = metadata1.get('ref_source', 'ref')
        source2 = metadata1.get('query_source', 'query')
        base_filename = f"integrated_{source1}_{source2}"
        
        # Your function handles everything - accumulation, visualization, saving!
        df_query, df_reference = create_results_table(
            integrated_results,  # This now contains results from BOTH pipelines
            source1, 
            source2, 
            base_filename, 
            metadata=metadata1
        )
        
        print(f"✅ Integrated results saved and accumulated!")
        print(f"   - Results from first pipeline: {len(results1)} methods")
        print(f"   - Results from second pipeline: {len(results2)} methods") 
        print(f"   - Total integrated results: {len(integrated_results)} methods")
        
        return integrated_results, df_query, df_reference
    
    else:
        # If only running first pipeline, still use your existing function
        source1 = metadata1.get('ref_source', 'ref')
        source2 = metadata1.get('query_source', 'query')
        base_filename = f"first_pipeline_{source1}_{source2}"
        
        df_query, df_reference = create_results_table(
            results1,
            source1,
            source2, 
            base_filename,
            metadata=metadata1
        )
        
        return results1, df_query, df_reference

def merge_results_tables(table1, table2):
    """
    Simple merge function - your create_results_table will handle the rest!
    """
    merged_table = {}
    
    # Copy all entries from first table
    for method_name, metrics in table1.items():
        merged_table[method_name] = metrics.copy()
    
    # Add entries from second table
    for method_name, metrics in table2.items():
        if method_name in merged_table:
            # If method exists in both, merge the metrics
            merged_table[method_name].update(metrics)
        else:
            # If method is new, add it
            merged_table[method_name] = metrics.copy()
    
    return merged_table

# Your existing create_results_table function will:
# ✅ Automatically accumulate results across multiple runs
# ✅ Handle the source comparison key matching
# ✅ Create comprehensive DataFrames with all methods
# ✅ Generate visualizations and save files
# ✅ Manage persistent storage

# Example usage:
def example_usage():
    """
    Example of how to use the integrated analysis.
    """
    
    # Run integrated analysis
    integrated_results, df_query, df_reference = run_integrated_analysis_with_existing_function(
        file_path="your_main_dataset.h5ad",
        reference_file=None,  # Will split by source if available
        transform_ref_path="F:/Thesis/Datasets/baron_2016h.h5ad",
        transform_new_path="F:/Thesis/Datasets/xin_2016.h5ad"
    )
    
    # Your existing function has already:
    # - Saved results to knn_results_accumulated.pkl
    # - Created comprehensive CSV files
    # - Generated enhanced visualizations
    # - Saved Excel files with metadata
    
    print("✅ All results integrated and saved!")
    print(f"Integrated results contain {len(integrated_results)} methods:")
    for method in integrated_results.keys():
        print(f"  - {method}")

# The beauty of this approach:
# 1. Your existing create_results_table function does ALL the heavy lifting
# 2. It already handles accumulation across runs with the same source comparison
# 3. It creates comprehensive tables with metadata
# 4. You just need to merge the results tables before passing to it
# 5. Results from both pipelines will appear in the same row (same source comparison key)