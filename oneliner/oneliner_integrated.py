import sys
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='anndata')
warnings.filterwarnings('ignore', category=SyntaxWarning, module='docrep')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.knn_plot_test_simple_re2 import compute_knn_tsne_simple
from scripts.pavlin_plot_on_top_anndata import transform_tsne_single
from scripts.pavlin_preprocess_anndata import process_single_anndata
from scripts.knn_plot_table import create_results_table
from scripts.data_utils import preprocessing
from scripts.compute_embeddings_scvi_scanvi_uce import compute_embeddings_scvi_scanvi_uce as ce
from scripts.test_harmony_cells import run_harmony_correction_simple, run_harmony_correction
from scripts import knn_plot_test_simple

def run_integrated_analysis_with_existing_function(file_path=None, reference_file=None, 
                                                   transform_ref_path=None, transform_new_path=None):
    """
    Run both pipelines and let your existing create_results_table function handle everything.
    Your function already accumulates results and creates comprehensive tables!
    
    Args:
        file_path: Main dataset path (currently unused but kept for compatibility)
        reference_file: Reference file path (currently unused but kept for compatibility)
        transform_ref_path: Path to reference dataset for transformation
        transform_new_path: Path to new dataset to be transformed
    
    Returns:
        tuple: (results_dict, df_query, df_reference)
    """
    
    if not transform_ref_path or not transform_new_path:
        raise ValueError("Both transform_ref_path and transform_new_path must be provided")
    
    print("=== Running First Pipeline (Oneliner) ===")
    
    try:
        # Step 1: Load and preprocess data
        print("Step 1: Loading and preprocessing data...")
        combined_data = preprocessing.load_and_preprocess_multi_embedder_individual_pca(
            file1=transform_ref_path, 
            file2=transform_new_path,
            save=False,          # Don't save intermediate files
            split_output=False   # Get combined AnnData object
        )
        
        print(f"📦 Available keys in uns after step 1 (Unstructured Data):")
        print(list(combined_data.uns.keys()) if hasattr(combined_data, 'uns') else "No 'uns' attribute")
        
        # Step 2: Add embeddings using scVI, scANVI, and UCE
        print("Step 2: Computing embeddings (scVI, scANVI, UCE)...")
        embedded_adata = ce(combined_data)
        
        print(f"📦 Available keys in uns after step 2:")
        print(list(embedded_adata.uns.keys()) if hasattr(embedded_adata, 'uns') else "No 'uns' attribute")
        
        # Step 2.5: Add harmony embedding
        print("Step 2.5: Running Harmony correction...")
        adata_harmony = run_harmony_correction_simple(embedded_adata, batch_key="source")
        
        print(f"📦 Available keys in uns after step 2.5:")
        print(list(adata_harmony.uns.keys()) if hasattr(adata_harmony, 'uns') else "No 'uns' attribute")
        
        # Step 3: Compute KNN and t-SNE results
        print("Step 3: Computing KNN and t-SNE...")
        results1, metadata1 = compute_knn_tsne_simple(adata_harmony)
        
    except Exception as e:
        print(f"❌ Error in first pipeline: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise
    
    print("\n=== Running Second Pipeline (Pavlin's) ===")
    
    try:
        # Step 1: Process reference data
        print("Step 1: Processing reference data...")
        transformed_ref_adata = process_single_anndata(transform_ref_path)
        
        # Step 2: Transform new data and get results
        print("Step 2: Transforming new data and computing results...")
        transformed_data, results2 = transform_tsne_single(transformed_ref_adata, transform_new_path)
        
    except Exception as e:
        print(f"❌ Error in second pipeline: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        # If second pipeline fails, continue with first pipeline results only
        print("⚠️  Continuing with first pipeline results only...")
        results2 = {}
        
    print("\n=== Integrating Results ===")
    
    # Merge the results tables
    if results2:
        integrated_results = merge_results_tables(results1, results2)
        print(f"✅ Merged results from both pipelines")
        print(f"   - First pipeline methods: {len(results1)}")
        print(f"   - Second pipeline methods: {len(results2)}")
        print(f"   - Total integrated methods: {len(integrated_results)}")
    else:
        integrated_results = results1
        print(f"ℹ️  Using first pipeline results only: {len(integrated_results)} methods")
    
    # Use your existing create_results_table function
    try:
        source1 = metadata1.get('ref_source', 'ref') if metadata1 else 'ref'
        source2 = metadata1.get('query_source', 'query') if metadata1 else 'query'
        base_filename = f"integrated_{source1}_{source2}"
        
        print(f"Creating results table with sources: {source1} vs {source2}")
        
        df_query, df_reference = create_results_table(
            integrated_results,
            source1, 
            source2, 
            base_filename, 
            metadata=metadata1
        )
        
        print(f"✅ Results successfully processed and saved!")
        return integrated_results, df_query, df_reference
        
    except Exception as e:
        print(f"❌ Error in create_results_table: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        # Return raw results even if table creation fails
        return integrated_results, None, None

def merge_results_tables(table1, table2):
    """
    Merge two results tables, handling potential conflicts.
    
    Args:
        table1 (dict): First results table
        table2 (dict): Second results table
    
    Returns:
        dict: Merged results table
    """
    merged_table = {}
    
    # Copy all entries from first table
    for method_name, metrics in table1.items():
        merged_table[method_name] = metrics.copy() if hasattr(metrics, 'copy') else metrics
    
    # Add entries from second table
    for method_name, metrics in table2.items():
        if method_name in merged_table:
            # If method exists in both, merge the metrics
            if isinstance(merged_table[method_name], dict) and isinstance(metrics, dict):
                merged_table[method_name].update(metrics)
            else:
                # If not dict, create a combined entry
                merged_table[f"{method_name}_pipeline2"] = metrics
        else:
            # If method is new, add it
            merged_table[method_name] = metrics.copy() if hasattr(metrics, 'copy') else metrics
    
    return merged_table

def main():
    """
    Main execution function with error handling and example paths.
    """
    
    # Example file paths - modify these for your actual data
    transform_ref_path = "F:/Thesis/Datasets/baron_2016h.h5ad"
    transform_new_path = "F:/Thesis/Datasets/xin_2016.h5ad"
    
    print("🚀 Starting Integrated Analysis")
    print(f"Reference dataset: {transform_ref_path}")
    print(f"Query dataset: {transform_new_path}")
    
    # Check if files exist
    if not os.path.exists(transform_ref_path):
        print(f"❌ Reference file not found: {transform_ref_path}")
        return
    
    if not os.path.exists(transform_new_path):
        print(f"❌ Query file not found: {transform_new_path}")
        return
    
    try:
        # Run integrated analysis
        integrated_results, df_query, df_reference = run_integrated_analysis_with_existing_function(
            transform_ref_path=transform_ref_path,
            transform_new_path=transform_new_path
        )
        
        print("\n" + "="*50)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*50)
        
        if integrated_results:
            print(f"📊 Results Summary:")
            print(f"   - Total methods analyzed: {len(integrated_results)}")
            print(f"   - Methods included:")
            for method in integrated_results.keys():
                print(f"     • {method}")
        
        if df_query is not None:
            print(f"   - Query results shape: {df_query.shape}")
        if df_reference is not None:
            print(f"   - Reference results shape: {df_reference.shape}")
            
        print("\n✅ Results have been saved to:")
        print("   - knn_results_accumulated.pkl")
        print("   - CSV files with comprehensive results")
        print("   - Excel files with metadata")
        print("   - Enhanced visualizations")
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED!")
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Print traceback for debugging
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()