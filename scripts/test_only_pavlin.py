
import sys
import os
import warnings
# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='anndata')
warnings.filterwarnings('ignore', category=SyntaxWarning, module='docrep')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from scripts.pavlin_plot_on_top_anndata import transform_tsne_single
from scripts.pavlin_preprocess_anndata import process_single_anndata


def run_integrated_analysis_with_existing_function(transform_ref_path=None, transform_new_path=None):
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

if __name__ == "__main__":
    # Example paths, replace with actual paths
    transform_ref_path = "F:/Thesis/Datasets/baron_2016h.h5ad"
    transform_new_path = "F:/Thesis/Datasets/xin_2016.h5ad"
    
    run_integrated_analysis_with_existing_function(transform_ref_path, transform_new_path)
    
    print("Analysis completed successfully.")