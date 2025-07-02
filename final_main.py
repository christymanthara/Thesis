import sys
import os
import subprocess
import scanpy as sc
from collections import defaultdict
from pathlib import Path
import pandas as pd
import anndata
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from simple_code import preprocessing
from preprocessing import preprocessing_metadata
from simple_code.compute_embeddings_scvi_scanvi_uce import compute_embeddings_scvi_scanvi_uce as ce
from harmony.test_harmony_cells import run_harmony_correction_with_original_x
# from simple_code.knn_plot_table_integrated import create_results_table
from simple_code.knn_plot_table_simple import create_results_table
from simple_code.pavlin_preprocess_anndata import process_single_anndata
from simple_code.pavlin_plot_on_top_anndata import transform_tsne_single
# from simple_code.knn_plot_test_integrated_plot import compute_knn_tsne_simple_with_transformation_plots
from simple_code.knn_plot_test_integrated_plot_met import compute_knn_tsne_simple_with_transformation_plots

from anndata import AnnData


def run_uce_processing(adata_path, output_dir="../output/", 
                      model_loc="/shared/home/christy.jo.manthara/batch-effect-analysis/UCE/model_files/33l_8ep_1024t_1280.torch", 
                      nlayers=33, uce_dir="UCE"):
    """
    Run UCE processing on a single AnnData file using command line
    
    Args:
        adata_path (str): Path to the input .h5ad file
        output_dir (str): Directory where UCE output will be saved
        model_loc (str): Path to the UCE model file
        nlayers (int): Number of layers parameter for UCE
        uce_dir (str): Directory containing the UCE eval_single_anndata.py script
    
    Returns:
        str: Path to the generated UCE file
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the full path to the UCE script
    uce_script_path = os.path.join(uce_dir, "eval_single_anndata.py")
    
    # Check if the UCE script exists
    if not os.path.exists(uce_script_path):
        print(f"Error: UCE script not found at {uce_script_path}")
        return None
    
    # Construct the UCE command
    python_executable = sys.executable
    uce_command = [
        python_executable, uce_script_path,
        "--adata_path", adata_path,
        "--filter", "False",
        "--dir", output_dir,
        "--nlayers", str(nlayers),
        "--model_loc", model_loc
    ]
    
    try:
        print(f"Running UCE processing for {adata_path}...")
        result = subprocess.run(uce_command, check=True, capture_output=True, text=True)
        print(f"UCE processing completed successfully for {adata_path}")
        print("STDOUT:", result.stdout)
        
        # Generate expected output filename
        input_filename = Path(adata_path).stem
        
        # Convert relative path to absolute path if needed
        if os.path.isabs(output_dir):
            uce_output_path = os.path.join(output_dir, f"{input_filename}_uce_adata.h5ad")
        else:
            abs_output_dir = os.path.abspath(output_dir)
            uce_output_path = os.path.join(abs_output_dir, f"{input_filename}_uce_adata.h5ad")
        
        # Verify the output file was created
        if os.path.exists(uce_output_path):
            print(f"UCE output file created: {uce_output_path}")
            return uce_output_path
        else:
            print(f"Warning: Expected UCE output file not found at {uce_output_path}")
            
            # Additional debugging: try to find the file in the current directory structure
            possible_paths = [
                os.path.join("../output/", f"{input_filename}_uce_adata.h5ad"),
                os.path.join("./output/", f"{input_filename}_uce_adata.h5ad"),
                os.path.join("output/", f"{input_filename}_uce_adata.h5ad"),
                f"{input_filename}_uce_adata.h5ad"
            ]
            
            print("Searching for output file in alternative locations:")
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                print(f"  Checking: {abs_path}")
                if os.path.exists(abs_path):
                    print(f"  Found at: {abs_path}")
                    return abs_path
            
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running UCE processing for {adata_path}: {e}")
        print("STDERR:", e.stderr)
        return None




def process_datasets_with_uce(dataset_paths, uce_dir="UCE", output_dir="../output/", 
                             model_loc="/shared/home/christy.jo.manthara/batch-effect-analysis/UCE/model_files/33l_8ep_1024t_1280.torch", 
                             nlayers=33):
    """
    Process multiple datasets with UCE before combining them
    
    Args:
        dataset_paths (list): List of paths to dataset (.h5ad files)
        uce_dir (str): Directory containing the UCE eval_single_anndata.py script
        output_dir (str): Directory where UCE output will be saved
        model_loc (str): Path to the UCE model file
        nlayers (int): Number of layers parameter for UCE
        
    Returns:
        list: Paths to the UCE-processed files
    """
    uce_files = []
    
    # Process each dataset with UCE
    for dataset_path in dataset_paths:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_path} with UCE")
        print(f"{'='*50}")
        
        uce_output = run_uce_processing(
            adata_path=dataset_path,
            output_dir=output_dir,
            model_loc=model_loc,
            nlayers=nlayers,
            uce_dir=uce_dir
        )
        
        if uce_output:
            uce_files.append(uce_output)
        else:
            print(f"Failed to process {dataset_path} with UCE")
            # Fallback to original file if UCE processing fails
            uce_files.append(dataset_path)
    
    return uce_files




def analyze_datasets_by_tissue(datasets_dir="../Datasets/"):
    """
    Analyze all .h5ad files in the datasets directory and group them by tissue.
    
    Parameters:
    -----------
    datasets_dir : str
        Path to the datasets directory
        
    Returns:
    --------
    dict: Dictionary with tissue as key and list of (filepath, cell_count) tuples as values
    """
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
    
    tissue_groups = defaultdict(list)
    
    # Find all .h5ad files
    h5ad_files = list(datasets_path.glob("*.h5ad"))
    
    if not h5ad_files:
        raise ValueError(f"No .h5ad files found in {datasets_dir}")
    
    print(f"Found {len(h5ad_files)} .h5ad files in {datasets_dir}")
    
    # Analyze each file
    for filepath in h5ad_files:
        try:
            # Load only the metadata to check tissue information
            adata = sc.read_h5ad(filepath)
            
            # Check if 'tissue' exists in uns
            if 'tissue' in adata.uns:
                tissue = adata.uns['tissue']
                cell_count = adata.n_obs
                tissue_groups[tissue].append((str(filepath), cell_count))
                print(f"File: {filepath.name} | Tissue: {tissue} | Cells: {cell_count}")
            else:
                print(f"Warning: 'tissue' not found in uns for {filepath.name}")
                # Try to infer tissue from filename or other metadata
                tissue = "unknown"
                cell_count = adata.n_obs
                tissue_groups[tissue].append((str(filepath), cell_count))
                
        except Exception as e:
            print(f"Error reading {filepath.name}: {str(e)}")
            continue
    
    # Sort files within each tissue group by cell count (descending)
    for tissue in tissue_groups:
        tissue_groups[tissue].sort(key=lambda x: x[1], reverse=True)
    
    return dict(tissue_groups)

def process_tissue_group(tissue, file_info_list, enable_uce=True, 
                        uce_config=None, batch_key='batch',
                        enable_pavlin=True,  # NEW
                        enable_integrated=True,  # NEW
                        analysis_mode='standard'):  # NEW
    """
    Process a group of files from the same tissue with integrated UCE processing.
    
    Parameters:
    -----------
    tissue : str
        Tissue name
    file_info_list : list
        List of (filepath, cell_count) tuples sorted by cell count (descending)
    enable_uce : bool
        Whether to enable UCE processing
    uce_config : dict
        UCE configuration parameters
    batch_key : str
        Batch key for harmony correction
    enable_pavlin : bool
        Whether to enable Pavlin's pipeline
    enable_integrated : bool
        Whether to enable integrated analysis
    analysis_mode : str
        Analysis mode ('standard', etc.)
        
    Returns:
    --------
    dict: Dictionary containing results for each iteration
    """
    
    if len(file_info_list) < 2:
        print(f"Skipping tissue '{tissue}' - only {len(file_info_list)} file(s) available")
        return {}
    
    print(f"\n{'='*50}")
    print(f"Processing tissue: {tissue}")
    print(f"Files available: {len(file_info_list)}")
    for i, (filepath, cell_count) in enumerate(file_info_list):
        print(f"  {i+1}. {Path(filepath).name} ({cell_count} cells)")
    print(f"{'='*50}")
    
    # Default UCE configuration
    if uce_config is None:
        uce_config = {
            'uce_dir': "/shared/home/christy.jo.manthara/batch-effect-analysis/UCE",
            'output_dir': "../output/",
            'model_loc': "/shared/home/christy.jo.manthara/batch-effect-analysis/UCE/model_files/33l_8ep_1024t_1280.torch",
            'nlayers': 33
        }
    
    # Store results for all iterations
    all_results = {}
    
    # Process files iteratively: largest as file1, all others as file2
    for i in range(len(file_info_list) - 1):
        file1_path, file1_cells = file_info_list[i]
        file2_path, file2_cells = file_info_list[i+1]
        
        print(f"\nIteration {i+1} for tissue '{tissue}':")
        print(f"File1 (reference): {Path(file1_path).name} ({file1_cells} cells)")
        print(f"File2 (query): {Path(file2_path).name} ({file2_cells} cells)")
        
        # Initialize variables to track success/failure of each step
        pipeline_status = {
            'file_check': False,
            'uce_processing': False,
            'uce_knn': False,
            'preprocessing': False,
            'embedding': False,
            'harmony': False,
            'knn_analysis': False,
            'second_pipeline': False,
            'results_integration': False,
            'table_creation': False
        }
        
        # Initialize return variables
        results_dict = {}
        df_query = None
        df_reference = None
        metadata = {}
        
        # =================================================================
        # STEP -1: FILE EXISTENCE CHECK
        # =================================================================
        print("\n" + "="*40)
        print("STEP -1: FILE EXISTENCE CHECK")
        print("="*40)
        
        try:
            if not os.path.exists(file1_path):
                raise FileNotFoundError(f"Reference file not found: {file1_path}")
            
            if not os.path.exists(file2_path):
                raise FileNotFoundError(f"Query file not found: {file2_path}")
            
            print("✅ Both files exist and are accessible")
            pipeline_status['file_check'] = True
            
        except Exception as e:
            print(f"❌ File check failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            continue

        # Initialize processed file paths
        processed_file1 = file1_path
        processed_file2 = file2_path
        
        try:
            # =================================================================
            # STEP 0: UCE EMBEDDINGS PROCESSING
            # =================================================================
            if enable_uce:
                print(f"\n{'='*40}")
                print("STEP 0: UCE EMBEDDINGS PROCESSING")
                print(f"{'='*40}")
                
                try:
                    print("🔄 Starting UCE processing...")
                    uce_processed_files = process_datasets_with_uce(
                        dataset_paths=[file1_path, file2_path],
                        **uce_config
                    )
                    
                    if len(uce_processed_files) != 2:
                        print("⚠️  Could not process both datasets with UCE")
                        raise Exception("UCE processing returned incomplete results")
                    else:
                        print(f"✅ UCE processing successful:")
                        for j, file_path in enumerate(uce_processed_files, 1):
                            print(f"     File {j}: {file_path}")
                        processed_file1 = uce_processed_files[0]
                        processed_file2 = uce_processed_files[1]
                        pipeline_status['uce_processing'] = True
                        
                except Exception as e:
                    print(f"❌ UCE processing failed: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    print("⚠️  Continuing with original files...")
                    processed_file1 = file1_path
                    processed_file2 = file2_path
                    pipeline_status['uce_processing'] = False
            else:
                print("ℹ️  UCE processing disabled, using original files...")
                pipeline_status['uce_processing'] = True  # Mark as successful (skipped)
                
            # =================================================================
            # STEP 0.1: KNN AND t-SNE ANALYSIS FOR UCE
            # =================================================================
            print("\n" + "="*40)
            print("STEP 0.1: KNN AND t-SNE ANALYSIS for UCE")
            print("="*40)
            
            print("Adding the source field")
            def extract_filename(path):
                filename = os.path.basename(path)  # Get file name
                return filename.rsplit('.h5ad', 1)[0]  # Remove the extension

            # Store original source names for later splitting
            source1_name = extract_filename(processed_file1).removesuffix('_uce_adata') 
            source2_name = extract_filename(processed_file2).removesuffix('_uce_adata') 
            
            file1 = anndata.read_h5ad(processed_file1)
            file2 = anndata.read_h5ad(processed_file2)

            # Add source labels
            file1.obs["source"] = pd.Categorical([source1_name] * file1.n_obs)
            file2.obs["source"] = pd.Categorical([source2_name] * file2.n_obs)

            # Debug print to check source labels
            print("Unique source labels in adata:", file1.obs["source"].unique())
            print("Unique source labels in new:", file2.obs["source"].unique())
            
            results0 = {}
            metadata0 = {}
            
            try:
                print("🔄 Running KNN and t-SNE analysis on UCE output...")
                results0, metadata0 = compute_knn_tsne_simple_with_transformation_plots(file1, reference_file=file2)
                
                if not results0:
                    raise Exception("KNN analysis on uce returned empty results")
                
                print(f"✅ KNN and t-SNE analysis on uce successful!")
                print(f"   - Number of methods analyzed: {len(results0)}")
                print(f"   - Methods: {list(results0.keys())}")
                
                pipeline_status['uce_knn'] = True
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"❌ KNN and t-SNE analysis failed for UCE: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                print("⚠️  Continuing with empty results for zeroth pipeline on UCE")
                results0 = {}
                metadata0 = {}
                pipeline_status['uce_knn'] = False

            # =================================================================
            # Step 1: Loading and preprocessing the datasets
            # =================================================================
            print(f"\n{'='*40}")
            print("STEP 1: LOADING AND PREPROCESSING DATASETS")
            print(f"{'='*40}")
            
            # Use original file paths for preprocessing, not UCE processed files
            original_file1_path = file1_path
            original_file2_path = file2_path
            
            combined_data = None
            try:
                print("🔄 Loading and preprocessing datasets...")
                print(f"   - File 1: {original_file1_path}")
                print(f"   - File 2: {original_file2_path}")
                
                combined_data = preprocessing_metadata.load_and_preprocess_multi_embedder(
                    file1=original_file1_path,
                    file2=original_file2_path,
                    save=False,  # Changed from True to False like in run_simple_pipeline
                    split_output=False
                )
                
                if combined_data is None:
                    raise Exception("Preprocessing returned None")
                
                print(f"✅ Preprocessing successful!")
                print(f"   - Combined data shape: {combined_data.shape}")
                print(f"   - Available layers: {list(combined_data.layers.keys()) if hasattr(combined_data, 'layers') else 'None'}")
                print(f"   - Available obsm: {list(combined_data.obsm.keys()) if hasattr(combined_data, 'obsm') else 'None'}")
                print(f"   - Available uns keys: {list(combined_data.uns.keys()) if hasattr(combined_data, 'uns') else 'None'}")
                
                pipeline_status['preprocessing'] = True
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"❌ Preprocessing failed: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                print("🛑 Cannot continue without preprocessed data")
                continue
            
            # =================================================================
            # Step 2: Adding embeddings using scVI, scANVI
            # =================================================================
            print(f"\n{'='*40}")
            print("STEP 2: ADDING ADDITIONAL EMBEDDINGS (scVI, scANVI, UCE)")
            print(f"{'='*40}")
            
            embedded_adata = combined_data  # Default fallback
            try:
                print("🔄 Adding embeddings...")
                embedded_adata = ce(combined_data)
                
                if embedded_adata is None:
                    raise Exception("Embedding function returned None")
                
                print(f"✅ Embedding successful!")
                print(f"   - Embedded data shape: {embedded_adata.shape}")
                print(f"   - Available obsm after embedding: {list(embedded_adata.obsm.keys()) if hasattr(embedded_adata, 'obsm') else 'None'}")
                print(f"   - Available uns keys after embedding: {list(embedded_adata.uns.keys()) if hasattr(embedded_adata, 'uns') else 'None'}")
                
                # Check for metadata preservation
                metadata_cols = [col for col in embedded_adata.obs.columns if col.startswith('dataset_')]
                if metadata_cols:
                    print(f"📦 Metadata preserved in .obs columns:")
                    for col in metadata_cols:
                        print(f"  {col}: {embedded_adata.obs[col].unique()}")
                
                pipeline_status['embedding'] = True
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"❌ Embedding failed: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                print("⚠️  Continuing with preprocessed data (no additional embeddings)")
                embedded_adata = combined_data
                pipeline_status['embedding'] = False
            
            # =================================================================
            # Step 2.5: Run Harmony correction
            # =================================================================
            print(f"\n{'='*40}")
            print("STEP 2.5: HARMONY CORRECTION")
            print(f"{'='*40}")
            
            adata_harmony = embedded_adata  # Default fallback
            try:
                print(f"🔄 Running harmony correction with batch_key='{batch_key}'...")
                
                # Check if batch_key exists in obs
                if batch_key not in embedded_adata.obs.columns:
                    print(f"⚠️  Batch key '{batch_key}' not found in obs. Available columns:")
                    print(f"   {list(embedded_adata.obs.columns)}")
                    raise Exception(f"Batch key '{batch_key}' not found")
                
                adata_harmony = run_harmony_correction_with_original_x(
                    embedded_adata,
                    batch_key=batch_key,
                    theta=2
                )
                
                if adata_harmony is None:
                    raise Exception("Harmony correction returned None")
                
                print(f"✅ Harmony correction successful!")
                print(f"   - Harmony data shape: {adata_harmony.shape}")
                print(f"   - Available obsm after harmony: {list(adata_harmony.obsm.keys()) if hasattr(adata_harmony, 'obsm') else 'None'}")
                print(f"   - Available uns keys after harmony: {list(adata_harmony.uns.keys()) if hasattr(adata_harmony, 'uns') else 'None'}")
                
                # Check for metadata preservation
                metadata_cols = [col for col in adata_harmony.obs.columns if col.startswith('dataset_')]
                if metadata_cols:
                    print(f"📦 Metadata preserved in .obs columns after harmony:")
                    for col in metadata_cols:
                        print(f"  {col}: {adata_harmony.obs[col].unique()}")
                
                pipeline_status['harmony'] = True
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"❌ Harmony correction failed: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                print("⚠️  Continuing with embedded data (no harmony correction)")
                adata_harmony = embedded_adata
                pipeline_status['harmony'] = False
            
            # =================================================================
            # Step 3: Run the knn plot test and get results
            # =================================================================
            print(f"\n{'='*40}")
            print("STEP 3: KNN AND t-SNE ANALYSIS")
            print(f"{'='*40}")
            
            results1 = {}
            metadata1 = {}
            try:
                print("🔄 Running KNN and t-SNE analysis...")
                results1, metadata1 = compute_knn_tsne_simple_with_transformation_plots(adata_harmony)
                
                print("\n📊 Metadata from analysis:")
                for key, value in metadata1.items():
                    print(f"   - {key}: {value}")
                    
                if not metadata1:
                    raise ValueError("Metadata from KNN and t-SNE analysis is empty")
                
                if not results1:
                    raise Exception("KNN analysis returned empty results")
                
                print(f"✅ KNN and t-SNE analysis successful!")
                print(f"   - Number of methods analyzed: {len(results1)}")
                print(f"   - Methods: {list(results1.keys())}")
                
                pipeline_status['knn_analysis'] = True
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"❌ KNN and t-SNE analysis failed: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                print("⚠️  Continuing with empty results for first pipeline")
                results1 = {}
                metadata1 = {}
                pipeline_status['knn_analysis'] = False
            
            # =================================================================
            # Step 4: Run second pipeline (Pavlin's) if enabled
            # =================================================================
            results2 = {}
            if enable_pavlin:
                print(f"\n{'='*40}")
                print("STEP 4: PAVLIN'S PIPELINE")
                print(f"{'='*40}")
                
                try:
                    # Step 1: Process reference data
                    print("🔄 Processing reference data...")
                    transformed_ref_adata = process_single_anndata(file1_path)
                    
                    if transformed_ref_adata is None:
                        raise Exception("Reference data processing returned None")
                    
                    print("✅ Reference data processed successfully")
                    
                    # Step 2: Transform new data and get results
                    print("🔄 Transforming new data and computing results...")
                    transformed_data, results2 = transform_tsne_single(transformed_ref_adata, file2_path)
                    
                    if not results2:
                        raise Exception("Second pipeline returned empty results")
                    
                    print(f"✅ Pavlin's pipeline completed with {len(results2)} methods")
                    print(f"   - Methods: {list(results2.keys())}")
                    
                    pipeline_status['second_pipeline'] = True
                    
                except Exception as e:
                    print(f"❌ Error in Pavlin's pipeline: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    print("Traceback:")
                    traceback.print_exc()
                    print("⚠️  Continuing without second pipeline results")
                    results2 = {}
                    pipeline_status['second_pipeline'] = False
            
            # =================================================================
            # Step 5: Integrate results if enabled
            # =================================================================
            print(f"\n{'='*40}")
            print("STEP 5: INTEGRATING RESULTS")
            print(f"{'='*40}")
            
            integrated_results = {}
            try:
                print("🔄 Merging results from all pipelines...")
                
                # Collect available results with their names
                available_results = []
                pipeline_names = []
                
                if results0:
                    available_results.append(results0)
                    pipeline_names.append("uce")
                    print(f"   - UCE pipeline: {len(results0)} methods")
                
                if results1:
                    available_results.append(results1)
                    pipeline_names.append("main")
                    print(f"   - Main pipeline: {len(results1)} methods")
                
                if results2:
                    available_results.append(results2)
                    pipeline_names.append("pavlin")
                    print(f"   - Pavlin pipeline: {len(results2)} methods")
                
                if len(available_results) >= 1:
                    if enable_integrated and len(available_results) > 1:
                        # Use the enhanced merge function for multiple results
                        integrated_results = merge_multiple_results_tables(available_results, pipeline_names)
                        print(f"✅ Results merged from {len(available_results)} pipelines")
                    else:
                        # Use simple merge for two results or single result
                        if len(available_results) == 1:
                            integrated_results = available_results[0]
                        else:
                            integrated_results = merge_results_tables(results1, results2) if results1 and results2 else (results1 or results2)
                        print(f"✅ Using integrated results: {len(integrated_results)} methods")
                    
                    print(f"   - Total integrated methods: {len(integrated_results)}")
                    
                else:
                    raise Exception("No results available from any pipeline")
                
                pipeline_status['results_integration'] = True
                
            except Exception as e:
                print(f"❌ Results integration failed: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                print("⚠️  Using whatever results are available")
                # Fallback logic
                if results0:
                    integrated_results = results0
                elif results1:
                    integrated_results = results1
                elif results2:
                    integrated_results = results2
                pipeline_status['results_integration'] = False
            
            # =================================================================
            # Step 6: Create comprehensive results tables
            # =================================================================
            print(f"\n{'='*40}")
            print("STEP 6: GENERATING RESULTS TABLES")
            print(f"{'='*40}")
            
            try:
                if not integrated_results:
                    raise Exception("No integrated results available for table creation")
                
                print("🔄 Creating comprehensive results tables...")
                
                # Extract source information
                source1 = metadata1.get('ref_source', 'ref') if metadata1 else 'ref'
                source2 = metadata1.get('query_source', 'query') if metadata1 else 'query'
                base_filename = f"{tissue}_iter{i+1}_{source1}_{source2}"
                
                print(f"   - Source 1: {source1}")
                print(f"   - Source 2: {source2}")
                print(f"   - Base filename: {base_filename}")
                
                print("metadata1 passed to step 6: create results table")
                if metadata1:
                    print(" | ".join([f"{k}: {v}" for k, v in metadata1.items()]))
                
                df_query, df_reference = create_results_table(
                    integrated_results,
                    source1, 
                    source2, 
                    base_filename, 
                    metadata=metadata1
                )
                
                print(f"✅ Results tables created successfully!")
                if df_query is not None:
                    print(f"   - Query results shape: {df_query.shape}")
                if df_reference is not None:
                    print(f"   - Reference results shape: {df_reference.shape}")
                
                pipeline_status['table_creation'] = True
                
                # Store results for this iteration
                all_results[f"iteration_{i+1}"] = {
                    'results': integrated_results,
                    'df_query': df_query,
                    'df_reference': df_reference,
                    'metadata': metadata1,
                    'files': {
                        'reference': Path(file1_path).name,
                        'query': Path(file2_path).name
                    },
                    'pipeline_status': pipeline_status
                }
                
                print(f"✅ Results successfully processed and saved for iteration {i+1}!")
                
            except Exception as e:
                print(f"❌ Results table creation failed: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                print("⚠️  Returning raw results without formatted tables")
                
                # Store raw results even if table creation fails
                all_results[f"iteration_{i+1}"] = {
                    'results': integrated_results,
                    'df_query': None,
                    'df_reference': None,
                    'metadata': metadata1,
                    'files': {
                        'reference': Path(file1_path).name,
                        'query': Path(file2_path).name
                    },
                    'pipeline_status': pipeline_status
                }
                pipeline_status['table_creation'] = False
            
            # =================================================================
            # ITERATION SUMMARY
            # =================================================================
            print(f"\n{'='*40}")
            print(f"🎉 ITERATION {i+1} COMPLETED!")
            print(f"{'='*40}")
            
            print("📊 Pipeline Status Summary:")
            for step, status in pipeline_status.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {step.replace('_', ' ').title()}: {'Success' if status else 'Failed'}")
            
            successful_steps = sum(pipeline_status.values())
            total_steps = len(pipeline_status)
            print(f"\n📈 Overall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps*100:.1f}%)")
            
            print(f"✅ Completed iteration {i+1} for tissue '{tissue}'")
            
        except Exception as e:
            print(f"❌ Error processing iteration {i+1} for tissue '{tissue}': {str(e)}")
            import traceback
            print("Traceback:")
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Completed all iterations for tissue '{tissue}' with {len(all_results)} successful results")
    return all_results



def run_simple_pipeline_robust_uce(dataset1_path, dataset2_path,
                               enable_pavlin=False,
                               enable_integrated=False,
                               analysis_mode='standard',
                               enable_uce=True, 
                               uce_config=None, 
                               batch_key="source"):
                            
    """
    Run the enhanced simple pipeline with comprehensive error handling for each step
    
    Parameters:
    -----------
    dataset1_path : str
        Path to first dataset
    dataset2_path : str
        Path to second dataset
    enable_uce : bool
        Whether to enable UCE processing
    uce_config : dict
        UCE configuration parameters
    batch_key : str
        Batch key for harmony correction
        
    Returns:
    --------
    tuple: (results_dict, df_query, df_reference, metadata)
        - results_dict: Dictionary containing all method results
        - df_query: DataFrame with query results
        - df_reference: DataFrame with reference results  
        - metadata: Dictionary with pipeline metadata
    """
    print("using separate uce embedding")
    print("🚀 Starting Enhanced Simple Pipeline with Robust Error Handling...")
    print(f"Reference dataset: {dataset1_path}")
    print(f"Query dataset: {dataset2_path}")
    
    # Initialize variables to track success/failure of each step
    pipeline_status = {
        'file_check': False,
        'uce_processing': False,
        'uce_knn':False,
        'preprocessing': False,
        'embedding': False,
        'harmony': False,
        'knn_analysis': False,
        'second_pipeline': False,
        'results_integration': False,
        'table_creation': False
    }
    
    # Initialize return variables
    results_dict = {}
    df_query = None
    df_reference = None
    metadata = {}
    
    # =================================================================
    # STEP -1: FILE EXISTENCE CHECK
    # =================================================================
    print("\n" + "="*60)
    print("STEP -1: FILE EXISTENCE CHECK")
    print("="*60)
    
    try:
        if not os.path.exists(dataset1_path):
            raise FileNotFoundError(f"Reference file not found: {dataset1_path}")
        
        if not os.path.exists(dataset2_path):
            raise FileNotFoundError(f"Query file not found: {dataset2_path}")
        
        print("✅ Both files exist and are accessible")
        pipeline_status['file_check'] = True
        
    except Exception as e:
        print(f"❌ File check failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None, None, None, None

    # Initialize paths
    file1_path = dataset1_path
    file2_path = dataset2_path
    
    # Default UCE configuration
    if uce_config is None:
        uce_config = {
            'uce_dir': "/shared/home/christy.jo.manthara/batch-effect-analysis/UCE",
            'output_dir': "../output/",
            'model_loc': "/shared/home/christy.jo.manthara/batch-effect-analysis/UCE/model_files/33l_8ep_1024t_1280.torch",
            'nlayers': 33
        }

    # =================================================================
    # STEP 0: UCE EMBEDDINGS PROCESSING
    # =================================================================
    if enable_uce:
        print("\n" + "="*60)
        print("STEP 0: UCE EMBEDDINGS PROCESSING")
        print("="*60)
        
        try:
            print("🔄 Starting UCE processing...")
            uce_processed_files = process_datasets_with_uce(
                dataset_paths=[dataset1_path, dataset2_path],
                **uce_config
            )
            
            if len(uce_processed_files) != 2:
                print("⚠️  Could not process both datasets with UCE")
                raise Exception("UCE processing returned incomplete results")
            else:
                print(f"✅ UCE processing successful:")
                for i, file_path in enumerate(uce_processed_files, 1):
                    print(f"     File {i}: {file_path}")
                file1_path = uce_processed_files[0]
                file2_path = uce_processed_files[1]
                pipeline_status['uce_processing'] = True
                
        except Exception as e:
            print(f"❌ UCE processing failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print("⚠️  Continuing with original files...")
            file1_path = dataset1_path
            file2_path = dataset2_path
            pipeline_status['uce_processing'] = False
    else:
        print("\nℹ️  UCE processing disabled, using original files...")
        pipeline_status['uce_processing'] = True  # Mark as successful (skipped)
        
    # =================================================================
    # STEP 0.1: KNN AND t-SNE ANALYSIS FOR UCE
    # =================================================================
    print("\n" + "="*60)
    print("STEP 0.1: KNN AND t-SNE ANALYSIS for UCE")
    print("="*60)
    
    print("Adding the source field")
    def extract_filename(path):
        filename = os.path.basename(path)  # Get file name
        return filename.rsplit('.h5ad', 1)[0]  # Remove the extension

    # Store original source names for later splitting
    source1_name = extract_filename(file1_path).removesuffix('_uce_adata') 
    source2_name = extract_filename(file2_path).removesuffix('_uce_adata') 
    
    
    file1 = anndata.read_h5ad(file1_path)
    file2 = anndata.read_h5ad(file2_path)

    # Add source labels (simplified logic since both branches are identical)
    file1.obs["source"] = pd.Categorical([source1_name] * file1.n_obs)
    file2.obs["source"] = pd.Categorical([source2_name] * file2.n_obs)

    # Debug print to check source labels
    print("Unique source labels in adata:", file1.obs["source"].unique())
    print("Unique source labels in new:", file2.obs["source"].unique())
    
    
    results0 = {}
    metadata0 = {}
    
    
    try:
        print("🔄 Running KNN and t-SNE analysis on UCE output...")
        # results0, metadata0 = compute_knn_tsne_simple_with_transformation_plots(file1_path,reference_file=file2_path)
        results0, metadata0 = compute_knn_tsne_simple_with_transformation_plots(file1,reference_file=file2)
        
        if not results0:
            raise Exception("KNN analysis on uce returned empty results")
        
        print(f"✅ KNN and t-SNE analysis on uce successful!")
        print(f"   - Number of methods analyzed: {len(results0)}")
        print(f"   - Methods: {list(results0.keys())}")
        
        pipeline_status['uce_knn'] = True
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ KNN and t-SNE analysis failed for UCE: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("⚠️  Continuing with empty results for zeroth pipeline on UCE")
        results0 = {}
        metadata0 = {}
        pipeline_status['uce_knn'] = False

    # =================================================================
    # STEP 1: LOADING AND PREPROCESSING DATASETS
    # =================================================================
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PREPROCESSING DATASETS")
    print("="*60)
    
    # instead of the uce filepath, we use the original dataset now.
    file1_path = dataset1_path
    file2_path = dataset2_path
    
    
    combined_data = None
    try:
        print("🔄 Loading and preprocessing datasets...")
        print(f"   - File 1: {file1_path}")
        print(f"   - File 2: {file2_path}")
        
        combined_data = preprocessing_metadata.load_and_preprocess_multi_embedder(
            file1=file1_path,
            file2=file2_path,
            save=False,
            split_output=False
        )
        
        if combined_data is None:
            raise Exception("Preprocessing returned None")
        
        print(f"✅ Preprocessing successful!")
        print(f"   - Combined data shape: {combined_data.shape}")
        print(f"   - Available layers: {list(combined_data.layers.keys()) if hasattr(combined_data, 'layers') else 'None'}")
        print(f"   - Available obsm: {list(combined_data.obsm.keys()) if hasattr(combined_data, 'obsm') else 'None'}")
        print(f"   - Available uns keys: {list(combined_data.uns.keys()) if hasattr(combined_data, 'uns') else 'None'}")
        
        pipeline_status['preprocessing'] = True
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("🛑 Cannot continue without preprocessed data")
        return None, None, None, None

    # =================================================================
    # STEP 2: ADDING ADDITIONAL EMBEDDINGS
    # =================================================================
    print("\n" + "="*60)
    print("STEP 2: ADDING ADDITIONAL EMBEDDINGS (scVI, scANVI, UCE)")
    print("="*60)
    
    embedded_adata = combined_data  # Default fallback
    try:
        print("🔄 Adding embeddings...")
        embedded_adata = ce(combined_data)
        
        if embedded_adata is None:
            raise Exception("Embedding function returned None")
        
        print(f"✅ Embedding successful!")
        print(f"   - Embedded data shape: {embedded_adata.shape}")
        print(f"   - Available obsm after embedding: {list(embedded_adata.obsm.keys()) if hasattr(embedded_adata, 'obsm') else 'None'}")
        print(f"   - Available uns keys after embedding: {list(embedded_adata.uns.keys()) if hasattr(embedded_adata, 'uns') else 'None'}")
        print(f"📦 Metadata preserved in .obs columns:")
        metadata_cols = [col for col in embedded_adata.obs.columns if col.startswith('dataset_')]
        for col in metadata_cols:
            print(f"  {col}: {embedded_adata.obs[col].unique()}")
        
        pipeline_status['embedding'] = True
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ Embedding failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("⚠️  Continuing with preprocessed data (no additional embeddings)")
        embedded_adata = combined_data
        pipeline_status['embedding'] = False

    # =================================================================
    # STEP 2.5: HARMONY CORRECTION
    # =================================================================
    print("\n" + "="*60)
    print("STEP 2.5: HARMONY CORRECTION")
    print("="*60)
    
    adata_harmony = embedded_adata  # Default fallback
    try:
        print(f"🔄 Running harmony correction with batch_key='{batch_key}'...")
        
        # Check if batch_key exists in obs
        if batch_key not in embedded_adata.obs.columns:
            print(f"⚠️  Batch key '{batch_key}' not found in obs. Available columns:")
            print(f"   {list(embedded_adata.obs.columns)}")
            raise Exception(f"Batch key '{batch_key}' not found")
        
        # adata_harmony = run_harmony_correction_simple(embedded_adata, batch_key=batch_key)
        adata_harmony = run_harmony_correction_with_original_x(embedded_adata, batch_key=batch_key)
        
        if adata_harmony is None:
            raise Exception("Harmony correction returned None")
        
        print(f"📦 Metadata preserved in .obs columns after harmony:")
        metadata_cols = [col for col in adata_harmony.obs.columns if col.startswith('dataset_')]
        for col in metadata_cols:
            print(f"  {col}: {adata_harmony.obs[col].unique()}")
        
        print(f"✅ Harmony correction successful!")
        print(f"   - Harmony data shape: {adata_harmony.shape}")
        print(f"   - Available obsm after harmony: {list(adata_harmony.obsm.keys()) if hasattr(adata_harmony, 'obsm') else 'None'}")
        print(f"   - Available uns keys after harmony: {list(adata_harmony.uns.keys()) if hasattr(adata_harmony, 'uns') else 'None'}")
        
        pipeline_status['harmony'] = True
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ Harmony correction failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("⚠️  Continuing with embedded data (no harmony correction)")
        adata_harmony = embedded_adata
        pipeline_status['harmony'] = False

    # =================================================================
    # STEP 3: KNN AND t-SNE ANALYSIS
    # =================================================================
    print("\n" + "="*60)
    print("STEP 3: KNN AND t-SNE ANALYSIS")
    print("="*60)
    
    results1 = {}
    metadata1 = {}
    try:
        print("🔄 Running KNN and t-SNE analysis...")
        results1, metadata1 = compute_knn_tsne_simple_with_transformation_plots(adata_harmony)
        
        print("\n📊 Metadata from analysis:")
        for key, value in metadata1.items():
            print(f"   - {key}: {value}")
            
        if not metadata1:
            raise ValueError("Metadata from KNN and t-SNE analysis is empty")
        
        if not results1:
            raise Exception("KNN analysis returned empty results")
        
        print(f"✅ KNN and t-SNE analysis successful!")
        print(f"   - Number of methods analyzed: {len(results1)}")
        print(f"   - Methods: {list(results1.keys())}")
        
        pipeline_status['knn_analysis'] = True
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ KNN and t-SNE analysis failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("⚠️  Continuing with empty results for first pipeline")
        results1 = {}
        metadata1 = {}
        pipeline_status['knn_analysis'] = False

    # =================================================================
    # STEP 4: SECOND PIPELINE (PAVLIN'S)
    # =================================================================
    print("\n" + "="*60)
    print("STEP 4: RUNNING SECOND PIPELINE (PAVLIN'S)")
    print("="*60)
    
    results2 = {}
    try:
        print("🔄 Processing reference data...")
        transformed_ref_adata = process_single_anndata(dataset1_path)
        
        if transformed_ref_adata is None:
            raise Exception("Reference data processing returned None")
        
        print("✅ Reference data processed successfully")
        
        print("🔄 Transforming new data and computing results...")
        transformed_data, results2 = transform_tsne_single(transformed_ref_adata, dataset2_path)
        
        if not results2:
            raise Exception("Second pipeline returned empty results")
        
        print(f"✅ Second pipeline successful!")
        print(f"   - Number of methods: {len(results2)}")
        print(f"   - Methods: {list(results2.keys())}")
        
        pipeline_status['second_pipeline'] = True
        
    except Exception as e:
        print(f"❌ Second pipeline failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("⚠️  Continuing without second pipeline results")
        results2 = {}
        pipeline_status['second_pipeline'] = False

    # =================================================================
    # STEP 5: INTEGRATING RESULTS
    # =================================================================
    # Replace the existing STEP 5: INTEGRATING RESULTS section with this:
    # =================================================================
    # STEP 5: INTEGRATING RESULTS
    # =================================================================
    print("\n" + "="*60)
    print("STEP 5: INTEGRATING RESULTS")
    print("="*60)

    integrated_results = {}
    try:
        print("🔄 Merging results from all pipelines...")
        
        # Collect available results with their names
        available_results = []
        pipeline_names = []
        
        if results0:
            available_results.append(results0)
            pipeline_names.append("uce")
            print(f"   - UCE pipeline: {len(results0)} methods")
        
        if results1:
            available_results.append(results1)
            pipeline_names.append("main")
            print(f"   - Main pipeline: {len(results1)} methods")
        
        if results2:
            available_results.append(results2)
            pipeline_names.append("pavlin")
            print(f"   - Pavlin pipeline: {len(results2)} methods")
        
        if len(available_results) >= 1:
            # Use the enhanced merge function
            integrated_results = merge_multiple_results_tables(available_results, pipeline_names)
            
            print(f"✅ Results merged successfully!")
            print(f"   - Total pipelines: {len(available_results)}")
            print(f"   - Total integrated methods: {len(integrated_results)}")
            
        else:
            raise Exception("No results available from any pipeline")
        
        pipeline_status['results_integration'] = True
        
    except Exception as e:
        print(f"❌ Results integration failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("⚠️  Using whatever results are available")
        # Fallback logic
        if results0:
            integrated_results = results0
        elif results1:
            integrated_results = results1
        elif results2:
            integrated_results = results2
        pipeline_status['results_integration'] = False

    # =================================================================
    # STEP 6: CREATING RESULTS TABLES
    # =================================================================
    print("\n" + "="*60)
    print("STEP 6: CREATING RESULTS TABLES")
    print("="*60)
    
    try:
        if not integrated_results:
            raise Exception("No integrated results available for table creation")
        
        print("🔄 Creating comprehensive results tables...")
        
        # Extract source information
        source1 = metadata1.get('ref_source', 'ref') if metadata1 else 'ref'
        source2 = metadata1.get('query_source', 'query') if metadata1 else 'query'
        base_filename = f"integrated_{source1}_{source2}"
        
        print(f"   - Source 1: {source1}")
        print(f"   - Source 2: {source2}")
        print(f"   - Base filename: {base_filename}")
        
        
        
        print("metadata1 passed to step 6:create results table")
        print(" | ".join([f"{k}: {v}" for k, v in metadata1.items()]))
        
        
        
        df_query, df_reference = create_results_table(
            integrated_results,
            source1, 
            source2, 
            base_filename, 
            metadata=metadata1
        )
        
        print(f"✅ Results tables created successfully!")
        if df_query is not None:
            print(f"   - Query results shape: {df_query.shape}")
        if df_reference is not None:
            print(f"   - Reference results shape: {df_reference.shape}")
        
        pipeline_status['table_creation'] = True
        
    except Exception as e:
        print(f"❌ Results table creation failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("⚠️  Returning raw results without formatted tables")
        df_query = None
        df_reference = None
        pipeline_status['table_creation'] = False

    # =================================================================
    # FINAL PIPELINE SUMMARY
    # =================================================================
    print("\n" + "="*60)
    print("🎉 PIPELINE EXECUTION COMPLETED!")
    print("="*60)
    
    print("📊 Pipeline Status Summary:")
    for step, status in pipeline_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {step.replace('_', ' ').title()}: {'Success' if status else 'Failed'}")
    
    successful_steps = sum(pipeline_status.values())
    total_steps = len(pipeline_status)
    print(f"\n📈 Overall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps*100:.1f}%)")
    
    if integrated_results:
        print(f"\n📋 Final Results Summary:")
        print(f"   - Total methods analyzed: {len(integrated_results)}")
        
        # Show which pipelines contributed
        contributing_pipelines = []
        if results0:
            contributing_pipelines.append(f"UCE pipeline ({len(results0)} methods)")
        if results1:
            contributing_pipelines.append(f"First pipeline ({len(results1)} methods)")
        if results2:
            contributing_pipelines.append(f"Second pipeline ({len(results2)} methods)")
        
        print(f"   - Contributing pipelines: {', '.join(contributing_pipelines)}")
        print(f"   - Methods included:")
        for method in integrated_results.keys():
            print(f"     • {method}")
        
        print(f"\n💾 Output Files:")
        print(f"   - Raw results: Available in memory")
        if df_query is not None:
            print(f"   - Query results CSV: Generated")
        if df_reference is not None:
            print(f"   - Reference results CSV: Generated")
        print(f"   - Enhanced visualizations: Generated")
    else:
        print(f"\n⚠️  No final results available - check individual step failures above")
    
    # Final memory cleanup
    import gc
    gc.collect()
    
    return integrated_results, df_query, df_reference, metadata1




def run_simple_pipeline_with_monitoring_uce(dataset1_path, dataset2_path, **kwargs):
    """
    Wrapper function that adds memory monitoring to the robust pipeline
    """
    
    print("🚀 Starting Pipeline with Memory Monitoring...")
    monitor_memory_usage("at start")
    
    try:
        # Run the robust pipeline
        results = run_simple_pipeline_robust_uce(dataset1_path, dataset2_path, **kwargs)
        
        monitor_memory_usage("at completion")
        return results
        
    except Exception as e:
        monitor_memory_usage("at error")
        print(f"❌ Pipeline failed with monitoring: {e}")
        raise



def monitor_memory_usage(step_name=""):
    """Helper function to monitor memory usage at each step"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / 1024 / 1024 / 1024
        
        print(f"🔍 Memory Usage {step_name}:")
        print(f"   - Process Memory: {memory_gb:.2f} GB")
        print(f"   - Available System Memory: {available_gb:.2f} GB")
        print(f"   - Memory Usage: {system_memory.percent:.1f}%")
        
        if memory_gb > 32:  # Warning if using more than 32GB
            print(f"   ⚠️  High memory usage detected!")
        
    except ImportError:
        print(f"🔍 Memory monitoring not available (psutil not installed)")
    except Exception as e:
        print(f"🔍 Memory monitoring failed: {e}")



def main_robust_uce():
    """
    Main pipeline function with comprehensive error handling for each component
    """
    
    # Configuration
    ENABLE_UCE = True  # Set to False to skip UCE processing
    PROCESSING_MODE = "tissue"  # Options: "tissue", "simple"
    ENABLE_MEMORY_MONITORING = True  # Enable detailed memory monitoring
    
    # UCE Configuration
    UCE_CONFIG = {
        'uce_dir': "/shared/home/christy.jo.manthara/batch-effect-analysis/UCE",
        'output_dir': "../output/",
        'model_loc': "/shared/home/christy.jo.manthara/batch-effect-analysis/UCE/model_files/33l_8ep_1024t_1280.torch",
        'nlayers': 33
    }
    
    # =================================================================
    # INITIALIZATION AND ENVIRONMENT CHECK
    # =================================================================
    print("🚀 Starting Robust Pipeline Analysis")
    print("="*60)
    
    try:
        # Check Python environment
        import sys
        print(f"Python version: {sys.version}")
        
        # Check key libraries
        import scanpy as sc
        import anndata
        import pandas as pd
        import numpy as np
        
        print(f"Scanpy version: {sc.__version__}")
        print(f"Anndata version: {anndata.__version__}")
        print(f"Pandas version: {pd.__version__}")
        print(f"Numpy version: {np.__version__}")
        
        # Memory check
        if ENABLE_MEMORY_MONITORING:
            monitor_memory_usage("initial environment check")
        
    except Exception as e:
        print(f"❌ Environment check failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

    if PROCESSING_MODE == "simple":
        # =================================================================
        # SIMPLE MODE: PROCESS TWO SPECIFIC DATASETS
        # =================================================================
        
        # dataset1_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Prostate1_dge_txt_Prostate1_dge.h5ad"
        # dataset2_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Prostate2_dge_txt_Prostate2_dge.h5ad"
        
        #working ovary
        # dataset1_path = "/shared/home/christy.jo.manthara/Datasets/Ovary1_dge_txt_Ovary1_dge.h5ad"
        # dataset2_path = "/shared/home/christy.jo.manthara/Datasets/Ovary2_dge_txt_Ovary2_dge.h5ad"
        # dataset1_path = "/shared/home/ch
        # o.manthara/Datasets/MCA500/mca_output/individual_tissues/Ovary2_dge_txt_Ovary2_dge.h5ad"
        
        # dataset1_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Uterus1_dge_txt_Uterus1_dge.h5ad"
        # dataset2_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Uterus2_dge_txt_Uterus2_dge.h5ad"
       
        # this also works
        # dataset1_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Ovary1_dge_txt_Ovary1_dge.h5ad"
        # dataset2_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Ovary2_dge_txt_Ovary2_dge.h5ad"
        
        # dataset1_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Lung1_dge_txt_Lung1_dge.h5ad"
        # dataset2_path = "/shared/home/christy.jo.manthara/Datasets/MCA500/mca_output/individual_tissues/Lung2_dge_txt_Lung2_dge.h5ad"
        
        dataset1_path = "/shared/home/christy.jo.manthara/Datasets/Uterus1.h5ad"
        dataset2_path = "/shared/home/christy.jo.manthara/Datasets/Uterus2.h5ad"
        
        # dataset1_path = "/shared/home/christy.jo.manthara/Datasets/baron_2016h.h5ad"
        # dataset2_path = "/shared/home/christy.jo.manthara/Datasets/xin_2016.h5ad"
        
        print(f"\n📁 Dataset Configuration:")
        print(f"   - Dataset 1 (Reference): {dataset1_path}")
        print(f"   - Dataset 2 (Query): {dataset2_path}")
        print(f"   - UCE Enabled: {ENABLE_UCE}")
        print(f"   - Memory Monitoring: {ENABLE_MEMORY_MONITORING}")
        
        # File existence check with detailed info
        try:
            print(f"\n🔍 Pre-flight File Check:")
            
            # Check dataset 1
            if not os.path.exists(dataset1_path):
                raise FileNotFoundError(f"Dataset 1 not found: {dataset1_path}")
            
            file1_size = os.path.getsize(dataset1_path) / (1024**2)  # MB
            print(f"   ✅ Dataset 1: {file1_size:.1f} MB")
            
            # Check dataset 2
            if not os.path.exists(dataset2_path):
                raise FileNotFoundError(f"Dataset 2 not found: {dataset2_path}")
            
            file2_size = os.path.getsize(dataset2_path) / (1024**2)  # MB
            print(f"   ✅ Dataset 2: {file2_size:.1f} MB")
            
            print(f"   📊 Total data size: {file1_size + file2_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ File check failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return False

        # =================================================================
        # RUN THE ROBUST PIPELINE
        # =================================================================
        try:
            print(f"\n🎯 Launching Robust Pipeline...")
            
            if ENABLE_MEMORY_MONITORING:
                # Use the version with memory monitoring
                results_dict, df_query, df_reference, metadata = run_simple_pipeline_with_monitoring_uce(
                    dataset1_path=dataset1_path,
                    dataset2_path=dataset2_path,
                    enable_uce=ENABLE_UCE,
                    uce_config=UCE_CONFIG,
                    batch_key="source"
                )
            else:
                # Use the standard robust version
                results_dict, df_query, df_reference, metadata = run_simple_pipeline_robust_uce(
                    dataset1_path=dataset1_path,
                    dataset2_path=dataset2_path,
                    enable_uce=ENABLE_UCE,
                    uce_config=UCE_CONFIG,
                    batch_key="source"
                )
            
            # =================================================================
            # ANALYZE PIPELINE RESULTS
            # =================================================================
            print(f"\n📊 Final Results Analysis:")
            
            if results_dict is not None and len(results_dict) > 0:
                print(f"✅ Pipeline completed with results!")
                print(f"   - Total methods analyzed: {len(results_dict)}")
                print(f"   - Methods included:")
                for method in results_dict.keys():
                    print(f"     • {method}")
                
                # Check data quality
                if df_query is not None:
                    print(f"   - Query results: {df_query.shape[0]} rows, {df_query.shape[1]} columns")
                else:
                    print(f"   - Query results: Not generated")
                
                if df_reference is not None:
                    print(f"   - Reference results: {df_reference.shape[0]} rows, {df_reference.shape[1]} columns")
                else:
                    print(f"   - Reference results: Not generated")
                
                # Check metadata
                if metadata and len(metadata) > 0:
                    print(f"   - Metadata entries: {len(metadata)}")
                    print(f"   - Metadata keys: {list(metadata.keys())}")
                else:
                    print(f"   - Metadata: Not available")
                
                print(f"\n💾 Output Files Generated:")
                print(f"   - Raw results: Available in memory")
                print(f"   - Pickle files: Check knn_results_accumulated.pkl")
                print(f"   - CSV files: Check output directory")
                print(f"   - Excel files: Check output directory")
                print(f"   - Visualization plots: Check output directory")
                
                return True
                
            else:
                print(f"⚠️  Pipeline completed but no results generated")
                print(f"   - This could indicate issues in all analysis steps")
                print(f"   - Check the individual step failures above")
                return False
                
        except KeyboardInterrupt:
            print(f"\n🛑 Pipeline interrupted by user (Ctrl+C)")
            print(f"   - This is normal if you want to stop the analysis")
            return False
            
        except MemoryError:
            print(f"\n💾 Pipeline failed due to insufficient memory")
            print(f"   - Try reducing dataset size or increasing available memory")
            print(f"   - Consider running with more memory: srun --mem=64G")
            return False
            
        except Exception as e:
            print(f"\n❌ ROBUST PIPELINE FAILED!")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            
            # Enhanced error reporting
            import traceback
            print(f"\nFull traceback:")
            traceback.print_exc()
            
            # System information for debugging
            try:
                import platform
                print(f"\nSystem Information:")
                print(f"   - Platform: {platform.platform()}")
                print(f"   - Python: {platform.python_version()}")
                print(f"   - Architecture: {platform.architecture()}")
            except:
                pass
            
            return False
        
    elif PROCESSING_MODE == "tissue":
        # Tissue mode: Process all datasets grouped by tissue
        
        print("🚀 Starting Tissue-Based Pipeline Analysis")
        
        try:
            # Analyze datasets by tissue
            tissue_groups = analyze_datasets_by_tissue("../Datasets/")
            
            if not tissue_groups:
                print("❌ No tissue groups found. Exiting.")
                return
            
            print(f"\n📊 Found tissues: {list(tissue_groups.keys())}")
            
            # Show summary of what will be processed
            total_iterations = 0
            for tissue, file_info_list in tissue_groups.items():
                print(f"\n{tissue.upper()} tissue group:")
                print(f"  - {len(file_info_list)} datasets found")
                if len(file_info_list) >= 2:
                    iterations = len(file_info_list) - 1
                    total_iterations += iterations
                    print(f"  - Will run {iterations} iterations")
                else:
                    print(f"  - ⚠️  Skipped (need at least 2 datasets)")
                for filepath, cell_count in file_info_list:
                    print(f"    • {Path(filepath).name}: {cell_count} cells")
            
            print(f"\n📈 Total iterations planned: {total_iterations}")
            
            # Process each tissue group with enhanced error handling
            successful_tissues = []
            failed_tissues = []
            
            for tissue, file_info_list in tissue_groups.items():
                print(f"\n" + "="*60)
                print(f"PROCESSING {tissue.upper()} TISSUE GROUP")
                print("="*60)
                
                try:
                    # Process tissue group and capture results if your function returns them
                    result = process_tissue_group(
                    # result = process_tissue_group_int(
                        tissue=tissue, 
                        file_info_list=file_info_list,
                        enable_uce=ENABLE_UCE,
                        uce_config=UCE_CONFIG,
                        batch_key='batch'
                    )
                    
                    successful_tissues.append(tissue)
                    print(f"✅ {tissue.upper()} tissue group completed successfully!")
                    
                    # If process_tissue_group returns results, you can handle them here
                    if result is not None:
                        print(f"   - Results generated for {tissue} tissue")
                    
                except Exception as e:
                    failed_tissues.append((tissue, str(e)))
                    print(f"❌ {tissue.upper()} tissue group failed!")
                    print(f"   Error: {str(e)}")
                    print(f"   Error type: {type(e).__name__}")
                    
                    # Print traceback for debugging but continue with other tissues
                    import traceback
                    print(f"   Traceback for {tissue}:")
                    traceback.print_exc()
                    print("   Continuing with next tissue group...")
            
            # Final summary
            print("\n" + "="*60)
            print("🎉 TISSUE-BASED PIPELINE ANALYSIS COMPLETE!")
            print("="*60)
            
            print(f"📊 Processing Summary:")
            print(f"   - Total tissue groups found: {len(tissue_groups)}")
            print(f"   - Successfully processed: {len(successful_tissues)}")
            print(f"   - Failed: {len(failed_tissues)}")
            
            if successful_tissues:
                print(f"\n✅ Successfully processed tissues:")
                for tissue in successful_tissues:
                    print(f"   • {tissue}")
            
            if failed_tissues:
                print(f"\n❌ Failed tissues:")
                for tissue, error in failed_tissues:
                    print(f"   • {tissue}: {error}")
            
            print(f"\n📁 Results have been saved to respective tissue directories")
            print(f"   - CSV files with comprehensive results")
            print(f"   - Excel files with metadata")
            print(f"   - Enhanced visualizations")
            print(f"   - Accumulated results files")
            
        except Exception as e:
            print(f"\n❌ TISSUE-BASED PIPELINE FAILED!")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            
            # Print traceback for debugging
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()

    else:
        print(f"Unknown processing mode: {PROCESSING_MODE}")
        print("Available modes: 'simple', 'tissue'")



if __name__ == "__main__":
    # main_robust()
    main_robust_uce()