import pandas as pd
import numpy as np

def transform_to_baron_format(adata, dataset_name="unknown"):
    """
    Transform AnnData object to match Baron 2016 dataset format
    
    Parameters:
    adata: AnnData object to transform
    dataset_name: Name of the dataset (for batch_id generation)
    
    Returns:
    adata: Transformed AnnData object
    """
    
    # 1. Rename cell_type1 column to labels
    if 'cell_type1' in adata.obs.columns:
        adata.obs['labels'] = adata.obs['cell_type1'].copy()
        adata.obs.drop('cell_type1', axis=1, inplace=True)
    
    # 2. Transform cell type names to Baron format (title case with "cells" suffix)
    cell_type_mapping = {
        'acinar': 'Acinar cells',
        'alpha': 'Alpha cells', 
        'beta': 'Beta cells',
        'delta': 'Delta cells',
        'ductal': 'Ductal cells',
        'endothelial': 'Endothelial cell',  # Note: singular in Baron
        'gamma': 'PP cells',  # PP cells are gamma cells
        'epsilon': 'Other',   # Epsilon cells are rare, map to Other
        'mesenchymal': 'PaSC',  # Pancreatic stellate cells
        'stellate': 'PaSC',   # Also pancreatic stellate cells
        'MHC class II': 'Other',  # Immune cells -> Other
        'mast': 'Other'       # Mast cells -> Other
    }
    
    # Apply mapping
    if 'labels' in adata.obs.columns:
        adata.obs['labels'] = adata.obs['labels'].map(cell_type_mapping).fillna('Other')
    
    # 3. Add batch_id column (create artificial batches if needed)
    if 'batch_id' not in adata.obs.columns:
        # Create batch IDs based on dataset name
        # For simplicity, create batches based on cell count
        n_cells = adata.n_obs
        batch_size = max(500, n_cells // 4)  # Aim for 4 batches or min 500 cells per batch
        
        batch_ids = []
        batch_counter = 1
        for i in range(n_cells):
            batch_id = f"{dataset_name}_lib{((i // batch_size) % 4) + 1}"
            batch_ids.append(batch_id)
        
        adata.obs['batch_id'] = batch_ids
    
    # 4. Fix gene naming - move feature_symbol to var.index if it exists
    if 'feature_symbol' in adata.var.columns:
        # Use feature_symbol as the main gene identifier
        adata.var.index = adata.var['feature_symbol'].astype(str)
        # Remove the feature_symbol column to match Baron format
        adata.var.drop('feature_symbol', axis=1, inplace=True)
    
    # 5. Add uns metadata to match Baron format
    adata.uns['name'] = dataset_name
    adata.uns['organism'] = 'human'  # Assuming human data
    adata.uns['tissue'] = 'pancreas'
    
    # Determine year based on dataset name (you may want to adjust these)
    year_mapping = {
        'muraro': 2016,
        'segerstolpe': 2016, 
        'wang': 2016
    }
    adata.uns['year'] = year_mapping.get(dataset_name.lower(), 2016)
    
    print(f"   📦 Created uns metadata:")
    print(f"      🔑 name: '{adata.uns['name']}'")
    print(f"      🔑 organism: '{adata.uns['organism']}'") 
    print(f"      🔑 tissue: '{adata.uns['tissue']}'")
    print(f"      🔑 year: {adata.uns['year']}")
    
    # 6. Ensure obs columns are in the same order as Baron
    desired_order = ['batch_id', 'labels']
    existing_cols = [col for col in desired_order if col in adata.obs.columns]
    other_cols = [col for col in adata.obs.columns if col not in desired_order]
    adata.obs = adata.obs[existing_cols + other_cols]
    
    print(f"✅ Transformed {dataset_name} dataset:")
    print(f"   Shape: {adata.shape}")
    print(f"   Cell types: {adata.obs['labels'].value_counts().to_dict()}")
    print(f"   Batches: {adata.obs['batch_id'].value_counts().to_dict()}")
    
    return adata

def transform_and_save_dataset(input_file, dataset_name):
    """
    Load, transform, and save a dataset to Baron 2016 format
    
    Parameters:
    input_file: Path to input .h5ad file
    dataset_name: Name of the dataset (for batch_id generation)
    """
    import scanpy as sc
    
    # Load the dataset
    print(f"Loading {input_file}...")
    adata = sc.read_h5ad(input_file)
    
    # Transform to Baron format
    adata_transformed = transform_to_baron_format(adata, dataset_name)
    
    # Generate output filename
    base_name = input_file.replace('.h5ad', '').replace('.adata', '')
    output_file = f"{base_name}_transformed.h5ad"
    
    # Save the transformed dataset
    print(f"Saving to {output_file}...")
    adata_transformed.write(output_file)
    
    print(f"✅ Successfully saved {output_file}")
    return adata_transformed

# Batch processing function
def process_all_datasets(file_mapping):
    """
    Process multiple datasets at once
    
    Parameters:
    file_mapping: dict mapping dataset_name -> input_file_path
    
    Example:
    file_mapping = {
        'muraro': 'muraro.h5ad',
        'segerstolpe': 'segerstolpe.h5ad', 
        'wang': 'wang.h5ad'
    }
    """
    transformed_datasets = {}
    
    for dataset_name, input_file in file_mapping.items():
        try:
            adata_transformed = transform_and_save_dataset(input_file, dataset_name)
            transformed_datasets[dataset_name] = adata_transformed
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {str(e)}")
    
    return transformed_datasets

# Example usage for each dataset with saving:

# For Muraro dataset
# adata_muraro_transformed = transform_to_baron_format(adata_muraro, "muraro")
# adata_muraro_transformed.write("muraro_transformed.h5ad")

# For Segerstolpe dataset  
# adata_segerstolpe_transformed = transform_to_baron_format(adata_segerstolpe, "segerstolpe")
# adata_segerstolpe_transformed.write("segerstolpe_transformed.h5ad")

# For Wang dataset
# adata_wang_transformed = transform_to_baron_format(adata_wang, "wang")
# adata_wang_transformed.write("wang_transformed.h5ad")

# Alternative: More sophisticated batch assignment based on existing metadata
def create_smart_batches(adata, dataset_name, n_batches=4):
    """
    Create batch IDs more intelligently based on cell types or random assignment
    """
    n_cells = adata.n_obs
    
    # Option 1: Random assignment
    np.random.seed(42)  # For reproducibility
    batch_assignments = np.random.choice(range(1, n_batches + 1), size=n_cells)
    batch_ids = [f"{dataset_name}_lib{batch}" for batch in batch_assignments]
    
    return batch_ids

# If you want more control over batch assignment:
def transform_with_custom_batches(adata, dataset_name, batch_strategy="random"):
    """Enhanced transformation with custom batch strategies"""
    
    # Apply basic transformation first
    adata = transform_to_baron_format(adata, dataset_name)
    
    if batch_strategy == "random":
        adata.obs['batch_id'] = create_smart_batches(adata, dataset_name)
    elif batch_strategy == "by_celltype":
        # Create batches within each cell type
        batch_ids = []
        for cell_type in adata.obs['labels'].unique():
            mask = adata.obs['labels'] == cell_type
            n_cells_type = mask.sum()
            type_batches = np.random.choice(range(1, 5), size=n_cells_type)
            type_batch_ids = [f"{dataset_name}_lib{batch}" for batch in type_batches]
            
            # Assign to the right positions
            full_batch_assignment = [''] * len(adata.obs)
            positions = np.where(mask)[0]
            for i, pos in enumerate(positions):
                full_batch_assignment[pos] = type_batch_ids[i]
            batch_ids.extend([bid for bid in full_batch_assignment if bid])
        
        adata.obs['batch_id'] = batch_ids
    
    return adata