import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DebugMCADataProcessor:
    def __init__(self, csv_file_path):
        """
        Debug version to identify cell matching issues
        """
        self.csv_file_path = csv_file_path

    def load_and_examine_metadata(self):
        """Load metadata and examine its structure"""
        logger.info("Loading and examining metadata...")
        
        # Load metadata
        metadata = pd.read_csv(self.csv_file_path, index_col=0)
        
        print("\n" + "="*50)
        print("METADATA ANALYSIS")
        print("="*50)
        print(f"Shape: {metadata.shape}")
        print(f"Columns: {list(metadata.columns)}")
        
        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(metadata.head())
        
        # Analyze Cell.name format
        print(f"\nCell.name format analysis:")
        cell_names = metadata['Cell.name'].head(10)
        for i, name in enumerate(cell_names):
            print(f"  {i+1}: '{name}' (length: {len(name)})")
        
        # Check for unique tissues and batches
        print(f"\nUnique tissues: {metadata['Tissue'].unique()}")
        print(f"Unique batches: {metadata['Batch'].unique()[:10]}...")  # Show first 10
        
        return metadata

    def examine_dge_file(self, dge_file_path):
        """Examine a DGE file structure"""
        logger.info(f"Examining DGE file: {dge_file_path}")
        
        print("\n" + "="*50)
        print(f"DGE FILE ANALYSIS: {Path(dge_file_path).name}")
        print("="*50)
        
        # Try different separators
        separators = ['\t', ',', ' ', None]
        
        for sep in separators:
            try:
                print(f"\nTrying separator: {repr(sep)}")
                
                # Read just the first few rows and columns
                df_sample = pd.read_csv(dge_file_path, sep=sep, nrows=5, index_col=0)
                print(f"Shape: {df_sample.shape}")
                print(f"Columns (first 5): {list(df_sample.columns[:5])}")
                print(f"Index (first 5): {list(df_sample.index[:5])}")
                
                # Check if data looks numeric
                numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
                print(f"Numeric columns: {len(numeric_cols)}/{len(df_sample.columns)}")
                
                if len(numeric_cols) > 0:
                    print(f"Sample values:\n{df_sample.iloc[:3, :3]}")
                    
                    # This separator seems to work
                    print(f"✓ Separator {repr(sep)} looks good!")
                    return sep, df_sample.columns
                
            except Exception as e:
                print(f"✗ Separator {repr(sep)} failed: {e}")
        
        return None, None

    def load_full_dge_file(self, dge_file_path, separator='\t'):
        """Load the full DGE file with specified separator"""
        logger.info(f"Loading full DGE file with separator: {repr(separator)}")
        
        try:
            data = pd.read_csv(dge_file_path, sep=separator, index_col=0, low_memory=False)
            
            print(f"\nFull DGE file loaded:")
            print(f"Shape: {data.shape}")
            print(f"Sample cell names (columns):")
            for i, col in enumerate(data.columns[:5]):
                print(f"  {i+1}: '{col}'")
            
            print(f"\nSample gene names (index):")
            for i, gene in enumerate(data.index[:5]):
                print(f"  {i+1}: '{gene}'")
            
            # Check data types
            numeric_data = data.select_dtypes(include=[np.number])
            print(f"\nNumeric columns: {numeric_data.shape[1]}/{data.shape[1]}")
            
            if numeric_data.shape[1] > 0:
                print(f"Data range: {numeric_data.min().min()} to {numeric_data.max().max()}")
                print(f"Non-zero values: {(numeric_data > 0).sum().sum()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load DGE file: {e}")
            return None

    def test_cell_matching(self, metadata, dge_data):
        """Test different cell matching strategies"""
        print("\n" + "="*50)
        print("CELL MATCHING ANALYSIS")
        print("="*50)
        
        # Get cell names from both sources
        dge_cells = list(dge_data.columns)
        metadata_cells = list(metadata['Cell.name'])
        
        print(f"DGE cells count: {len(dge_cells)}")
        print(f"Metadata cells count: {len(metadata_cells)}")
        
        # Strategy 1: Direct matching
        direct_matches = set(dge_cells) & set(metadata_cells)
        print(f"\nDirect matches: {len(direct_matches)}")
        
        if len(direct_matches) > 0:
            print("✓ Direct matching works!")
            return 'direct'
        
        # Strategy 2: Show sample names for manual inspection
        print(f"\nSample DGE cell names:")
        for i, cell in enumerate(dge_cells[:10]):
            print(f"  DGE {i+1}: '{cell}'")
        
        print(f"\nSample metadata cell names:")
        for i, cell in enumerate(metadata_cells[:10]):
            print(f"  META {i+1}: '{cell}'")
        
        # Strategy 3: Check if DGE cells contain metadata barcodes
        # Extract barcodes from metadata
        metadata_barcodes = list(metadata['Cell.Barcode'])
        
        matches_found = 0
        for dge_cell in dge_cells[:100]:  # Check first 100
            for meta_barcode in metadata_barcodes[:100]:
                if meta_barcode in dge_cell:
                    matches_found += 1
                    break
        
        print(f"\nBarcode-in-cell matches (first 100x100): {matches_found}")
        
        # Strategy 4: Check if we need to construct the key differently
        # Try matching by tissue + barcode
        print(f"\nTrying tissue + barcode matching...")
        
        # Create tissue-barcode combinations from metadata
        tissue_barcode_combos = []
        for _, row in metadata.head(10).iterrows():
            combo1 = f"{row['Tissue']}.{row['Cell.Barcode']}"
            combo2 = f"{row['Tissue']}_{row['Batch']}.{row['Cell.Barcode']}"
            tissue_barcode_combos.extend([combo1, combo2])
        
        print(f"Sample tissue+barcode combinations:")
        for i, combo in enumerate(tissue_barcode_combos[:5]):
            print(f"  COMBO {i+1}: '{combo}'")
        
        # Check matches
        combo_matches = set(dge_cells) & set(tissue_barcode_combos)
        print(f"Tissue+barcode matches: {len(combo_matches)}")
        
        return None

    def debug_single_file(self, dge_file_path):
        """Complete debug analysis for a single DGE file"""
        print("\n" + "="*80)
        print("COMPLETE DEBUG ANALYSIS")
        print("="*80)
        
        # Step 1: Load and examine metadata
        metadata = self.load_and_examine_metadata()
        
        # Step 2: Examine DGE file structure
        separator, sample_columns = self.examine_dge_file(dge_file_path)
        
        if separator is None:
            print("❌ Could not determine file format!")
            return
        
        # Step 3: Load full DGE file
        dge_data = self.load_full_dge_file(dge_file_path, separator)
        
        if dge_data is None:
            print("❌ Could not load DGE file!")
            return
        
        # Step 4: Test cell matching
        matching_strategy = self.test_cell_matching(metadata, dge_data)
        
        # Step 5: Create a small test AnnData
        print("\n" + "="*50)
        print("CREATING TEST ANNDATA")
        print("="*50)
        
        try:
            # Create AnnData (genes as rows, cells as columns in DGE -> transpose for AnnData)
            test_adata = ad.AnnData(X=dge_data.T.values.astype(np.float32))
            test_adata.var_names = dge_data.index.astype(str)
            test_adata.obs_names = dge_data.columns.astype(str)
            
            print(f"Test AnnData created: {test_adata.shape[0]} cells × {test_adata.shape[1]} genes")
            
            # Try to match metadata
            metadata_dict = metadata.set_index('Cell.name').to_dict('index')
            
            matched_cells = []
            for cell_name in test_adata.obs_names:
                if cell_name in metadata_dict:
                    matched_cells.append(True)
                else:
                    matched_cells.append(False)
            
            n_matched = sum(matched_cells)
            print(f"Cells with metadata: {n_matched}/{len(test_adata.obs_names)}")
            
            if n_matched > 0:
                print("✅ Success! Cells can be matched to metadata.")
                
                # Filter to matched cells only
                test_adata_filtered = test_adata[matched_cells].copy()
                print(f"Filtered AnnData: {test_adata_filtered.shape[0]} cells × {test_adata_filtered.shape[1]} genes")
                
                return test_adata_filtered
            else:
                print("❌ No cells could be matched to metadata!")
                
                # Show detailed comparison
                print("\nDETAILED COMPARISON:")
                print("First 5 DGE cell names:")
                for i, name in enumerate(test_adata.obs_names[:5]):
                    print(f"  '{name}'")
                
                print("First 5 metadata cell names:")
                for i, name in enumerate(metadata['Cell.name'][:5]):
                    print(f"  '{name}'")
                
        except Exception as e:
            print(f"❌ Error creating test AnnData: {e}")
            import traceback
            traceback.print_exc()

# Usage functions
def debug_metadata_only(csv_file_path):
    """Just examine the metadata file"""
    debugger = DebugMCADataProcessor(csv_file_path)
    debugger.load_and_examine_metadata()

def debug_dge_only(dge_file_path):
    """Just examine a DGE file"""
    debugger = DebugMCADataProcessor("")
    debugger.examine_dge_file(dge_file_path)

def full_debug(csv_file_path, dge_file_path):
    """Complete debug analysis"""
    debugger = DebugMCADataProcessor(csv_file_path)
    return debugger.debug_single_file(dge_file_path)

# Quick diagnostic functions
def quick_file_peek(file_path, n_lines=5):
    """Quick peek at any file"""
    print(f"\nQuick peek at: {file_path}")
    print("-" * 40)
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                print(f"Line {i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"Error reading file: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    csv_path = "F:/Thesis/Datasets/MCA_CellAssignments.csv"
    dge_path = "F:/Thesis/Datasets/Bladder_dge.txt"
    
    print("Use these functions to debug:")
    print("1. debug_metadata_only(csv_path)")
    print("2. debug_dge_only(dge_path)")
    print("3. full_debug(csv_path, dge_path)")
    print("4. quick_file_peek(file_path)")
    full_debug(csv_path, dge_path)