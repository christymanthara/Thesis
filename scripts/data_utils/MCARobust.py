import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import tarfile
import zipfile
import gzip
import os
import tempfile
from pathlib import Path
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustMCADataProcessor:
    def __init__(self, tar_file_path, csv_file_path, output_dir=None):
        """
        Initialize the MCA data processor with robust cell matching
        
        Parameters:
        -----------
        tar_file_path : str
            Path to the main tar file containing all tissue zip files
        csv_file_path : str
            Path to the CSV file with cell metadata
        output_dir : str, optional
            Directory to save intermediate and final results
        """
        self.tar_file_path = tar_file_path
        self.csv_file_path = csv_file_path
        
        current_dir = Path.cwd()
        self.output_dir = Path(output_dir) if output_dir else current_dir
        self.temp_dir = current_dir / "mca_temp_extraction"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Working directory: {current_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temporary extraction directory: {self.temp_dir}")

    def extract_tar_files(self):
        """Extract the main tar file and individual tissue zip/gz files"""
        logger.info("Extracting main tar file...")
        
        with tarfile.open(self.tar_file_path, 'r') as main_tar:
            main_tar.extractall(self.temp_dir)
        
        zip_files = list(Path(self.temp_dir).rglob("*.zip"))
        gz_files = [f for f in Path(self.temp_dir).rglob("*.gz") if not str(f).endswith('.tar.gz')]
        
        logger.info(f"Found {len(zip_files)} tissue zip files")
        logger.info(f"Found {len(gz_files)} tissue .gz files")
        
        tissue_dirs = {}
        
        # Extract zip files
        for zip_file in zip_files:
            tissue_name = zip_file.stem
            tissue_dir = Path(self.temp_dir) / tissue_name
            tissue_dir.mkdir(exist_ok=True)
            
            try:
                with zipfile.ZipFile(zip_file, 'r') as tissue_zip:
                    tissue_zip.extractall(tissue_dir)
                tissue_dirs[tissue_name] = tissue_dir
                logger.info(f"Extracted {tissue_name} (zip)")
            except Exception as e:
                logger.warning(f"Failed to extract {tissue_name} (zip): {e}")
        
        # Extract gz files
        for gz_file in gz_files:
            tissue_name = gz_file.stem
            tissue_dir = Path(self.temp_dir) / tissue_name
            tissue_dir.mkdir(exist_ok=True)
            
            try:
                with gzip.open(gz_file, 'rb') as f_in:
                    output_file = tissue_dir / gz_file.stem
                    with open(output_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                tissue_dirs[tissue_name] = tissue_dir
                logger.info(f"Extracted {tissue_name} (gz)")
            except Exception as e:
                logger.warning(f"Failed to extract {tissue_name} (gz): {e}")
        
        logger.info(f"Total extracted: {len(tissue_dirs)} tissue directories")
        return tissue_dirs

    def load_metadata(self):
        """Load and process the metadata CSV file"""
        logger.info("Loading metadata...")
        metadata = pd.read_csv(self.csv_file_path, index_col=0)
        
        # Create a direct mapping using Cell.name as the key
        metadata_dict = {}
        
        for idx, row in metadata.iterrows():
            cell_name = row['Cell.name']
            
            metadata_entry = {
                'batch_id': row['Batch'],
                'labels': row['Annotation'],
                'tissue': row['Tissue'],
                'cluster_id': row['ClusterID'],
                'cell_barcode': row['Cell.Barcode']
            }
            
            metadata_dict[cell_name] = metadata_entry
        
        logger.info(f"Loaded metadata for {len(metadata)} cells")
        return metadata_dict, metadata

    def read_dge_file(self, file_path):
        """Read a DGE file with improved handling for different formats"""
        try:
            logger.info(f"Reading DGE file: {file_path}")
            
            # First, let's examine the file structure
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
            
            logger.info(f"First line preview: {first_line[:200]}...")
            logger.info(f"Second line preview: {second_line[:200]}...")
            
            # Determine separator based on file extension and content
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                sep = ','
            elif file_ext in ['.txt', '.tsv', '.dge']:
                # Check if tab-separated
                if '\t' in first_line:
                    sep = '\t'
                elif ' ' in first_line and first_line.count(' ') > first_line.count('\t'):
                    sep = ' '
                else:
                    sep = '\t'  # Default to tab
            else:
                # Auto-detect separator
                if '\t' in first_line:
                    sep = '\t'
                elif ',' in first_line:
                    sep = ','
                elif ' ' in first_line:
                    sep = ' '
                else:
                    sep = '\t'  # Default to tab
            
            logger.info(f"Using separator: '{sep}' for file: {file_path.name}")
            
            # Try different reading approaches
            data = None
            
            # Approach 1: Standard reading with proper header handling
            try:
                data = pd.read_csv(file_path, sep=sep, index_col=0, header=0, low_memory=False)
                logger.info(f"Approach 1 - Shape after reading: {data.shape}")
                
                if data.shape[1] == 0:
                    raise ValueError("No columns found")
                    
            except Exception as e1:
                logger.warning(f"Approach 1 failed: {e1}")
                
                # Approach 2: Read without index_col first, then set it
                try:
                    data = pd.read_csv(file_path, sep=sep, header=0, low_memory=False)
                    logger.info(f"Approach 2 - Raw shape: {data.shape}")
                    
                    # Set first column as index
                    if data.shape[1] > 1:
                        data = data.set_index(data.columns[0])
                        logger.info(f"Approach 2 - After setting index: {data.shape}")
                    else:
                        raise ValueError("Only one column found")
                        
                except Exception as e2:
                    logger.warning(f"Approach 2 failed: {e2}")
                    
                    # Approach 3: Manual parsing for space-separated files
                    try:
                        logger.info("Trying manual parsing approach...")
                        
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        # Parse header (cell names)
                        header_line = lines[0].strip()
                        if sep == ' ':
                            # Handle space-separated with quoted strings
                            cell_names = []
                            current_cell = ""
                            in_quotes = False
                            
                            i = 0
                            while i < len(header_line):
                                char = header_line[i]
                                if char == '"':
                                    if in_quotes:
                                        # End of quoted string
                                        cell_names.append(current_cell)
                                        current_cell = ""
                                        in_quotes = False
                                    else:
                                        # Start of quoted string
                                        in_quotes = True
                                elif in_quotes:
                                    current_cell += char
                                elif char == ' ' and not in_quotes and current_cell:
                                    # Unquoted cell name
                                    cell_names.append(current_cell)
                                    current_cell = ""
                                elif char != ' ' and not in_quotes:
                                    current_cell += char
                                i += 1
                            
                            if current_cell:
                                cell_names.append(current_cell)
                                
                        else:
                            # Use standard split for tab/comma separated
                            cell_names = header_line.split(sep)
                        
                        # Remove empty strings and quotes
                        cell_names = [name.strip().strip('"').strip("'") for name in cell_names if name.strip()]
                        
                        # First cell name might be empty (for gene name column)
                        if not cell_names[0]:
                            cell_names = cell_names[1:]
                        
                        logger.info(f"Found {len(cell_names)} cell names")
                        logger.info(f"Sample cell names: {cell_names[:5]}")
                        
                        # Parse data rows
                        gene_names = []
                        expression_data = []
                        
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                                
                            parts = line.strip().split(sep)
                            if len(parts) < 2:
                                continue
                                
                            gene_name = parts[0].strip().strip('"').strip("'")
                            gene_names.append(gene_name)
                            
                            # Parse expression values
                            expr_values = []
                            for val in parts[1:]:
                                try:
                                    expr_values.append(float(val))
                                except ValueError:
                                    expr_values.append(0.0)
                            
                            expression_data.append(expr_values)
                        
                        # Create DataFrame
                        data = pd.DataFrame(expression_data, index=gene_names, columns=cell_names[:len(expression_data[0])])
                        logger.info(f"Approach 3 - Manual parsing result: {data.shape}")
                        
                    except Exception as e3:
                        logger.error(f"Approach 3 failed: {e3}")
                        return None
            
            if data is None:
                logger.error("All reading approaches failed")
                return None
            
            logger.info(f"Raw data shape: {data.shape}")
            
            # Clean up column names (remove quotes)
            data.columns = [str(col).strip().strip('"').strip("'") for col in data.columns]
            
            # Ensure numeric data
            numeric_columns = []
            for col in data.columns:
                try:
                    pd.to_numeric(data[col])
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    logger.warning(f"Column {col} is not numeric, skipping")
            
            if numeric_columns:
                data = data[numeric_columns]
                # Convert to numeric, coercing errors to 0
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            else:
                logger.warning("No numeric columns found")
                return None
            
            # Basic sanity checks
            if data.shape[0] < 1000:
                logger.warning(f"Low gene count: {data.shape[0]} genes")
            if data.shape[1] < 10:
                logger.warning(f"Low cell count: {data.shape[1]} cells")
                
            logger.info(f"Final data shape: {data.shape[0]} genes, {data.shape[1]} cells")
            logger.info(f"Sample cell names: {list(data.columns[:5])}")
            logger.info(f"Sample gene names: {list(data.index[:5])}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def test_cell_matching_strategies(self, sample_cells, metadata_dict, metadata_df):
        """Test different cell matching strategies - adapted from debug file"""
        logger.info("Testing cell matching strategies...")
        
        # Strategy 1: Direct matching
        direct_matches = set(sample_cells) & set(metadata_dict.keys())
        logger.info(f"Direct matches: {len(direct_matches)}/{len(sample_cells)}")
        
        if len(direct_matches) > 0:
            match_rate = len(direct_matches) / len(sample_cells)
            logger.info(f"✓ Direct matching works! Match rate: {match_rate:.2%}")
            return 'direct', match_rate
        
        # Strategy 2: Check barcode substring matching
        logger.info("Testing barcode substring matching...")
        metadata_barcodes = list(metadata_df['Cell.Barcode'].unique())
        
        barcode_matches = 0
        barcode_mapping = {}
        
        for cell in sample_cells[:100]:  # Test first 100 cells
            for barcode in metadata_barcodes:
                if str(barcode) in str(cell):
                    barcode_matches += 1
                    # Find the metadata entry with this barcode
                    matching_rows = metadata_df[metadata_df['Cell.Barcode'] == barcode]
                    if len(matching_rows) > 0:
                        barcode_mapping[cell] = matching_rows['Cell.name'].iloc[0]
                    break
        
        if barcode_matches > 0:
            logger.info(f"✓ Barcode substring matching works! Found {barcode_matches} matches in sample")
            return 'barcode_substring', barcode_matches / 100
        
        # Strategy 3: Tissue + barcode combinations
        logger.info("Testing tissue+barcode combinations...")
        
        # Create potential combinations
        tissue_barcode_combos = set()
        for _, row in metadata_df.iterrows():
            combo1 = f"{row['Tissue']}.{row['Cell.Barcode']}"
            combo2 = f"{row['Tissue']}_{row['Batch']}.{row['Cell.Barcode']}"
            tissue_barcode_combos.update([combo1, combo2])
        
        combo_matches = set(sample_cells) & tissue_barcode_combos
        if len(combo_matches) > 0:
            match_rate = len(combo_matches) / len(sample_cells)
            logger.info(f"✓ Tissue+barcode matching works! Match rate: {match_rate:.2%}")
            return 'tissue_barcode', match_rate
        
        # No strategy worked
        logger.warning("❌ No matching strategy worked!")
        logger.warning("Sample DGE cell names:")
        for i, cell in enumerate(sample_cells[:5]):
            logger.warning(f"  '{cell}'")
        logger.warning("Sample metadata cell names:")
        sample_meta_names = list(metadata_dict.keys())[:5]
        for i, name in enumerate(sample_meta_names):
            logger.warning(f"  '{name}'")
        
        return None, 0

    def match_cells_with_strategy(self, cell_names, metadata_dict, metadata_df, strategy):
        """Match cells using the specified strategy - FIXED VERSION"""
        matched_metadata = []
        valid_cells = []
        
        if strategy == 'direct':
            for cell_name in cell_names:
                clean_name = str(cell_name).strip().strip('"').strip("'")
                if clean_name in metadata_dict:
                    matched_metadata.append(metadata_dict[clean_name])
                    valid_cells.append(True)
                else:
                    matched_metadata.append(None)  # FIX: Always append something
                    valid_cells.append(False)
        
        elif strategy == 'barcode_substring':
            # Create reverse mapping: barcode -> metadata
            barcode_to_meta = {}
            for cell_name, meta in metadata_dict.items():
                barcode_to_meta[meta['cell_barcode']] = meta
            
            for cell_name in cell_names:
                matched = False
                for barcode, meta in barcode_to_meta.items():
                    if str(barcode) in str(cell_name):
                        matched_metadata.append(meta)
                        valid_cells.append(True)
                        matched = True
                        break
                if not matched:
                    matched_metadata.append(None)  # FIX: Always append something
                    valid_cells.append(False)
        
        elif strategy == 'tissue_barcode':
            # Create tissue+barcode combinations mapping
            combo_to_meta = {}
            for cell_name, meta in metadata_dict.items():
                combo1 = f"{meta['tissue']}.{meta['cell_barcode']}"
                combo2 = f"{meta['tissue']}_{meta['batch_id']}.{meta['cell_barcode']}"
                combo_to_meta[combo1] = meta
                combo_to_meta[combo2] = meta
            
            for cell_name in cell_names:
                if cell_name in combo_to_meta:
                    matched_metadata.append(combo_to_meta[cell_name])
                    valid_cells.append(True)
                else:
                    matched_metadata.append(None)  # FIX: Always append something
                    valid_cells.append(False)
        
        # Ensure matched_metadata and valid_cells have same length
        assert len(matched_metadata) == len(valid_cells) == len(cell_names), \
            f"Length mismatch: metadata={len(matched_metadata)}, valid={len(valid_cells)}, cells={len(cell_names)}"
        
        matched_count = sum(valid_cells)
        logger.info(f"Matched {matched_count}/{len(cell_names)} cells using {strategy} strategy")
        
        return matched_metadata, valid_cells

    def create_tissue_anndata(self, dge_data, matched_metadata, valid_cells, tissue_name, source_file):
        """Create AnnData following the debug file logic - FIXED VERSION"""
        logger.info(f"Creating AnnData for {tissue_name}...")
        
        # Step 1: Create full AnnData (genes as rows, cells as columns -> transpose for AnnData)
        adata_full = ad.AnnData(X=dge_data.T.values.astype(np.float32))
        adata_full.var_names = dge_data.index.astype(str)  # Gene names
        adata_full.obs_names = dge_data.columns.astype(str)  # Cell names
        
        logger.info(f"Full AnnData created: {adata_full.shape[0]} cells × {adata_full.shape[1]} genes")
        
        # Step 2: Filter to matched cells only (following debug logic)
        matched_count = sum(valid_cells)
        if matched_count == 0:
            logger.warning(f"No matched cells for {tissue_name} - skipping")
            return None
        
        adata_filtered = adata_full[valid_cells].copy()
        
        # FIX: Filter valid_metadata to only include non-None entries (matching cells)
        valid_metadata = [m for m, v in zip(matched_metadata, valid_cells) if v and m is not None]
        
        logger.info(f"Filtered AnnData: {adata_filtered.shape[0]} cells × {adata_filtered.shape[1]} genes")
        logger.info(f"Valid metadata entries: {len(valid_metadata)}")
        
        # Double-check lengths match
        if len(valid_metadata) != adata_filtered.shape[0]:
            logger.error(f"Length mismatch: AnnData has {adata_filtered.shape[0]} cells but metadata has {len(valid_metadata)} entries")
            return None
        
        # Step 3: Add metadata to observations
        adata_filtered.obs['batch_id'] = [m['batch_id'] for m in valid_metadata]
        adata_filtered.obs['labels'] = [m['labels'] for m in valid_metadata]
        adata_filtered.obs['Tissue'] = [m['tissue'] for m in valid_metadata]  # Using 'Tissue' for your concat logic
        adata_filtered.obs['cluster_id'] = [m['cluster_id'] for m in valid_metadata]
        
        # Step 4: Add tissue information to uns
        adata_filtered.uns['tissue'] = tissue_name
        adata_filtered.uns['source_file'] = source_file
        
        return adata_filtered

    def process_tissue_data(self, tissue_dirs, metadata_dict, metadata_df):
        """Process each tissue directory to create AnnData objects"""
        all_adatas = []
        matching_strategy = None
        
        for tissue_name, tissue_dir in tissue_dirs.items():
            logger.info(f"Processing tissue: {tissue_name}")
            
            # Find DGE files
            dge_files = []
            for pattern in ["*.txt", "*.tsv", "*.csv", "*dge*", "*matrix*", "*expression*"]:
                potential_files = list(tissue_dir.rglob(pattern))
                dge_files.extend([f for f in potential_files if f.is_file()])
            
            dge_files = list(set(dge_files))
            dge_files.sort(key=lambda x: x.stat().st_size, reverse=True)
            
            if not dge_files:
                logger.warning(f"No DGE files found for {tissue_name}")
                continue
            
            logger.info(f"Found DGE files for {tissue_name}: {[f.name for f in dge_files]}")
            
            # Process each DGE file
            for dge_file in dge_files:
                dge_data = self.read_dge_file(dge_file)
                if dge_data is None:
                    continue
                
                # Test cell matching strategies (only for first file if not determined)
                if matching_strategy is None:
                    sample_cells = list(dge_data.columns[:100])  # Sample first 100 cells
                    matching_strategy, match_rate = self.test_cell_matching_strategies(
                        sample_cells, metadata_dict, metadata_df
                    )
                    
                    if matching_strategy is None:
                        logger.error("No viable cell matching strategy found!")
                        continue
                    
                    logger.info(f"Using matching strategy: {matching_strategy}")
                
                # Match cells using determined strategy
                matched_metadata, valid_cells = self.match_cells_with_strategy(
                    dge_data.columns, metadata_dict, metadata_df, matching_strategy
                )
                
                # Create AnnData following debug logic
                tissue_adata = self.create_tissue_anndata(
                    dge_data, matched_metadata, valid_cells, tissue_name, dge_file.name
                )
                
                if tissue_adata is not None:
                    all_adatas.append(tissue_adata)
                    logger.info(f"✓ Added {tissue_name}: {tissue_adata.shape[0]} cells, {tissue_adata.shape[1]} genes")
        
        return all_adatas

    def concatenate_all_adatas(self, adatas):
        """Concatenate all AnnData objects using your suggested approach"""
        logger.info("Concatenating all AnnData objects...")
        
        if not adatas:
            raise ValueError("No AnnData objects to concatenate")
        
        # Concatenate all adatas
        adata_full = ad.concat(adatas, axis=0, join='outer', fill_value=0)
        
        # Optional: fill in uns per tissue (your suggested approach)
        for tissue in adata_full.obs['Tissue'].unique():
            adata_full.uns[tissue] = {
                'cells': adata_full.obs.query('Tissue == @tissue').index.tolist(),
                'n_cells': (adata_full.obs['Tissue'] == tissue).sum()
            }
        
        # Add global metadata
        adata_full.uns['name'] = 'Mouse Cell Atlas'
        adata_full.uns['organism'] = 'mouse'
        adata_full.uns['year'] = 2018
        adata_full.uns['n_tissues'] = len(adata_full.obs['Tissue'].unique())
        
        logger.info(f"Final concatenated dataset:")
        logger.info(f"  Shape: {adata_full.shape[0]} cells × {adata_full.shape[1]} genes")
        logger.info(f"  Tissues: {adata_full.obs['Tissue'].nunique()}")
        logger.info(f"  Batches: {adata_full.obs['batch_id'].nunique()}")
        logger.info(f"  Cell types: {adata_full.obs['labels'].nunique()}")
        
        return adata_full

    def process(self):
        """Run the complete processing pipeline"""
        try:
            # Step 1: Extract files
            tissue_dirs = self.extract_tar_files()
            
            # Step 2: Load metadata
            metadata_dict, metadata_df = self.load_metadata()
            
            # Step 3: Process each tissue (with robust cell matching)
            all_adatas = self.process_tissue_data(tissue_dirs, metadata_dict, metadata_df)
            
            # Step 4: Concatenate all AnnData objects
            final_adata = self.concatenate_all_adatas(all_adatas)
            
            # Step 5: Save result
            output_path = self.output_dir / "mca_merged_final.h5ad"
            final_adata.write(output_path)
            logger.info(f"Saved final dataset to: {output_path}")
            
            # Step 6: Print summary
            self.print_detailed_summary(final_adata)
            
            return final_adata
            
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            raise
        finally:
            # Cleanup
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info("Cleaned up temporary files")

    def print_detailed_summary(self, adata):
        """Print comprehensive summary of the final dataset"""
        print("\n" + "="*60)
        print("MOUSE CELL ATLAS - FINAL DATASET SUMMARY")
        print("="*60)
        print(f"📊 Dataset dimensions: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
        
        print(f"\n🧬 Tissue distribution:")
        tissue_counts = adata.obs['Tissue'].value_counts()
        for tissue, count in tissue_counts.items():
            print(f"   {tissue}: {count:,} cells")
        
        print(f"\n🔬 Batch distribution:")
        batch_counts = adata.obs['batch_id'].value_counts()
        print(f"   Total batches: {len(batch_counts)}")
        print(f"   Cells per batch: {batch_counts.min():,} - {batch_counts.max():,}")
        
        print(f"\n🏷️ Cell type diversity:")
        label_counts = adata.obs['labels'].value_counts()
        print(f"   Total cell types: {len(label_counts)}")
        print(f"   Top 10 cell types:")
        for cell_type, count in label_counts.head(10).items():
            print(f"      {cell_type}: {count:,} cells")
        
        print(f"\n📦 Metadata structure:")
        print(f"   adata.obs columns: {list(adata.obs.columns)}")
        print(f"   adata.uns keys: {list(adata.uns.keys())}")
        
        # Per-tissue summary (using your uns structure)
        print(f"\n🧪 Per-tissue summary:")
        for tissue in adata.obs['Tissue'].unique():
            if tissue in adata.uns:
                n_cells = adata.uns[tissue]['n_cells']
                print(f"   {tissue}: {n_cells:,} cells")
        
        print("="*60)


# Usage example
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process MCA datasets with robust cell matching')
    parser.add_argument('tar_file', help='Path to the main tar file')
    parser.add_argument('csv_file', help='Path to the metadata CSV file')
    parser.add_argument('--output_dir', help='Output directory', default='./mca_output')
    
    args = parser.parse_args()
    
    processor = RobustMCADataProcessor(args.tar_file, args.csv_file, args.output_dir)
    final_adata = processor.process()
    
    return final_adata

if __name__ == "__main__":
    final_adata = main()