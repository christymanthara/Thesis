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

class ImprovedMCADataProcessor:
    def __init__(self, tar_file_path, csv_file_path, output_dir=None):
        """
        Initialize the MCA data processor
        
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
        """Load and process the metadata CSV file with simplified mapping"""
        logger.info("Loading metadata...")
        metadata = pd.read_csv(self.csv_file_path, index_col=0)
        
        # Create a direct mapping using Cell.name as the key
        metadata_dict = {}
        
        for idx, row in metadata.iterrows():
            cell_name = row['Cell.name']  # This should match DGE headers directly
            
            metadata_entry = {
                'batch_id': row['Batch'],
                'labels': row['Annotation'],
                'tissue': row['Tissue'],
                'cluster_id': row['ClusterID'],
                'cell_barcode': row['Cell.Barcode']
            }
            
            metadata_dict[cell_name] = metadata_entry
        
        logger.info(f"Loaded metadata for {len(metadata)} cells")
        return metadata_dict

    def read_dge_file(self, file_path):
        """Read a DGE file with improved handling for your specific format"""
        try:
            logger.info(f"Reading DGE file: {file_path}")
            
            # Since you know the format: genes as rows, cells as columns
            # Use pandas with appropriate settings
            data = pd.read_csv(file_path, sep='\t', index_col=0, low_memory=False)
            
            logger.info(f"Raw data shape: {data.shape}")
            logger.info(f"Sample cell names: {list(data.columns[:3])}")
            logger.info(f"Sample gene names: {list(data.index[:3])}")
            
            # Ensure numeric data
            data = data.select_dtypes(include=[np.number])
            
            # Basic sanity checks
            if data.shape[0] < 1000:  # Less than 1000 genes seems low
                logger.warning(f"Low gene count: {data.shape[0]} genes")
            if data.shape[1] < 10:  # Less than 10 cells seems low
                logger.warning(f"Low cell count: {data.shape[1]} cells")
                
            logger.info(f"Final data shape: {data.shape[0]} genes, {data.shape[1]} cells")
            return data
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def match_cells_to_metadata(self, cell_names, metadata_dict):
        """Match cell names to metadata with improved strategy"""
        matched_metadata = []
        valid_cells = []
        
        for cell_name in cell_names:
            # Clean the cell name
            clean_name = str(cell_name).strip().strip('"').strip("'")
            
            print(f"Matching cell of the cleaned name: '{clean_name}'")
            if clean_name in metadata_dict:
                print(f"found metadata for cell: '{clean_name}'")
                matched_metadata.append(metadata_dict[clean_name])
                valid_cells.append(True)
            else:
                valid_cells.append(False)
        
        matched_count = sum(valid_cells)
        logger.info(f"Matched {matched_count}/{len(cell_names)} cells to metadata")
        
        if matched_count == 0:
            # Debug: show some examples of cell names that didn't match
            logger.warning("No matches found. Sample cell names:")
            for i, name in enumerate(cell_names[:5]):
                logger.warning(f"  Cell {i}: '{name}'")
            logger.warning("Sample metadata keys:")
            sample_keys = list(metadata_dict.keys())[:5]
            for i, key in enumerate(sample_keys):
                logger.warning(f"  Key {i}: '{key}'")
        
        return matched_metadata, valid_cells

    def process_tissue_data(self, tissue_dirs, metadata_dict):
        """Process each tissue directory to create AnnData objects"""
        tissue_datasets = {}
        
        for tissue_name, tissue_dir in tissue_dirs.items():
            logger.info(f"Processing tissue: {tissue_name}")
            
            # Find potential DGE files
            all_files = list(tissue_dir.rglob("*"))
            logger.info(f"Files in {tissue_name}: {[f.name for f in all_files if f.is_file()]}")
            
            # Look for DGE files (common patterns)
            dge_files = []
            for pattern in ["*.txt", "*.tsv", "*.csv", "*dge*", "*matrix*", "*expression*"]:
                print(f"Searching for files matching pattern: {pattern}")
                print(f"files found")
                potential_files = list(tissue_dir.rglob(pattern))
                dge_files.extend([f for f in potential_files if f.is_file()])
            
            # Remove duplicates and sort by size
            dge_files = list(set(dge_files))
            dge_files.sort(key=lambda x: x.stat().st_size, reverse=True)
            
            if not dge_files:
                logger.warning(f"No DGE files found for {tissue_name}")
                continue
            
            logger.info(f"Found DGE files for {tissue_name}: {[f.name for f in dge_files]}")
            
            # Process each DGE file
            batch_datasets = []
            for dge_file in dge_files:
                dge_data = self.read_dge_file(dge_file)
                if dge_data is None:
                    continue
                
                # Create AnnData object
                # X should be cells x genes, so we transpose the DGE data
                adata = ad.AnnData(X=dge_data.T.values.astype(np.float32))
                adata.var_names = dge_data.index.astype(str)  # Gene names
                adata.obs_names = dge_data.columns.astype(str)  # Cell names
                
                logger.info(f"Created AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")
                
                # Match cells to metadata
                matched_metadata, valid_cells = self.match_cells_to_metadata(
                    adata.obs_names, metadata_dict
                )
                
                matched_count = sum(valid_cells)
                if matched_count == 0:
                    logger.warning(f"No metadata matches for {dge_file.name}")
                    continue
                
                # Filter to cells with metadata
                adata_filtered = adata[valid_cells].copy()
                valid_metadata = [m for m, v in zip(matched_metadata, valid_cells) if v]
                
                # Add metadata to observations
                adata_filtered.obs['batch_id'] = [m['batch_id'] for m in valid_metadata]
                adata_filtered.obs['labels'] = [m['labels'] for m in valid_metadata]
                adata_filtered.obs['tissue'] = [m['tissue'] for m in valid_metadata]
                adata_filtered.obs['cluster_id'] = [m['cluster_id'] for m in valid_metadata]
                
                # Add tissue information to uns
                adata_filtered.uns['tissue'] = tissue_name
                adata_filtered.uns['source_file'] = dge_file.name
                
                batch_datasets.append(adata_filtered)
                logger.info(f"Added dataset: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes")
            
            # Concatenate batches for this tissue
            if batch_datasets:
                if len(batch_datasets) == 1:
                    tissue_adata = batch_datasets[0]
                else:
                    tissue_adata = ad.concat(batch_datasets, axis=0, join='outer', fill_value=0)
                
                tissue_datasets[tissue_name] = tissue_adata
                logger.info(f"Tissue {tissue_name} final: {tissue_adata.shape[0]} cells, {tissue_adata.shape[1]} genes")
        
        return tissue_datasets

    def apply_quality_filters(self, tissue_datasets):
        """Apply quality control filters"""
        logger.info("Applying quality control filters...")
        
        filtered_datasets = {}
        
        for tissue_name, adata in tissue_datasets.items():
            logger.info(f"Filtering {tissue_name}...")
            
            # Check batch diversity
            unique_batches = adata.obs['batch_id'].unique()
            if len(unique_batches) < 2:
                logger.info(f"Skipping {tissue_name}: only {len(unique_batches)} batch(es)")
                continue
            
            # Calculate QC metrics
            sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
            
            # Filter cells: at least 250 genes expressed
            n_cells_before = adata.shape[0]
            adata = adata[adata.obs['n_genes_by_counts'] >= 250].copy()
            n_cells_after = adata.shape[0]
            logger.info(f"Cell filter (>=250 genes): {n_cells_before} -> {n_cells_after}")
            
            # Filter genes: expressed in at least 50 cells
            n_genes_before = adata.shape[1]
            sc.pp.filter_genes(adata, min_cells=50)
            n_genes_after = adata.shape[1]
            logger.info(f"Gene filter (>=50 cells): {n_genes_before} -> {n_genes_after}")
            
            # Filter cell types: at least 1% of cells
            cell_type_counts = adata.obs['labels'].value_counts()
            min_cells_per_type = max(10, len(adata) * 0.01)  # At least 10 cells or 1%
            valid_cell_types = cell_type_counts[cell_type_counts >= min_cells_per_type].index
            
            n_cells_before_ct = adata.shape[0]
            adata = adata[adata.obs['labels'].isin(valid_cell_types)].copy()
            n_cells_after_ct = adata.shape[0]
            logger.info(f"Cell type filter (>=1%): {n_cells_before_ct} -> {n_cells_after_ct}")
            
            # Check final batch distribution
            final_batch_counts = adata.obs['batch_id'].value_counts()
            if len(final_batch_counts) >= 2 and final_batch_counts.min() >= 50:  # At least 50 cells per batch
                filtered_datasets[tissue_name] = adata
                logger.info(f"✓ Kept {tissue_name}: {adata.shape[0]} cells, {adata.shape[1]} genes")
            else:
                logger.info(f"✗ Skipped {tissue_name}: insufficient batch representation")
        
        logger.info(f"Selected {len(filtered_datasets)}/{len(tissue_datasets)} tissues after filtering")
        return filtered_datasets

    def merge_all_datasets(self, filtered_datasets):
        """Merge all tissue datasets into a single AnnData object"""
        logger.info("Merging all datasets...")
        
        if not filtered_datasets:
            raise ValueError("No datasets remaining after filtering")
        
        # Concatenate all datasets
        all_datasets = list(filtered_datasets.values())
        merged_adata = ad.concat(all_datasets, axis=0, join='outer', fill_value=0)
        
        # Keep only essential columns in obs
        essential_cols = ['batch_id', 'labels', 'tissue']
        merged_adata.obs = merged_adata.obs[essential_cols].copy()
        
        # Add global metadata
        merged_adata.uns['name'] = 'Mouse Cell Atlas'
        merged_adata.uns['organism'] = 'mouse'
        merged_adata.uns['year'] = 2018
        merged_adata.uns['tissues_included'] = list(filtered_datasets.keys())
        merged_adata.uns['n_tissues'] = len(filtered_datasets)
        
        logger.info(f"Final merged dataset:")
        logger.info(f"  Shape: {merged_adata.shape[0]} cells × {merged_adata.shape[1]} genes")
        logger.info(f"  Tissues: {merged_adata.obs['tissue'].nunique()}")
        logger.info(f"  Batches: {merged_adata.obs['batch_id'].nunique()}")
        logger.info(f"  Cell types: {merged_adata.obs['labels'].nunique()}")
        
        return merged_adata

    def process(self):
        """Run the complete processing pipeline"""
        try:
            # Step 1: Extract files
            tissue_dirs = self.extract_tar_files()
            
            # Step 2: Load metadata
            metadata_dict = self.load_metadata()
            
            # Step 3: Process each tissue
            tissue_datasets = self.process_tissue_data(tissue_dirs, metadata_dict)
            
            # Step 4: Apply quality filters
            # filtered_datasets = self.apply_quality_filters(tissue_datasets)
            
            # Step 5: Merge all datasets
            final_adata = self.merge_all_datasets(tissue_datasets)
            
            # Step 6: Save result
            output_path = self.output_dir / "mca_merged_final.h5ad"
            final_adata.write(output_path)
            logger.info(f"Saved final dataset to: {output_path}")
            
            # Step 7: Print summary
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
        tissue_counts = adata.obs['tissue'].value_counts()
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
        
        print("="*60)


# Usage example
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process MCA datasets (improved version)')
    parser.add_argument('tar_file', help='Path to the main tar file')
    parser.add_argument('csv_file', help='Path to the metadata CSV file')
    parser.add_argument('--output_dir', help='Output directory', default='./mca_output')
    
    args = parser.parse_args()
    
    processor = ImprovedMCADataProcessor(args.tar_file, args.csv_file, args.output_dir)
    final_adata = processor.process()
    
    return final_adata

if __name__ == "__main__":
    final_adata = main()