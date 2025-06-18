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

class MCADataProcessor:
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
        
        # Use current working directory for operations
        current_dir = Path.cwd()
        self.output_dir = Path(output_dir) if output_dir else current_dir
        self.temp_dir = current_dir / "mca_temp_extraction"
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Working directory: {current_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temporary extraction directory: {self.temp_dir}")

    def extract_tar_files(self):
        """Extract the main tar file and individual tissue zip/gz files"""
        logger.info("Extracting main tar file...")
        
        # Extract main tar file
        with tarfile.open(self.tar_file_path, 'r') as main_tar:
            main_tar.extractall(self.temp_dir)
        
        # Find all zip files
        zip_files = list(Path(self.temp_dir).rglob("*.zip"))
        logger.info(f"Found {len(zip_files)} tissue zip files")
        
        # Find all .gz files (gzipped files, not tar.gz archives)
        gz_files = list(Path(self.temp_dir).rglob("*.gz"))
        # Filter out .tar.gz files if any exist
        gz_files = [f for f in gz_files if not str(f).endswith('.tar.gz')]
        logger.info(f"Found {len(gz_files)} tissue .gz files")
        
        tissue_dirs = {}
        
        # Extract each tissue zip file
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
        
        # Extract each .gz file (these are individual gzipped files, not archives)
        for gz_file in gz_files:
            tissue_name = gz_file.stem  # Remove .gz extension
            tissue_dir = Path(self.temp_dir) / tissue_name
            tissue_dir.mkdir(exist_ok=True)
            
            try:
                # Extract the gzipped file
                with gzip.open(gz_file, 'rb') as f_in:
                    # Create the uncompressed file in the tissue directory
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
        
        # Create a mapping from cell barcode to metadata
        metadata_dict = {}
        for idx, row in metadata.iterrows():
            cell_key = f"{row['Tissue']}_{row['Batch']}.{row['Cell.Barcode']}"
            metadata_dict[cell_key] = {
                'batch_id': row['Batch'],
                'labels': row['Annotation'],
                'tissue': row['Tissue'],
                'cluster_id': row['ClusterID']
            }
        
        logger.info(f"Loaded metadata for {len(metadata_dict)} cells")
        return metadata_dict

    def read_dge_file(self, file_path):
        """Read a DGE (Digital Gene Expression) file"""
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    # Read the first line to get cell names
                    first_line = f.readline().strip()
                    cell_names = first_line.split('\t')[1:]  # Skip first column (gene names)
                    
                    # Read the rest of the file
                    f.seek(0)
                    data = pd.read_csv(f, sep='\t', index_col=0)
            else:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    cell_names = first_line.split('\t')[1:]
                    
                    f.seek(0)
                    data = pd.read_csv(f, sep='\t', index_col=0)
            
            logger.info(f"Loaded DGE file: {data.shape[0]} genes, {data.shape[1]} cells")
            return data
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def process_tissue_data(self, tissue_dirs, metadata_dict):
        """Process data for each tissue"""
        tissue_datasets = {}
        
        for tissue_name, tissue_dir in tissue_dirs.items():
            logger.info(f"Processing tissue: {tissue_name}")
            
            # Find DGE files in the tissue directory (look for common patterns)
            dge_files = []
            for pattern in ["*_dge.txt*", "*dge.txt*", "*_expression.txt*", "*.tsv*", "*.csv*"]:
                dge_files.extend(list(tissue_dir.rglob(pattern)))
            
            # Remove duplicates and filter out non-data files
            dge_files = list(set(dge_files))
            dge_files = [f for f in dge_files if not any(exclude in str(f).lower() 
                                                        for exclude in ['metadata', 'annotation', 'barcode'])]
            
            if not dge_files:
                logger.warning(f"No DGE files found for {tissue_name}")
                # Print directory contents for debugging
                all_files = list(tissue_dir.rglob("*"))
                logger.info(f"Files in {tissue_name}: {[f.name for f in all_files[:10]]}")
                continue
            
            # Process each DGE file (batch)
            batch_data = []
            for dge_file in dge_files:
                logger.info(f"Processing file: {dge_file}")
                
                # Load the DGE data
                dge_data = self.read_dge_file(dge_file)
                if dge_data is None:
                    continue
                
                # Create AnnData object
                adata = ad.AnnData(X=dge_data.T.values.astype(np.float32))
                adata.var_names = dge_data.index.astype(str)
                adata.obs_names = dge_data.columns.astype(str)
                
                # Add metadata
                obs_metadata = []
                valid_cells = []
                
                for cell_name in adata.obs_names:
                    if cell_name in metadata_dict:
                        obs_metadata.append(metadata_dict[cell_name])
                        valid_cells.append(True)
                    else:
                        # Try to match with tissue prefix
                        potential_keys = [k for k in metadata_dict.keys() if k.endswith(cell_name)]
                        if potential_keys:
                            obs_metadata.append(metadata_dict[potential_keys[0]])
                            valid_cells.append(True)
                        else:
                            valid_cells.append(False)
                
                # Filter cells with metadata
                if sum(valid_cells) == 0:
                    logger.warning(f"No cells with metadata found in {dge_file}")
                    continue
                
                adata = adata[valid_cells].copy()
                
                # Add metadata to obs
                for i, metadata in enumerate([m for m, v in zip(obs_metadata, valid_cells) if v]):
                    for key, value in metadata.items():
                        if key not in adata.obs.columns:
                            adata.obs[key] = ""
                        adata.obs.iloc[i, adata.obs.columns.get_loc(key)] = value
                
                batch_data.append(adata)
            
            if batch_data:
                # Concatenate batches for this tissue
                tissue_adata = ad.concat(batch_data, axis=0, join='outer', fill_value=0)
                tissue_datasets[tissue_name] = tissue_adata
                logger.info(f"Tissue {tissue_name}: {tissue_adata.shape[0]} cells, {tissue_adata.shape[1]} genes")
        
        return tissue_datasets

    def filter_datasets(self, tissue_datasets):
        """Apply filtering criteria to select appropriate datasets"""
        logger.info("Applying filtering criteria...")
        
        filtered_datasets = {}
        
        for tissue_name, adata in tissue_datasets.items():
            logger.info(f"Filtering {tissue_name}...")
            
            # Check if tissue has multiple batches
            unique_batches = adata.obs['batch_id'].unique()
            if len(unique_batches) < 2:
                logger.info(f"Skipping {tissue_name}: only {len(unique_batches)} batch(es)")
                continue
            
            # Check batch distribution
            batch_counts = adata.obs['batch_id'].value_counts()
            total_cells = len(adata)
            
            # Remove batches with <5% of total cells
            min_cells_per_batch = total_cells * 0.05
            valid_batches = batch_counts[batch_counts >= min_cells_per_batch].index
            
            if len(valid_batches) < 2:
                logger.info(f"Skipping {tissue_name}: insufficient cells in multiple batches")
                continue
            
            # Filter to keep only valid batches
            adata_filtered = adata[adata.obs['batch_id'].isin(valid_batches)].copy()
            
            # Remove cells expressing <250 genes
            sc.pp.calculate_qc_metrics(adata_filtered, percent_top=None, log1p=False, inplace=True)
            adata_filtered = adata_filtered[adata_filtered.obs['n_genes_by_counts'] >= 250].copy()
            
            # Remove genes expressed in <50 cells
            sc.pp.filter_genes(adata_filtered, min_cells=50)
            
            # Remove cell types representing <1% of total cell population
            cell_type_counts = adata_filtered.obs['labels'].value_counts()
            min_cells_per_type = len(adata_filtered) * 0.01
            valid_cell_types = cell_type_counts[cell_type_counts >= min_cells_per_type].index
            adata_filtered = adata_filtered[adata_filtered.obs['labels'].isin(valid_cell_types)].copy()
            
            # Final check: ensure reasonable proportion of cells across batches
            final_batch_counts = adata_filtered.obs['batch_id'].value_counts()
            batch_proportions = final_batch_counts / len(adata_filtered)
            
            if batch_proportions.min() >= 0.05:  # Each batch should have at least 5% of cells
                filtered_datasets[tissue_name] = adata_filtered
                logger.info(f"Kept {tissue_name}: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes, {len(valid_batches)} batches")
            else:
                logger.info(f"Skipping {tissue_name}: poor batch distribution after filtering")
        
        logger.info(f"Selected {len(filtered_datasets)} datasets out of {len(tissue_datasets)}")
        return filtered_datasets

    def merge_datasets(self, filtered_datasets):
        """Merge filtered datasets into a single AnnData object"""
        logger.info("Merging datasets...")
        
        if not filtered_datasets:
            raise ValueError("No datasets to merge after filtering")
        
        # Concatenate all datasets
        merged_adata = ad.concat(list(filtered_datasets.values()), axis=0, join='outer', fill_value=0)
        
        # Clean up obs columns to match the target format
        merged_adata.obs = merged_adata.obs[['batch_id', 'labels']].copy()
        
        # Add uns metadata
        merged_adata.uns['name'] = 'Mouse Cell Atlas'
        merged_adata.uns['organism'] = 'mouse'
        merged_adata.uns['tissue'] = 'multi-tissue'
        merged_adata.uns['year'] = 2018
        
        logger.info(f"Final merged dataset: {merged_adata.shape[0]} cells, {merged_adata.shape[1]} genes")
        logger.info(f"Unique batches: {merged_adata.obs['batch_id'].nunique()}")
        logger.info(f"Unique cell types: {merged_adata.obs['labels'].nunique()}")
        
        return merged_adata

    def process(self):
        """Run the complete processing pipeline"""
        try:
            # Extract tar files
            tissue_dirs = self.extract_tar_files()
            
            # Load metadata
            metadata_dict = self.load_metadata()
            
            # Process tissue data
            tissue_datasets = self.process_tissue_data(tissue_dirs, metadata_dict)
            
            # Filter datasets
            filtered_datasets = self.filter_datasets(tissue_datasets)
            
            # Merge datasets
            final_adata = self.merge_datasets(filtered_datasets)
            
            # Save final result
            output_path = Path(self.output_dir) / "mca_merged_data.h5ad"
            final_adata.write(output_path)
            logger.info(f"Saved final dataset to: {output_path}")
            
            # Print summary
            self.print_summary(final_adata)
            
            return final_adata
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise
        finally:
            # Clean up temporary directory in current working directory
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

    def print_summary(self, adata):
        """Print a summary of the final dataset"""
        print("\n" + "="*50)
        print("FINAL DATASET SUMMARY")
        print("="*50)
        print(f"📊 Dataset shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"\n🔍 Available columns in obs (Cell Metadata):")
        print(f"['batch_id', 'labels']")
        print(f"\n📋 Unique values in each obs column:")
        
        # Batch ID summary
        batch_counts = adata.obs['batch_id'].value_counts()
        print(f"  🔹 batch_id ({len(batch_counts)} unique): {list(batch_counts.index[:5])}{'...' if len(batch_counts) > 5 else ''} (showing first 5)")
        print(f"      Value counts: {dict(batch_counts.head(12))}")
        
        # Labels summary
        label_counts = adata.obs['labels'].value_counts()
        print(f"  🔹 labels ({len(label_counts)} unique): {list(label_counts.index)}")
        
        # Uns summary
        print(f"\n📦 Available keys in uns (Unstructured Data):")
        print(f"{list(adata.uns.keys())}")


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process MCA datasets')
    parser.add_argument('tar_file', help='Path to the main tar file containing tissue data')
    parser.add_argument('csv_file', help='Path to the metadata CSV file')
    parser.add_argument('--output_dir', help='Output directory for results', default='./mca_output')
    
    args = parser.parse_args()
    
    # Initialize and run processor
    processor = MCADataProcessor(args.tar_file, args.csv_file, args.output_dir)
    final_adata = processor.process()
    
    return final_adata


if __name__ == "__main__":
    final_adata = main()