import logging
import os
import sys
import time
import anndata
import torch
import pandas as pd
import numpy as np
from tqdm import trange
import scipy.sparse

# Set up logging
def setup_logging(log_file=None, level=logging.INFO):
    """
    Sets up logging configuration.
    
    Parameters:
        log_file (str): Path to log file. If None, logs to console only.
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    
    Returns:
        logger: Configured logger object
    """
    logger = logging.getLogger("FastSCBatch")
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Add logging to your functions
def read_h5ad_data(file_path, batch_key="Batch", logger=None):
    """
    Reads an h5ad file and returns expression matrix, batch labels, and AnnData object.
    
    Parameters:
        file_path (str): Path to the .h5ad file.
        batch_key (str): Name of the column in obs containing batch information.
        logger: Logger object for logging progress.
    
    Returns:
        rawdat (ndarray): Gene expression matrix (cells x genes).
        bat (Series): Batch labels.
        adata (AnnData): Full AnnData object.
    """
    if logger:
        logger.info(f"Loading H5AD file from: {file_path}")
    
    adata = anndata.read_h5ad(file_path)
    
    if logger:
        logger.info(f"Loaded AnnData object with shape: {adata.shape}")
        logger.info(f"Available obs keys: {list(adata.obs.columns)}")
    
    # Check if batch_key exists
    if batch_key not in adata.obs:
        available_keys = list(adata.obs.columns)
        error_msg = f"Batch key '{batch_key}' not found in obs. Available keys: {available_keys}"
        if logger:
            logger.error(error_msg)
        raise KeyError(error_msg)
    
    # Convert sparse matrix to dense if needed
    if logger:
        logger.info("Converting expression matrix to dense array if needed...")
    
    if scipy.sparse.issparse(adata.X):
        if logger:
            logger.info("Expression matrix is in sparse format, converting to dense...")
        rawdat = adata.X.toarray()
    else:
        rawdat = adata.X
    
    bat = adata.obs[batch_key].copy().iloc[:, 0] if isinstance(adata.obs[batch_key], pd.DataFrame) else adata.obs[batch_key]
    
    if logger:
        batch_counts = bat.value_counts().to_dict()
        logger.info(f"Batch distribution: {batch_counts}")
        logger.info(f"Expression matrix shape: {rawdat.shape}")
    
    return rawdat, bat, adata

def compute_reference_corr(rawdat, bat, logger=None):
    """
    Computes modified correlation matrix by aligning batch-specific correlations to a reference batch.
    
    Parameters:
        rawdat (ndarray): Gene expression matrix (cells x genes).
        bat (Series): Batch labels.
        logger: Logger object for logging progress.
    
    Returns:
        corr (Tensor): Transformed correlation matrix.
    """
    if logger:
        logger.info("Computing raw correlation matrix...")
        start_time = time.time()
    
    rawcorr = torch.corrcoef(torch.tensor(rawdat))
    
    if logger:
        logger.info(f"Raw correlation matrix shape: {rawcorr.shape}")
        logger.info(f"Computing reference correlation took {time.time() - start_time:.2f} seconds")
    
    mx = bat.value_counts().idxmax()
    
    if logger:
        logger.info(f"Using batch {mx} as reference (largest batch)")
    
    ref_block = rawdat[bat == mx]
    
    if logger:
        logger.info(f"Reference block shape: {ref_block.shape}")
        logger.info("Computing reference correlation...")
        start_time = time.time()
    
    ref_corr = torch.corrcoef(torch.tensor(ref_block))
    
    if logger:
        logger.info(f"Reference correlation computed in {time.time() - start_time:.2f} seconds")
    
    vec = ref_corr.reshape(-1).sort()[0]
    dattype = bat.unique()
    
    if logger:
        logger.info(f"Found {len(dattype)} unique batch types: {dattype}")
        logger.info("Transforming correlation blocks...")
    
    corr = rawcorr.clone()
    
    for i in dattype:
        for j in dattype:
            if i == mx and j == mx:
                continue
                
            if logger:
                logger.debug(f"Processing batch pair ({i}, {j})...")
            
            block = corr[bat == i][:, bat == j]
            block_ = block.reshape(-1)
            idx = block_.argsort()
            pos = torch.zeros_like(idx, dtype=torch.float)
            pos[idx] = torch.arange(len(idx), dtype=torch.float)
            pos = pos / len(block_) * len(vec)
            pos = torch.maximum(pos.int() - 1, torch.zeros_like(pos, dtype=torch.int))
            block_ = vec[pos]
            block_ = block_.reshape(block.shape)
            msk = (torch.tensor(bat == i, dtype=torch.bool).unsqueeze(1) &
                   torch.tensor(bat == j, dtype=torch.bool).unsqueeze(0))
            corr.masked_scatter_(msk, block_)
    
    if logger:
        logger.info("Smoothing correlation matrix...")
    
    # Smooth the matrix
    for s in range(1):
        if logger:
            logger.info(f"Smoothing iteration {s+1}...")
        
        batmsk = {i: (bat != i).to_numpy() for i in dattype}
        
        for i in trange(len(corr)):
            curbat = batmsk[bat[i]]
            block = corr[i, curbat]
            idx = block.argsort().int()
            pos = torch.zeros_like(idx, dtype=torch.float)
            pos[idx] = torch.arange(len(idx), dtype=torch.float)
            pos = pos / len(block) * len(vec)
            pos = torch.maximum(pos.int() - 1, torch.zeros_like(pos, dtype=torch.int))
            block = vec[pos]
            msk = torch.zeros_like(corr, dtype=torch.bool)
            msk[i] |= torch.tensor(curbat, dtype=torch.bool)
            corr.masked_scatter_(msk, block)
    
    # Symmetrize
    if logger:
        logger.info("Symmetrizing correlation matrix...")
    
    corr = (corr + corr.T) / 2
    
    if logger:
        logger.info("Correlation matrix computation complete.")
    
    return corr

def save_correlation_matrix(corr, out_path, logger=None):
    """
    Saves correlation matrix to a CSV file.
    
    Parameters:
        corr (Tensor): Correlation matrix.
        out_path (str): Output path for CSV file.
        logger: Logger object for logging progress.
    """
    if logger:
        logger.info(f"Saving correlation matrix to {out_path}...")
        start_time = time.time()
    
    pd.DataFrame(corr.numpy()).to_csv(out_path)
    
    if logger:
        logger.info(f"Correlation matrix saved in {time.time() - start_time:.2f} seconds")

def run_fastscbatch_pipeline(h5ad_path, corr_path, output_path, solver_fn, batch_key="Batch", group_key="Group", logger=None):
    """
    Runs the full FastSCBatch correction pipeline.
    
    Parameters:
        h5ad_path (str): Path to input .h5ad file.
        corr_path (str): Path to correlation .csv file.
        output_path (str): Path to save corrected .h5ad file.
        solver_fn (callable): FastSCBatch solver function.
        batch_key (str): Name of the column in obs containing batch information. Default is "Batch".
        group_key (str): Name of the column in obs containing group/cell type information. Default is "Group".
        logger: Logger object for logging progress.
    """
    if logger:
        logger.info(f"Loading H5AD file for batch correction: {h5ad_path}")
    
    cell = anndata.read_h5ad(h5ad_path)
    
    if logger:
        logger.info(f"Loading batch information from column '{batch_key}'")
    
    batch = cell.obs[[batch_key]].copy()
    
    if group_key in cell.obs:
        if logger:
            logger.info(f"Loading group/cell type information from column '{group_key}'")
        ctype = cell.obs[[group_key]].copy()
    else:
        if logger:
            logger.info(f"Group/cell type column '{group_key}' not found, proceeding without it")
        ctype = None
    
    if logger:
        logger.info("Converting data to gene x cell format...")
    
    cells = cell.to_df().T
    
    if logger:
        logger.info(f"Loading correlation matrix from {corr_path}")
    
    corr = pd.read_csv(corr_path, index_col=0)
    corr.columns = cells.columns
    corr.index = cells.columns
    
    p, n = cells.shape
    
    if logger:
        logger.info(f"Data dimensions: {p} genes x {n} cells")
        logger.info("Running FastSCBatch solver...")
        logger.info(f"Parameters: p=0.3, k=50, EPOCHS=(0, 0, 500)")
        start_time = time.time()
    
    res = solver_fn(
        cells, corr, batch, p=0.3, k=50,
        lr=(0.0002, 0.0001, 0.0003), EPOCHS=(0, 0, 500), verbose=True
    )
    
    if logger:
        logger.info(f"FastSCBatch solver completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Creating corrected AnnData object...")
    
    adata = anndata.AnnData(X=res.T, obs=cell.obs, var=cell.var)
    
    if logger:
        logger.info(f"Saving corrected AnnData object to {output_path}")
    
    adata.write(output_path)
    
    if logger:
        logger.info("FastSCBatch pipeline completed successfully!")

# EXAMPLE USAGE:
if __name__ == "__main__":
    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"fastscbatch_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("Starting FastSCBatch pipeline")
    logger.info("="*80)
    
    # Example usage of the functions
    h5ad_path = "realdata/pancreas/converted_data.h5ad"
    corr_path = "sample_corr.csv"
    output_path = "output_corrected.h5ad"
    batch_column = "batch"  # Change to your actual batch column name
    
    try:
        logger.info(f"Processing H5AD file: {h5ad_path}")
        rawdat, bat, adata = read_h5ad_data(h5ad_path, batch_key=batch_column, logger=logger)
        
        logger.info("Computing reference correlation")
        corr = compute_reference_corr(rawdat, bat, logger=logger)
        
        logger.info(f"Saving correlation matrix to {corr_path}")
        save_correlation_matrix(corr, corr_path, logger=logger)
        
        logger.info("Running FastSCBatch pipeline")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from data_utils.fast_scBatch import solver
        run_fastscbatch_pipeline(h5ad_path, corr_path, output_path, solver, 
                                batch_key=batch_column, logger=logger)
        
        logger.info("FastSCBatch processing complete!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise