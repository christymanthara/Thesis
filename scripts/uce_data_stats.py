import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import re

def check_anndata_format(file_path):
    """
    Check if AnnData object follows scRNA-seq guidelines:
    1. Check if .X contains count-like data (integers, sparse)
    2. Check if .var_names are gene symbols rather than ENSEMBLIDs
    
    Parameters:
    -----------
    file_path : str
        Path to the AnnData h5ad file
    """
    print(f"Loading AnnData file from: {file_path}")
    try:
        adata = ad.read_h5ad(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    print(f"\n📊 Dataset Shape: {adata.shape} (Cells × Genes)")
    
    # Check 1: Is .X likely to be a count matrix?
    print("\n🔍 Checking if .X contains count data:")
    
    # Get a sample of the data
    if sp.issparse(adata.X):
        print("  - ✓ Data is stored as a sparse matrix (good for count data)")
        # Get a dense sample for checking
        sample = adata.X[:min(100, adata.shape[0]), :min(100, adata.shape[1])].toarray()
    else:
        print("  - ℹ️ Data is stored as a dense matrix")
        sample = adata.X[:min(100, adata.shape[0]), :min(100, adata.shape[1])]
    
    # Check if values are integers or close to integers
    if np.all(np.equal(np.round(sample), sample)):
        print("  - ✓ All sampled values are integers (consistent with count data)")
    else:
        non_int_percent = np.mean(~np.isclose(np.round(sample), sample)) * 100
        if non_int_percent < 1:
            print(f"  - ⚠️ Most values are integers, but {non_int_percent:.2f}% are not")
            print("      This could be raw counts with some normalization applied")
        else:
            print(f"  - ❌ {non_int_percent:.2f}% of sampled values are not integers")
            print("      This suggests the data has been normalized or transformed")
    
    # Check for sparsity (common in count data)
    sparsity = 1 - (np.count_nonzero(sample) / sample.size)
    print(f"  - Data sparsity: {sparsity:.2%} zeros (typical scRNA-seq is >90% sparse)")
    
    # Check value range
    min_val = np.min(sample)
    max_val = np.max(sample)
    print(f"  - Value range: [{min_val}, {max_val}]")
    
    if min_val < 0:
        print("  - ❌ Negative values detected - these are NOT raw counts")
    
    # Determine if likely counts or normalized
    if (np.all(np.equal(np.round(sample), sample)) and 
        min_val >= 0 and 
        sparsity > 0.8):
        print("  - ✅ Data appears to be count data")
    elif min_val >= 0 and np.all(np.equal(np.round(sample), sample)):
        print("  - ⚠️ Data might be count data but has unusual properties")
    else:
        print("  - ❌ Data is likely normalized or transformed, NOT raw counts")
    
    # Check 2: Are .var_names gene symbols rather than ENSEMBLIDs?
    print("\n🧬 Checking if .var_names contains gene symbols rather than ENSEMBLIDs:")
    
    var_names = adata.var_names.tolist()
    
    # Sample a few gene names
    sample_size = min(10, len(var_names))
    print(f"  Sample of gene names: {', '.join(var_names[:sample_size])}")
    
    # Check for ENSEMBL ID patterns (e.g., ENSG00000139618)
    ensembl_pattern = re.compile(r'^ENS[A-Z]*\d{11}')
    ensembl_count = sum(1 for name in var_names if ensembl_pattern.match(name))
    ensembl_percent = ensembl_count / len(var_names) * 100
    
    # Common gene symbol patterns (uppercase letters, often with numbers)
    symbol_pattern = re.compile(r'^[A-Z0-9]+$')
    symbol_count = sum(1 for name in var_names if symbol_pattern.match(name))
    symbol_percent = symbol_count / len(var_names) * 100
    
    print(f"  - ENSEMBL-like IDs: {ensembl_count} ({ensembl_percent:.2f}%)")
    print(f"  - Gene symbol-like: {symbol_count} ({symbol_percent:.2f}%)")
    
    if ensembl_percent > 80:
        print("  - ❌ Most gene identifiers appear to be ENSEMBL IDs, not gene symbols")
    elif symbol_percent > 70:
        print("  - ✅ Most gene identifiers appear to be gene symbols")
    else:
        print("  - ⚠️ Gene identifiers don't clearly match ENSEMBL or symbol patterns")
        print("      Manual inspection recommended")
    
    # Additional info that might be helpful
    print("\n📈 Additional information:")
    print(f"  - Number of cells: {adata.n_obs}")
    print(f"  - Number of genes: {adata.n_vars}")
    if 'highly_variable' in adata.var:
        n_hvg = adata.var['highly_variable'].sum()
        print(f"  - Highly variable genes: {n_hvg}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    # You can replace this with the path to your .h5ad file
    # file_path = input("Enter the path to your .h5ad file: ")
    # check_anndata_format(file_path)
    check_anndata_format("F:/Thesis/Datasets/baron_2016h.h5ad")