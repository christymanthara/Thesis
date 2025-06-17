import anndata
import pandas as pd
import os

def merge_and_save_h5ad(file1, file2, output_dir=".", batch_key="batch", label_column=None, use_basename=True):
    """
    Merges two h5ad files, adds batch labels, filters by label_column (if provided),
    and saves the merged file as <filename1>_<filename2>.h5ad in output_dir.
    
    Parameters:
    - file1: str, path to first h5ad file.
    - file2: str, path to second h5ad file.
    - output_dir: str, directory to save the merged h5ad file.
    - batch_key: str, name of the new obs column for batch labels.
    - label_column: str or None, name of obs column to filter shared values on (optional).
    - use_basename: bool, whether to extract just the file basename for batch labeling.
    """

    # Helper to extract filename without extension
    def extract_filename(path):
        return os.path.basename(path).rsplit('.h5ad', 1)[0]

    # Load datasets
    adata1 = anndata.read_h5ad(file1)
    adata2 = anndata.read_h5ad(file2)

    # Generate labels
    label1 = extract_filename(file1)
    label2 = extract_filename(file2)

    adata1.obs[batch_key] = pd.Categorical([label1] * adata1.n_obs)
    adata2.obs[batch_key] = pd.Categorical([label2] * adata2.n_obs)

    # Optional filtering
    if label_column and label_column in adata1.obs and label_column in adata2.obs:
        adata2 = adata2[adata2.obs[label_column].isin(adata1.obs[label_column])].copy()

    # Concatenate datasets
    full = anndata.concat([adata1, adata2], join="outer", label=batch_key, keys=[label1, label2])

    # Build output filename
    output_filename = f"{label1}_{label2}_filtered.h5ad"
    output_path = os.path.join(output_dir, output_filename)

    # Save
    full.write(output_path)
    print(f"Merged dataset saved to: {output_path}")
    
    
if __name__ == "__main__":
    merge_and_save_h5ad(
        file1="F:/Thesis/Datasets/baron_2016h.h5ad",
        # file2="F:/Thesis/Datasets/xin_2016.h5ad",
        file2="F:/Thesis/muraro_transformed.h5ad",
        label_column="labels",
    )