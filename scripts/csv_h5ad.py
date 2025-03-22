import tarfile
import os
import pandas as pd
import anndata as ad
import gzip
from scipy.sparse import csr_matrix

def extract_and_convert_tar(tar_path, extract_path="extracted_csv"):
    """
    Extracts a .tar file containing .csv.gz files, processes them into AnnData format, 
    and saves them as .h5ad files.

    Parameters:
    - tar_path: str, path to the .tar file.
    - extract_path: str, directory where extracted files should be stored (default: "extracted_csv").
    """

    # Ensure extraction directory exists
    os.makedirs(extract_path, exist_ok=True)

    # Extract tar file
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)

    # Find all .csv.gz files
    csv_gz_files = [f for f in os.listdir(extract_path) if f.endswith('.csv.gz')]

    # Process each file separately
    for file in csv_gz_files:
        file_path = os.path.join(extract_path, file)

        # Read .csv.gz
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f)

        # Separate numeric data and metadata
        metadata = df.select_dtypes(exclude=['number'])  # Non-numeric columns (e.g., sample info)
        expression_data = df.select_dtypes(include=['number'])  # Only numerical values

        # Convert expression data to a sparse matrix to reduce memory usage
        adata = ad.AnnData(csr_matrix(expression_data.values))
        adata.obs = metadata  # Add metadata back

        # Ensure unique observation names
        adata.obs_names_make_unique()

        # Determine species from filename
        species = "unknown"
        if "mouse" in file.lower():
            species = "mouse"
        elif "human" in file.lower():
            species = "human"

        # Save as .h5ad
        h5ad_path = os.path.join(extract_path, f"{file.replace('.csv.gz', '')}_{species}.h5ad")
        adata.write(h5ad_path)
        print(f"✅ Saved: {h5ad_path}")

if __name__ == "__main__":
    # Example usage
    tar_file_path = "../datasets/extras/GSE84133_RAW.tar"
    extract_and_convert_tar(tar_file_path)
