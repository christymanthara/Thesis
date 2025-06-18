import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from plotting_with_knn import compute_tsne_embeddings
from scripts.data_utils import preprocessing
# from batch-effect-analysis import UCE.compute_uce_embeddings
from scripts.compute_embeddings_scvi_scanvi_uce import compute_embeddings_scvi_scanvi_uce as ce

from scripts.test_harmony_cells import run_harmony_correction_simple, run_harmony_correction
from scripts import knn_plot_test

from anndata import AnnData
if __name__ == "__main__":
    # 1.Loading and preprocessing the datasets
    combined_data = preprocessing.load_and_preprocess_multi_embedder(
        
    file1="F:/Thesis/Datasets/baron_2016h.h5ad", 
    # file2="F:/Thesis/muraro_transformed.h5ad",
    file2="F:/Thesis/Datasets/xin_2016.h5ad",
    save=False,          # Saves all files
    split_output=False   # Getting the combined AnnData object
)
    # 2.Adding embeddings using scVI, scANVI, and UCE
    embedded_adata = ce(combined_data)
    
    # 2.5 Add harmony embedding
    adata_harmony = run_harmony_correction_simple(embedded_adata, batch_key="source")
    
    # 3.Run the knn plot test
    knn_plot_test.compute_knn_tsne_all(adata_harmony, skip_preprocessing=True, n_jobs=1)
