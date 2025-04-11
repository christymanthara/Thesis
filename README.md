# Thesis Repository: Batch Effect Correction in Single-Cell Genomics

This repository serves as the main repository for my thesis, which focuses on comparing batch-effect correction methods in single-cell genomics. It contains various scripts for processing, analyzing, and visualizing single-cell RNA sequencing (scRNA-seq) data using AnnData, Scanpy, openTSNE, and UMAP.

## Repository Structure

The repository is organized into the following submodules:

- **`papers/referenced`**: Contains the papers that are referenced in the thesis.
- **`scripts`**: Includes the implementations of the methods discussed in the referenced papers.
- **`datasets`**: Contains the datasets used for analysis in the thesis.
- **`benchmark/papers`**: Includes the papers that will be analyzed for benchmarking purposes.

## Cloning the Repository

To clone this repository along with its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/christymanthara/Thesis.git
```

To update all submodules to the latest commit, use the following command:

```bash
git submodule update --remote
```

For datasets, use the link below:
[Hemberg Lab scRNA-seq Datasets - Human Pancreas](https://hemberg-lab.github.io/scRNA.seq.datasets/human/pancreas/)

The datasets should be downloaded using the `download.sh` file in the datasets folder. Simply navigate to the datasets directory and run:

```bash
cd datasets
./download.sh
```
## Key Concepts

### Batch Effect Visualization
Batch effects are a driving factor of variation between datasets. The t-SNE visualizations show pairs of datasets that share cell types. It would be expected that cells from the reference data (blue) would mix with cells in secondary datasets (orange). Instead, t-SNE visualization clusters data according to the data source, highlighting the batch effect problem.

### Independent Cell Mapping
The pipeline is designed to map secondary datasets into a reference embedding space by treating each data point independently. This approach intentionally disregards interactions present within the secondary dataset and prevents the formation of clusters that would be specific to the secondary data. By doing so, it enables direct comparison of cell populations across different studies within a unified reference space.

## Scripts Overview

### Data Processing Scripts

#### `csv_h5ad.py`
Converts .csv.gz files extracted from a .tar archive into AnnData (.h5ad) format.
- Extracts .csv.gz files from a .tar archive.
- Converts expression data to a sparse matrix format.
- Assigns species labels (human/mouse) based on filenames.
- Saves the processed data as .h5ad files.

#### `data_stats.py`
Prints key statistics about an AnnData (.h5ad) dataset.
- Displays dataset shape (Cells × Genes).
- Reports the number of genes before and after filtering.
- Summarizes cell and gene metadata.
- Lists unique labels in the observation metadata (if available).

### Visualization Pipelines

#### Batch Effect Pipeline 1: Concatenated Datasets

##### `show_batch_effect_1_concatenated_datasets.py`
Visualizes batch effects using t-SNE or UMAP.
- Calls `pavlin_alignment.py` for t-SNE visualization.
- Calls `umap_plot.py` for UMAP visualization.
- Allows switching between t-SNE and UMAP methods.

##### `pavlin_alignment.py`
Performs t-SNE visualization for comparing two datasets.
- Loads and preprocesses two .h5ad datasets.
- Computes t-SNE embeddings using openTSNE with multiscale perplexities.
- Saves the t-SNE plot as a PDF.

##### `umap_plot.py`
Computes and plots UMAP embeddings for two datasets.
- Loads and preprocesses two .h5ad datasets.
- Computes UMAP projections.
- Saves the UMAP plot as a PDF.

#### Batch Effect Pipeline 2: Reference Mapping

The pipeline consists of three main scripts that should be executed in the following order:

##### 1. `pavlin_preprocess_plot_on_top.py`
Preprocesses gene expression data and computes embeddings.
- Normalizes and log-transforms the data.
- Selects genes based on variance.
- Computes PCA and t-SNE embeddings.
- Saves the processed data as .h5ad files with different gene selections (250, 3000, and all genes).

##### 2. `multiplot_transform_and_plot.py`
Transforms new scRNA-seq datasets into an existing embedding space.
- Loads reference and new datasets.
- Finds shared genes between datasets.
- Creates embeddings for different gene subset sizes (250, 1000, all).
- Visualizes the embeddings by overlaying new data on reference data.
- **Importantly:** Each cell from the secondary dataset is mapped independently to the reference space.

##### 3. `multiplot_tsne_pca_plot_on_top.py`
Creates publication-ready visualizations comparing PCA and t-SNE results.
- Generates multi-panel figures with PCA and t-SNE visualizations.
- Automatically adapts layout based on number of input files.
- Adds alphabetical labels to panels.
- Outputs a PDF with separate pages for PCA and t-SNE visualizations.

## Usage Examples

### Batch Effect Pipeline 1

```bash
python scripts/csv_h5ad.py
python scripts/data_stats.py
python scripts/pavlin_alignment.py # For t-SNE
python scripts/umap_plot.py # For UMAP
python scripts/show_batch_effect_1_concatenated_datasets.py
```

### Batch Effect Pipeline 2

```bash
# Step 1: Preprocess the reference dataset and compute embeddings
python scripts/pavlin_preprocess_plot_on_top.py

# Step 2: Transform new dataset into the reference embedding space based on tsne
python scripts/multiplot_transform_and_plot_tsne.py


# Step 2.1: Transform new dataset into the reference embedding space based on umap
python scripts/multiplot_transform_and_plot_umap.py


# Step 3: Generate visualizations of the embeddings
python scripts/multiplot_tsne_pca_plot_on_top.py
```

## Data Organization

The scripts assume the following file structure:
```
project/
├── scripts/
│   ├── csv_h5ad.py
│   ├── data_stats.py
│   ├── pavlin_alignment.py
│   ├── pavlin_preprocess_plot_on_top.py
│   ├── umap_plot.py
│   ├── show_batch_effect_1_concatenated_datasets.py
│   ├── multiplot_transform_and_plot.py
│   ├── multiplot_tsne_pca_plot_on_top.py
│   └── utils.py
└── Datasets/
    ├── baron_2016h.h5ad
    └── xin_2016.h5ad
```

## 🧬 Bioinformatics Table

<!-- TABLE_START -->

| tissue   |   source 1 |   source 2 |   Pavlin | Pavlin_tsne                                                                                  | Pavlin_umap   |   fastscbatch |
|----------|------------|------------|----------|----------------------------------------------------------------------------------------------|---------------|---------------|
| kidney   |        nan |        nan |     0.88 | [baron_2016h-xin_2016-plotontop_tsne.pdf](./visuals/baron_2016h-xin_2016-plotontop_tsne.pdf) |               |           nan |

<!-- TABLE_END -->


## Notes

- The scripts use a custom `utils.py` module that needs to be available in the same directory.
- The default datasets used are from Baron et al. 2016 and Xin et al. 2016 (pancreatic islet cell datasets).
- For large datasets, the computation might be memory-intensive and time-consuming.
- The independent mapping approach helps identify corresponding cell types across studies without being influenced by batch effects or dataset-specific cluster formations.
- **Important:** Change normalization from million to 10,000 when processing datasets.