import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pavlin_alignment_re import tsne_pavlin_re
from umap_plot import process_and_plot_umap
from Datasets.datasets import datasets  # External dataset dictionary
from Datasets.download_by_category import download_files

def show_batch_effects(file1, file2, visual="tsne"):
    """
    Visualizes batch effects between two single-cell datasets using TSNE or UMAP.

    Args:
        file1 (str): Path to the first dataset (.h5ad)
        file2 (str): Path to the second dataset (.h5ad)
        visual (str): Visualization method ('tsne' or 'umap')
    """
    if visual.lower() == "tsne":
        tsne_pavlin_re(file1, file2)
    elif visual.lower() == "umap":
        process_and_plot_umap(file1, file2)
    else:
        raise ValueError("Invalid visual parameter. Choose 'tsne' or 'umap'.")

def resolve_file_paths(tissue, custom_files=None):
    """
    Resolves dataset file paths for a given tissue and optional filenames.

    Args:
        tissue (str): Tissue name (e.g., "pancreas", "brain")
        custom_files (list[str], optional): Two specific filenames to use

    Returns:
        tuple[str, str]: Full paths to the two dataset files
    """
    tissue = tissue.lower()

    if tissue not in datasets:
        raise ValueError(f"❌ Tissue '{tissue}' not found. Available options: {', '.join(datasets.keys())}")
    
    file_urls = datasets[tissue]
    file_names = [os.path.basename(url) for url in file_urls]
    data_dir = os.path.join("temp", "Datasets", tissue)
    
    # Automatically download datasets if missing
    download_files(tissue)

    file_urls = datasets[tissue]
    file_names = [os.path.basename(url) for url in file_urls]
    data_dir = os.path.join("temp", "Datasets", tissue)

    if custom_files:
        if len(custom_files) != 2:
            raise ValueError("❌ Please provide exactly two filenames.")
        for f in custom_files:
            if f not in file_names:
                raise ValueError(f"❌ File '{f}' not found in the dataset list for tissue '{tissue}'")
        return os.path.join(data_dir, custom_files[0]), os.path.join(data_dir, custom_files[1])

    # If no custom files provided
    if len(file_names) < 2:
        raise ValueError(f"❌ Not enough datasets for tissue '{tissue}' to compare.")
    elif len(file_names) > 2:
        print(f"⚠️ Multiple datasets found for tissue '{tissue}':")
        for f in file_names:
            print(f"  - {f}")
        print("\nPlease rerun the script with two filenames (and optionally a method) like:")
        print(f"  python show_batch_effect_1_concatenated_datasets.py {tissue} <file1> <file2> [tsne|umap]")
        sys.exit(0)
    else:
        return os.path.join(data_dir, file_names[0]), os.path.join(data_dir, file_names[1])

if __name__ == "__main__":
    # --------------------------------------
    # CLI Usage:
    #   python show_batch_effect_1_concatenated_datasets.py <tissue>
    #   python show_batch_effect_1_concatenated_datasets.py <tissue> tsne
    #   python show_batch_effect_1_concatenated_datasets.py <tissue> <file1> <file2>
    #   python show_batch_effect_1_concatenated_datasets.py <tissue> <file1> <file2> umap
    # --------------------------------------

    args = sys.argv[1:]

    if len(args) not in [1, 2, 3, 4]:
        print("Usage:")
        print("  python show_batch_effect_1_concatenated_datasets.py <tissue>")
        print("  python show_batch_effect_1_concatenated_datasets.py <tissue> tsne")
        print("  python show_batch_effect_1_concatenated_datasets.py <tissue> <file1> <file2>")
        print("  python show_batch_effect_1_concatenated_datasets.py <tissue> <file1> <file2> <tsne|umap>")
        sys.exit(1)

    tissue = args[0]
    custom_files = None
    visual_method = "tsne"  # Default method

    # Infer args based on count
    if len(args) == 2 and args[1].lower() in ["tsne", "umap"]:
        visual_method = args[1]
    elif len(args) == 3:
        custom_files = args[1:3]
    elif len(args) == 4:
        custom_files = args[1:3]
        visual_method = args[3]

    try:
        file1, file2 = resolve_file_paths(tissue, custom_files)
        show_batch_effects(file1, file2, visual=visual_method)
    except Exception as e:
        print(e)
