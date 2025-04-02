from sklearn.manifold import TSNE
import umap
import pavlin_alignment
import umap_plot
import matplotlib.pyplot as plt

def show_batch_effects(file1, file2, visual="tsne"):
    """
    Calls either TSNE or UMAP function based on the 'visual' parameter.
    """
    if visual.lower() == "tsne":
        pavlin_alignment.tsne_pavlin(file1, file2)
    elif visual.lower() == "umap":
        umap_plot.process_and_plot_umap(file1, file2)
    else:
        raise ValueError("Invalid visual parameter. Choose 'tsne' or 'umap'.")
    

if __name__ == "__main__":
    show_batch_effects("../datasets/baron_2016h.h5ad", "../datasets/xin_2016.h5ad", "tsne")
    show_batch_effects("../datasets/baron_2016h.h5ad", "../datasets/xin_2016.h5ad", "umap")