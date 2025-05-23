from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import scipy.sparse as sp

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}


def cell_type_counts(datasets, field: str = "labels"):
    cell_counts = pd.DataFrame(
        [pd.Series.value_counts(ds.obs[field]) for ds in datasets]
    )
    cell_counts.index = [ds.uns["name"] for ds in datasets]
    cell_counts = cell_counts.T.fillna(0).astype(int).sort_index()

    styler = cell_counts.style
    styler = styler.format(lambda x: "" if x == 0 else x)
    styler = styler.set_properties(**{"width": "10em"})

    return styler


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def select_genes(
    data,
    threshold=0,
    atleast=10,
    yoffset=0.02,
    xoffset=5,
    decay=1,
    n=None,
    plot=True,
    markers=None,
    genes=None,
    figsize=(6, 3.5),
    markeroffsets=None,
    labelsize=10,
    alpha=1,
):
    if sp.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
            1 - zeroRate[detected]
        )
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.nanmean(
            np.where(data[:, detected] > threshold, np.log2(data[:, detected]), np.nan),
            axis=0,
        )

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    # lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = (
                zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            )
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        print("Chosen offset: {:.2f}".format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = (
            zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
        )

    if plot:
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + 0.1, 0.1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-x+{:.2f})+{:.2f}".format(
                    np.sum(selected), xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )
        else:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}".format(
                    np.sum(selected), decay, xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )

        plt.plot(x, y, linewidth=2)
        xy = np.concatenate(
            (
                np.concatenate((x[:, None], y[:, None]), axis=1),
                np.array([[plt.xlim()[1], 1]]),
            )
        )
        t = plt.matplotlib.patches.Polygon(xy, color="r", alpha=0.2)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=3, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of zero expression")
        else:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of near-zero expression")
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color="k")
                dx, dy = markeroffsets[num]
                plt.text(
                    meanExpr[i] + dx + 0.1,
                    zeroRate[i] + dy,
                    g,
                    color="k",
                    fontsize=labelsize,
                )

    return selected


def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    classes=None,  # ✅ New
    save_path=None,
    kl_divergence=None,         # ✅ New
    js_divergence=None,         # ✅ New
    knn_scvi_accuracy=None,     # ✅ New for KNN accuracies
    knn_umap_accuracy=None,     # ✅ New for KNN accuracies
    save_as_svg=False,
    knn_scGPT_accuracy=None,
    knn_uce_accuracy=None,

    **kwargs
):
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = None

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    y = np.array(y)

    # ✅ Determine classes
    if classes is not None:
        assert all(np.isin(np.unique(y), classes)), "Some labels in y are not in provided classes"
        classes = list(classes)
    elif label_order is not None:
        assert all(np.isin(np.unique(y), label_order)), "Some labels in y are not in label_order"
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = list(np.unique(y))

    # ✅ Handle colors
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
    
    point_colors = [colors.get(label, "#CCCCCC") for label in y]  # fallback color if missing

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # ✅ Draw cluster centers
    if draw_centers:
        centers = []
        for yi in classes:
            mask = y == yi
            if np.sum(mask) == 0:
                continue
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = [colors[c] for c in classes if c in colors]
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [], [], marker="s", color="w", markerfacecolor=colors[cls],
                ms=10, alpha=1, linewidth=0, label=cls, markeredgecolor="k",
            )
            for cls in classes if cls in colors
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)


        # ✅ Add divergence metrics
    if kl_divergence is not None and js_divergence is not None:
        add_divergence_text(fig, kl_divergence, js_divergence, fontsize=kwargs.get("fontsize", 12))

       # ✅ Add KNN accuracy metrics (new functionality)
    if knn_scvi_accuracy is not None and knn_umap_accuracy is not None:
        add_knn_accuracy_scvi_text(fig, knn_scvi_accuracy, knn_umap_accuracy, fontsize=kwargs.get("fontsize", 12))

    if knn_scGPT_accuracy is not None and knn_umap_accuracy is not None:
        add_knn_accuracy_scgpt_text(fig, knn_scGPT_accuracy, knn_umap_accuracy, fontsize=kwargs.get("fontsize", 12))
    
    if knn_uce_accuracy is not None and knn_umap_accuracy is not None:
        add_knn_accuracy_uce_text(fig, knn_uce_accuracy, knn_umap_accuracy, fontsize=kwargs.get("fontsize", 12))


    if save_path is not None and fig is not None:
        # Always save as PDF
        pdf_path = save_path if save_path.endswith(".pdf") else save_path + ".pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"Plot saved as: {pdf_path}")

        # Optionally save as SVG
        if save_as_svg:
            svg_path = save_path.replace(".pdf", ".svg") if save_path.endswith(".pdf") else save_path + ".svg"
            plt.savefig(svg_path, format="svg", bbox_inches="tight")
            print(f"Also saved as: {svg_path}")
    


    return ax



def get_colors_for(adata):
    """Get pretty colors for each class."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#7f7f7f",  # This is the grey one
              "#e377c2", "#bcbd22", "#17becf",

              "#0000A6", "#63FFAC", "#004D43", "#8FB0FF"]

    colors = dict(zip(adata.obs["labels"].value_counts().sort_values(ascending=False).index, colors))

    colors["Other"] = "#7f7f7f"

    assert all(l in colors for l in adata.obs["labels"].unique())

    return colors


def add_divergence_text(fig, kl, js, fontsize=12):
    """Adds KL and JS divergence values as a caption to the figure."""
    if fig is not None:
        fig.text(0.5, 0.02, f"KL Divergence = {kl:.4f}    |    JS Divergence = {js:.4f}", 
                 ha='center', fontsize=fontsize)

def add_knn_accuracy_scvi_text(fig, knn_scvi, knn_umap, fontsize=12):
    """Adds KNN accuracy values as a caption to the figure."""
    if fig is not None:
        fig.text(0.5, 0.02, f"KNN(scVI): {knn_scvi:.4f}    |    KNN(UMAP): {knn_umap:.4f}", 
                 ha='center', fontsize=fontsize)
        
def add_knn_accuracy_scgpt_text(fig, knn_scGpt, knn_umap, fontsize=12):
    """Adds KNN accuracy values as a caption to the figure."""
    if fig is not None:
        fig.text(0.5, 0.02, f"KNN(scGPT): {knn_scGpt:.4f}    |    KNN(UMAP): {knn_umap:.4f}", 
                 ha='center', fontsize=fontsize)

def add_knn_accuracy_uce_text(fig, knn_uce, knn_umap, fontsize=12):
    """Adds KNN accuracy values as a caption to the figure."""
    if fig is not None:
        fig.text(0.5, 0.02, f"KNN(UCE): {knn_uce:.4f}    |    KNN(UMAP): {knn_umap:.4f}", 
                 ha='center', fontsize=fontsize)