import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def plot_embeddings_with_stats(adata, stats_df=None, figsize_per_plot=(6, 5), 
                              color_by='labels', palette='tab10', 
                              point_size=1, alpha=0.7, ncols=3):
    """
    Plot all embeddings from adata.obsm with statistics at the bottom.
    
    Parameters:
    -----------
    adata : AnnData object
        Annotated data object containing embeddings in obsm
    stats_df : pandas.DataFrame, optional
        DataFrame containing statistics to display at bottom
    figsize_per_plot : tuple
        Size for each individual embedding plot
    color_by : str
        Column name in adata.obs to color points by
    palette : str or list
        Color palette for the plots
    point_size : float
        Size of scatter plot points
    alpha : float
        Transparency of points
    ncols : int
        Number of columns for embedding plots
    """
    
    # Get available embeddings from obsm
    available_embeddings = list(adata.obsm.keys())
    n_embeddings = len(available_embeddings)
    
    if n_embeddings == 0:
        print("No embeddings found in adata.obsm")
        return
    
    # Calculate grid dimensions for embeddings
    nrows_embed = (n_embeddings + ncols - 1) // ncols
    
    # Determine if we need space for stats
    add_stats_space = stats_df is not None
    
    # Calculate total figure size
    total_width = ncols * figsize_per_plot[0]
    embed_height = nrows_embed * figsize_per_plot[1]
    stats_height = 3 if add_stats_space else 0
    total_height = embed_height + stats_height
    
    # Create figure with appropriate layout
    if add_stats_space:
        fig = plt.figure(figsize=(total_width, total_height))
        # Create gridspec with height ratios
        gs = fig.add_gridspec(2, 1, height_ratios=[embed_height, stats_height], 
                             hspace=0.3)
        
        # Embedding plots section
        embed_gs = gs[0].subgridspec(nrows_embed, ncols, hspace=0.4, wspace=0.3)
        
        # Stats section
        stats_ax = fig.add_subplot(gs[1])
    else:
        fig, axes = plt.subplots(nrows_embed, ncols, 
                                figsize=(total_width, embed_height))
        if n_embeddings == 1:
            axes = [axes]
        elif nrows_embed == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
    
    # Color mapping
    if color_by in adata.obs.columns:
        color_data = adata.obs[color_by]
        if color_data.dtype == 'object' or color_data.dtype.name == 'category':
            # Categorical coloring
            unique_categories = color_data.unique()
            if isinstance(palette, str):
                colors = sns.color_palette(palette, len(unique_categories))
            else:
                colors = palette[:len(unique_categories)]
            color_map = dict(zip(unique_categories, colors))
            point_colors = [color_map[cat] for cat in color_data]
        else:
            # Continuous coloring
            point_colors = color_data
            color_map = None
    else:
        # Default single color
        point_colors = 'blue'
        color_map = None
    
    # Plot each embedding
    for i, embedding_key in enumerate(available_embeddings):
        embedding_data = adata.obsm[embedding_key]
        
        # Determine which dimensions to plot (first 2)
        if embedding_data.shape[1] >= 2:
            x_data = embedding_data[:, 0]
            y_data = embedding_data[:, 1]
        else:
            print(f"Warning: {embedding_key} has only {embedding_data.shape[1]} dimension(s)")
            continue
        
        # Create subplot
        if add_stats_space:
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(embed_gs[row, col])
        else:
            ax = axes[i] if n_embeddings > 1 else axes[0]
        
        # Create scatter plot
        if isinstance(point_colors, list) and color_map is not None:
            # Categorical coloring
            for category, color in color_map.items():
                mask = color_data == category
                ax.scatter(x_data[mask], y_data[mask], 
                          c=[color], s=point_size, alpha=alpha, 
                          label=category, rasterized=True)
        else:
            # Single color or continuous coloring
            scatter = ax.scatter(x_data, y_data, c=point_colors, 
                               s=point_size, alpha=alpha, rasterized=True)
            if not isinstance(point_colors, str):
                plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        # Formatting
        ax.set_title(f'{embedding_key}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(f'{embedding_key}_1', fontsize=10)
        ax.set_ylabel(f'{embedding_key}_2', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend for categorical data
        if color_map is not None and len(unique_categories) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=8, markerscale=3)
    
    # Hide empty subplots
    if add_stats_space:
        total_subplots = nrows_embed * ncols
        for j in range(n_embeddings, total_subplots):
            row, col = divmod(j, ncols)
            ax = fig.add_subplot(embed_gs[row, col])
            ax.set_visible(False)
    else:
        if n_embeddings > 1:
            for j in range(n_embeddings, len(axes)):
                axes[j].set_visible(False)
    
    # Add statistics section if provided
    if add_stats_space and stats_df is not None:
        stats_ax.axis('off')
        
        # Create a nice table display
        stats_ax.text(0.5, 0.95, 'Dataset Statistics', 
                     transform=stats_ax.transAxes, 
                     fontsize=16, fontweight='bold', 
                     ha='center', va='top')
        
        # Display basic info
        basic_info = [
            f"📊 Dataset Shape: {adata.shape} (Cells × Genes)",
            f"🧬 Total Genes: {adata.n_vars:,}",
            f"🔬 Total Cells: {adata.n_obs:,}",
            f"🏷️ Cell Types: {len(adata.obs[color_by].unique()) if color_by in adata.obs.columns else 'N/A'}",
            f"📦 Embeddings Available: {len(available_embeddings)}"
        ]
        
        y_pos = 0.75
        for info in basic_info:
            stats_ax.text(0.1, y_pos, info, transform=stats_ax.transAxes,
                         fontsize=11, va='top')
            y_pos -= 0.12
        
        # If additional stats_df is provided, display it
        if stats_df is not None and not stats_df.empty:
            stats_ax.text(0.6, 0.75, 'Additional Statistics:', 
                         transform=stats_ax.transAxes,
                         fontsize=12, fontweight='bold', va='top')
            
            y_pos = 0.65
            for idx, row in stats_df.iterrows():
                for col in stats_df.columns:
                    stat_text = f"{col}: {row[col]}"
                    stats_ax.text(0.6, y_pos, stat_text, 
                                 transform=stats_ax.transAxes,
                                 fontsize=10, va='top')
                    y_pos -= 0.08
                y_pos -= 0.05  # Extra space between rows
    
    plt.tight_layout()
    return fig

# Example usage function for your specific dataset
def plot_baron_xin_embeddings(adata, stats_csv_path=None):
    """
    Specific function for Baron 2016h and Xin 2016 dataset visualization.
    
    Parameters:
    -----------
    adata : AnnData object
        Your Baron/Xin dataset
    stats_csv_path : str, optional
        Path to CSV file containing additional statistics
    """
    
    # Load stats if provided
    stats_df = None
    if stats_csv_path:
        try:
            stats_df = pd.read_csv(stats_csv_path)
        except Exception as e:
            print(f"Could not load stats CSV: {e}")
    
    # Create the plot
    fig = plot_embeddings_with_stats(
        adata=adata,
        stats_df=stats_df,
        color_by='labels',  # Color by cell types
        palette='Set3',     # Nice palette for cell types
        point_size=3,
        alpha=0.6,
        ncols=2,           # 2 columns for better layout
        figsize_per_plot=(7, 6)
    )
    
    # Add main title
    fig.suptitle('Baron 2016h and Xin 2016 Dataset - Embedding Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    return fig

# Simple usage example:
# fig = plot_baron_xin_embeddings(adata, 'path/to/your/stats.csv')
# plt.show()

if __name__ == "__main__":
    import anndata
    
    # Load your AnnData object
    adata = anndata.read_h5ad("baron_2016hxin_2016_uce_adata_X_scvi_X_scanvi_X_scGPT_test.h5ad")
    
    # Optionally load stats CSV if available
    stats_csv_path = "benchmark_results_20250605_222551_mean_scores.csv"  # Set to None if not available
    fig = plot_baron_xin_embeddings(adata, stats_csv_path)
    
    
    output_filename = "baron_xin_2016_embedding_analysis_from_csv.pdf"
    fig.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()  # Display the plot