import matplotlib.pyplot as plt
import pandas as pd
import os

def create_results_table(results_table, main_source: str, ref_source: str, base_filename, reference_file=None):
    """
    Create a summary table of all KNN results and save as PDF.
    
    Parameters:
    -----------
    results_table : dict
        Dictionary containing results for each embedding
    main_source : str
        Name of the main source dataset
    ref_source : str
        Name of the reference source dataset
    base_filename : str
        Base filename for output
    reference_file : str, optional
        Reference file path for naming
    """
    
    # Extract query transfer accuracies from results_table
    query_accuracies = {}
    for embedding, results in results_table.items():
        # Extract the query transfer accuracy (remove any formatting)
        query_transfer = results['Query Transfer']
        # Remove any extra formatting if present
        if isinstance(query_transfer, str):
            query_accuracies[embedding] = query_transfer
        else:
            query_accuracies[embedding] = f"{query_transfer:.3f}"
    
    # Create DataFrame with the desired structure
    # First column will be "Source Comparison", other columns will be embeddings
    data = {
        'Source Comparison': [f"{main_source} vs {ref_source}"]
    }
    
    # Add each embedding as a column
    for embedding, accuracy in query_accuracies.items():
        data[embedding] = [accuracy]
    
    df = pd.DataFrame(data)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 2), 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create title
    ref_name = os.path.splitext(os.path.basename(reference_file))[0] if reference_file else base_filename
    title = f"KNN Classification Results - Query Transfer Accuracy\n{main_source} → {ref_source}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.9)
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0.1, 1, 0.6])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Color the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the first column (Source Comparison)
    table[(1, 0)].set_facecolor('#E8F5E8')
    table[(1, 0)].set_text_props(weight='bold')
    
    # Color the data cells
    for j in range(1, len(df.columns)):
        table[(1, j)].set_facecolor('#F5F5F5')
    
    # Add explanatory text
    explanation = ("Table shows query transfer accuracy for each embedding method\n"
                  f"Training: {ref_source} → Testing: {main_source}")
    
    fig.text(0.5, 0.05, explanation, ha='center', fontsize=10,
             style='italic', wrap=True)
    
    # Save table as PDF
    table_output = f"knn_results_summary_final_{base_filename}.pdf"
    plt.savefig(table_output, dpi=300, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    print(f"Saved results table as {table_output}")
    plt.close()
    
    # Also save as CSV for easy access
    csv_output = f"knn_results_summary_{base_filename}.csv"
    df.to_csv(csv_output, index=False)
    print(f"Saved results table as {csv_output}")
    
    return df