import matplotlib.pyplot as plt
import pandas as pd
import os

def create_results_table(results_table, base_filename, reference_file=None):
    """
    Create a summary table of all KNN results and save as PDF.

    Parameters:
    -----------
    results_table : dict
        Dictionary containing results for each embedding
    base_filename : str
        Base filename for output
    reference_file : str, optional
        Reference file path for naming
    """

    # Convert results to DataFrame
    df = pd.DataFrame.from_dict(results_table, orient='index')

    # Create figure for table
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4 + 2)))
    ax.axis('tight')
    ax.axis('off')

    # Create title
    ref_name = os.path.splitext(os.path.basename(reference_file))[0] if reference_file else base_filename
    title = f"KNN Classification Results Summary\n{base_filename} vs {ref_name}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    # Create table
    table = ax.table(cellText=df.values,
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color the row labels
    for i in range(1, len(df) + 1):
        table[(i, -1)].set_facecolor('#E8F5E8')
        table[(i, -1)].set_text_props(weight='bold')

    # Alternate row colors for better readability
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')

    # Add explanatory text
    explanation = ("Reference CV: Cross-validation accuracy on reference dataset\n"
                    "Query Transfer: Accuracy when transferring to query dataset\n"
                    "Values show mean±std for CV, single value for transfer")

    fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, 
                style='italic', wrap=True)

    # Save table as PDF
    table_output = f"knn_results_summary_final_{base_filename}.pdf"
    plt.savefig(table_output, dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    print(f"Saved results table as {table_output}")
    plt.close()

    # Also save as CSV for easy access
    csv_output = f"knn_results_summary_{base_filename}.csv"
    df.to_csv(csv_output)
    print(f"Saved results table as {csv_output}")



