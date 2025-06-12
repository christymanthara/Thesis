import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import json

def create_results_table(results_table, main_source: str, ref_source: str, base_filename, reference_file=None):
    """
    Create a summary table of all KNN results and save as PDF.
    Also manages persistent storage of results across multiple runs.
    
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
    
    # Define persistent storage files
    results_pickle_file = "knn_results_accumulated.pkl"
    results_json_file = "knn_results_accumulated.json"
    
    # Load existing accumulated results if they exist
    accumulated_results = load_accumulated_results(results_pickle_file)
    
    # Create source comparison key
    source_comparison_key = f"{main_source} vs {ref_source}"
    
    # Extract query transfer accuracies from current results_table
    query_accuracies = {}
    for embedding, results in results_table.items():
        # Extract the query transfer accuracy (remove any formatting)
        query_transfer = results['Query Transfer']
        # Remove any extra formatting if present
        if isinstance(query_transfer, str):
            query_accuracies[embedding] = query_transfer
        else:
            query_accuracies[embedding] = f"{query_transfer:.3f}"
    
    # Update accumulated results
    print(f"Processing source comparison: {source_comparison_key}")
    if source_comparison_key in accumulated_results:
        print(f"  - Overwriting existing results for {source_comparison_key}")
    else:
        print(f"  - Adding new results for {source_comparison_key}")
    
    accumulated_results[source_comparison_key] = query_accuracies
    
    # Save updated accumulated results
    save_accumulated_results(accumulated_results, results_pickle_file, results_json_file)
    
    # Create comprehensive DataFrame from all accumulated results
    df = create_comprehensive_dataframe(accumulated_results)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 2), max(6, len(df) + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create title
    ref_name = os.path.splitext(os.path.basename(reference_file))[0] if reference_file else base_filename
    title = f"KNN Classification Results - Query Transfer Accuracy\nAccumulated Results Across All Runs"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0.1, 1, 0.8])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Color the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the first column (Source Comparison)
    for row in range(1, len(df) + 1):
        table[(row, 0)].set_facecolor('#E8F5E8')
        table[(row, 0)].set_text_props(weight='bold')
    
    # Color the data cells with alternating colors
    for row in range(1, len(df) + 1):
        color = '#F5F5F5' if row % 2 == 1 else '#FFFFFF'
        for col in range(1, len(df.columns)):
            table[(row, col)].set_facecolor(color)
    
    # Highlight the current run's row
    current_row = None
    for i, comparison in enumerate(df['Source Comparison']):
        if comparison == source_comparison_key:
            current_row = i + 1
            break
    
    if current_row:
        for col in range(len(df.columns)):
            table[(current_row, col)].set_facecolor('#FFE6CC')  # Light orange highlight
            if col == 0:
                table[(current_row, col)].set_text_props(weight='bold', color='#D2691E')
    
    # Add explanatory text
    explanation = (f"Table shows query transfer accuracy for each embedding method across all runs\n"
                  f"Current run: {source_comparison_key} (highlighted in orange)\n"
                  f"Total comparisons: {len(df)}")
    
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=10,
             style='italic', wrap=True)
    
    # Save comprehensive table as PDF
    table_output = f"knn_results_comprehensive_accumulated.pdf"
    plt.savefig(table_output, dpi=300, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    print(f"Saved comprehensive results table as {table_output}")
    plt.close()
    
    # Also save comprehensive CSV
    csv_output = f"knn_results_comprehensive_accumulated.csv"
    df.to_csv(csv_output, index=False)
    print(f"Saved comprehensive results as {csv_output}")
    
    # Save current run's results separately too
    current_df = pd.DataFrame({
        'Source Comparison': [source_comparison_key],
        **query_accuracies
    })
    current_csv = f"knn_results_current_run_{base_filename}.csv"
    current_df.to_csv(current_csv, index=False)
    print(f"Saved current run results as {current_csv}")
    
    return df


def load_accumulated_results(pickle_file):
    """Load accumulated results from pickle file, return empty dict if file doesn't exist."""
    if os.path.exists(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                accumulated_results = pickle.load(f)
            print(f"Loaded existing results from {pickle_file}")
            print(f"  - Found {len(accumulated_results)} existing source comparisons")
            return accumulated_results
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            print("Starting with empty results dictionary")
            return {}
    else:
        print(f"No existing results file found. Starting fresh.")
        return {}


def save_accumulated_results(accumulated_results, pickle_file, json_file):
    """Save accumulated results to both pickle and JSON formats."""
    try:
        # Save as pickle (preserves exact data types)
        with open(pickle_file, 'wb') as f:
            pickle.dump(accumulated_results, f)
        print(f"Saved accumulated results to {pickle_file}")
        
        # Save as JSON (human-readable)
        with open(json_file, 'w') as f:
            json.dump(accumulated_results, f, indent=2)
        print(f"Saved accumulated results to {json_file}")
        
    except Exception as e:
        print(f"Error saving accumulated results: {e}")


def create_comprehensive_dataframe(accumulated_results):
    """Create a comprehensive DataFrame from all accumulated results."""
    if not accumulated_results:
        return pd.DataFrame()
    
    # Get all unique embedding methods across all runs
    all_embeddings = set()
    for source_results in accumulated_results.values():
        all_embeddings.update(source_results.keys())
    
    all_embeddings = sorted(list(all_embeddings))
    
    # Create DataFrame
    data = {'Source Comparison': []}
    for embedding in all_embeddings:
        data[embedding] = []
    
    # Fill data
    for source_comparison, results in accumulated_results.items():
        data['Source Comparison'].append(source_comparison)
        for embedding in all_embeddings:
            data[embedding].append(results.get(embedding, 'N/A'))
    
    return pd.DataFrame(data)


def reset_accumulated_results():
    """Utility function to reset/clear accumulated results."""
    files_to_remove = [
        "knn_results_accumulated.pkl",
        "knn_results_accumulated.json",
        "knn_results_comprehensive_accumulated.pdf",
        "knn_results_comprehensive_accumulated.csv"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")
    
    print("Accumulated results have been reset.")


def show_accumulated_results():
    """Utility function to display current accumulated results."""
    pickle_file = "knn_results_accumulated.pkl"
    accumulated_results = load_accumulated_results(pickle_file)
    
    if not accumulated_results:
        print("No accumulated results found.")
        return
    
    print("\n=== Current Accumulated Results ===")
    for source_comparison, results in accumulated_results.items():
        print(f"\n{source_comparison}:")
        for embedding, accuracy in results.items():
            print(f"  {embedding}: {accuracy}")
    print(f"\nTotal source comparisons: {len(accumulated_results)}")


# Example usage functions for testing
if __name__ == "__main__":
    # Test the functionality
    
    # Example 1: First run
    print("=== Example 1: First run ===")
    results_table_1 = {
        'scVI': {'Reference CV': '0.850±0.023', 'Query Transfer': '0.834'},
        'scANVI': {'Reference CV': '0.901±0.015', 'Query Transfer': '0.867'},
        'RAW data': {'Reference CV': '0.612±0.045', 'Query Transfer': '0.598'}
    }
    create_results_table(results_table_1, "xin_2016", "baron_2016h", "test_file_1")
    
    # Example 2: Second run with same comparison (should overwrite)
    print("\n=== Example 2: Overwrite existing comparison ===")
    results_table_2 = {
        'scVI': {'Reference CV': '0.855±0.020', 'Query Transfer': '0.840'},
        'scANVI': {'Reference CV': '0.905±0.018', 'Query Transfer': '0.872'},
        'RAW data': {'Reference CV': '0.615±0.042', 'Query Transfer': '0.602'},
        'PCA': {'Reference CV': '0.723±0.035', 'Query Transfer': '0.698'}
    }
    create_results_table(results_table_2, "xin_2016", "baron_2016h", "test_file_1_updated")
    
    # Example 3: Third run with different comparison (should add new)
    print("\n=== Example 3: New comparison ===")
    results_table_3 = {
        'scVI': {'Reference CV': '0.782±0.028', 'Query Transfer': '0.756'},
        'scANVI': {'Reference CV': '0.823±0.022', 'Query Transfer': '0.801'},
        'UMAP': {'Reference CV': '0.689±0.038', 'Query Transfer': '0.654'}
    }
    create_results_table(results_table_3, "muraro_2016", "baron_2016h", "test_file_2")
    
    # Show all accumulated results
    print("\n=== Final accumulated results ===")
    show_accumulated_results()