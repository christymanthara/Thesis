import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import json

def create_results_table(results_table, main_source: str, ref_source: str, base_filename, reference_file=None, metadata=None):
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
    metadata : dict, optional
        Dictionary containing tissue and organism information
        Expected keys: main_tissue, ref_tissue, main_organism, ref_organism
    """
    
    # Define persistent storage files
    results_pickle_file = "knn_results_accumulated.pkl"
    results_json_file = "knn_results_accumulated.json"
    
    # Load existing accumulated results if they exist
    accumulated_results = load_accumulated_results(results_pickle_file)
    
    # Create source comparison key with metadata if available
    if metadata:
        # Extract metadata information
        main_tissue = metadata.get('main_tissue', 'unknown')
        ref_tissue = metadata.get('ref_tissue', 'unknown')
        main_organism = metadata.get('main_organism', 'unknown')
        ref_organism = metadata.get('ref_organism', 'unknown')
        
        # Create enhanced comparison key
        source_comparison_key = f"{main_source} vs {ref_source}"
        
        # Create metadata string for display
        metadata_str = f"Main: {main_organism}_{main_tissue} | Ref: {ref_organism}_{ref_tissue}"
        
        # Store enhanced metadata
        comparison_metadata = {
            'main_source': main_source,
            'ref_source': ref_source,
            'main_tissue': main_tissue,
            'ref_tissue': ref_tissue,
            'main_organism': main_organism,
            'ref_organism': ref_organism,
            'metadata_display': metadata_str
        }
    else:
        source_comparison_key = f"{main_source} vs {ref_source}"
        metadata_str = "No metadata available"
        comparison_metadata = {
            'main_source': main_source,
            'ref_source': ref_source,
            'metadata_display': metadata_str
        }
    
    # Extract query transfer accuracies from current results_table
    query_accuracies = {}
    reference_cv_scores = {}
    
    for embedding, results in results_table.items():
        # Extract the query transfer accuracy and reference CV
        query_transfer = results['Query Transfer']
        reference_cv = results['Reference CV']
        
        # Remove any extra formatting if present
        if isinstance(query_transfer, str):
            query_accuracies[embedding] = query_transfer
        else:
            query_accuracies[embedding] = f"{query_transfer:.3f}"
            
        if isinstance(reference_cv, str):
            reference_cv_scores[embedding] = reference_cv
        else:
            reference_cv_scores[embedding] = f"{reference_cv:.3f}"
    
    # Update accumulated results with enhanced structure
    print(f"Processing source comparison: {source_comparison_key}")
    if source_comparison_key in accumulated_results:
        print(f"  - Overwriting existing results for {source_comparison_key}")
    else:
        print(f"  - Adding new results for {source_comparison_key}")
    
    # Store both accuracies and metadata
    accumulated_results[source_comparison_key] = {
        'query_accuracies': query_accuracies,
        'reference_cv_scores': reference_cv_scores,
        'metadata': comparison_metadata
    }
    
    # Save updated accumulated results
    save_accumulated_results(accumulated_results, results_pickle_file, results_json_file)
    
    # Create comprehensive DataFrame from all accumulated results
    df_query, df_reference = create_comprehensive_dataframes(accumulated_results)
    
    # Create enhanced visualization with metadata
    create_enhanced_table_visualization(df_query, df_reference, accumulated_results, 
                                      source_comparison_key, base_filename, reference_file)
    
    # Save comprehensive CSV files
    save_comprehensive_results(df_query, df_reference, accumulated_results, base_filename)
    
    return df_query, df_reference


def create_enhanced_table_visualization(df_query, df_reference, accumulated_results, 
                                      current_comparison, base_filename, reference_file):
    """Create enhanced visualization with metadata information."""
    
    # Create figure with subplots for both query and reference results
    fig = plt.figure(figsize=(16, max(12, len(df_query) * 0.8 + 4)))
    
    # Create title with metadata
    current_metadata = accumulated_results[current_comparison]['metadata']
    title = (f"KNN Classification Results - Accumulated Results Across All Runs\n"
            f"Current Run: {current_comparison}\n"
            f"Metadata: {current_metadata['metadata_display']}")
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    
    # Create two subplots: one for query transfer, one for reference CV
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Query Transfer Results Table
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('tight')
    ax1.axis('off')
    ax1.set_title("Query Transfer Accuracy", fontweight='bold', pad=20)
    
    create_styled_table(ax1, df_query, current_comparison, accumulated_results, 'query')
    
    # Reference CV Results Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('tight')
    ax2.axis('off')
    ax2.set_title("Reference Cross-Validation Accuracy", fontweight='bold', pad=20)
    
    create_styled_table(ax2, df_reference, current_comparison, accumulated_results, 'reference')
    
    # Add metadata summary at the bottom
    metadata_text = create_metadata_summary(accumulated_results)
    fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=9,
             style='italic', wrap=True, bbox=dict(boxstyle="round,pad=0.3", 
                                                facecolor='lightblue', alpha=0.3))
    
    # Save enhanced table as PDF
    table_output = f"knn_results_comprehensive_with_metadata.pdf"
    plt.savefig(table_output, dpi=300, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    print(f"Saved comprehensive results table with metadata as {table_output}")
    plt.close()


def create_styled_table(ax, df, current_comparison, accumulated_results, table_type):
    """Create a styled table for either query or reference results."""
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color the header
    for i in range(len(df.columns)):
        color = '#4CAF50' if table_type == 'query' else '#2196F3'
        table[(0, i)].set_facecolor(color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the first column (Source Comparison)
    for row in range(1, len(df) + 1):
        bg_color = '#E8F5E8' if table_type == 'query' else '#E3F2FD'
        table[(row, 0)].set_facecolor(bg_color)
        table[(row, 0)].set_text_props(weight='bold')
    
    # Color the data cells with alternating colors
    for row in range(1, len(df) + 1):
        color = '#F5F5F5' if row % 2 == 1 else '#FFFFFF'
        for col in range(1, len(df.columns)):
            table[(row, col)].set_facecolor(color)
    
    # Highlight the current run's row
    current_row = None
    for i, comparison in enumerate(df['Source Comparison']):
        if comparison == current_comparison:
            current_row = i + 1
            break
    
    if current_row:
        highlight_color = '#FFE6CC' if table_type == 'query' else '#FFF3E0'
        for col in range(len(df.columns)):
            table[(current_row, col)].set_facecolor(highlight_color)
            if col == 0:
                table[(current_row, col)].set_text_props(weight='bold', color='#D2691E')


def create_metadata_summary(accumulated_results):
    """Create a summary of metadata across all runs."""
    metadata_summary = "Metadata Summary:\n"
    
    for comparison, data in accumulated_results.items():
        metadata = data.get('metadata', {})
        metadata_summary += f"• {comparison}: {metadata.get('metadata_display', 'No metadata')}\n"
    
    return metadata_summary


def create_comprehensive_dataframes(accumulated_results):
    """Create comprehensive DataFrames for both query and reference results."""
    if not accumulated_results:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all unique embedding methods across all runs
    all_embeddings = set()
    for comparison_data in accumulated_results.values():
        if 'query_accuracies' in comparison_data:
            all_embeddings.update(comparison_data['query_accuracies'].keys())
    
    all_embeddings = sorted(list(all_embeddings))
    
    # Create DataFrames for both query and reference results
    query_data = {'Source Comparison': []}
    reference_data = {'Source Comparison': []}
    
    for embedding in all_embeddings:
        query_data[embedding] = []
        reference_data[embedding] = []
    
    # Fill data
    for source_comparison, comparison_data in accumulated_results.items():
        query_results = comparison_data.get('query_accuracies', {})
        reference_results = comparison_data.get('reference_cv_scores', {})
        
        query_data['Source Comparison'].append(source_comparison)
        reference_data['Source Comparison'].append(source_comparison)
        
        for embedding in all_embeddings:
            query_data[embedding].append(query_results.get(embedding, 'N/A'))
            reference_data[embedding].append(reference_results.get(embedding, 'N/A'))
    
    return pd.DataFrame(query_data), pd.DataFrame(reference_data)


def save_comprehensive_results(df_query, df_reference, accumulated_results, base_filename):
    """Save comprehensive results to CSV files."""
    
    # Save query transfer results
    query_csv = f"knn_query_transfer_results_comprehensive.csv"
    df_query.to_csv(query_csv, index=False)
    print(f"Saved comprehensive query transfer results as {query_csv}")
    
    # Save reference CV results
    reference_csv = f"knn_reference_cv_results_comprehensive.csv"
    df_reference.to_csv(reference_csv, index=False)
    print(f"Saved comprehensive reference CV results as {reference_csv}")
    
    # Save metadata summary
    metadata_df = create_metadata_dataframe(accumulated_results)
    metadata_csv = f"knn_metadata_summary.csv"
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"Saved metadata summary as {metadata_csv}")
    
    # Save current run's results separately
    current_comparison = list(accumulated_results.keys())[-1]  # Get the last added comparison
    current_data = accumulated_results[current_comparison]
    
    current_df = pd.DataFrame({
        'Source Comparison': [current_comparison],
        'Metadata': [current_data['metadata']['metadata_display']],
        **current_data['query_accuracies']
    })
    current_csv = f"knn_results_current_run_{base_filename}.csv"
    current_df.to_csv(current_csv, index=False)
    print(f"Saved current run results as {current_csv}")


def create_metadata_dataframe(accumulated_results):
    """Create a DataFrame with metadata information."""
    metadata_data = {
        'Source Comparison': [],
        'Main Source': [],
        'Reference Source': [],
        'Main Tissue': [],
        'Reference Tissue': [],
        'Main Organism': [],
        'Reference Organism': [],
        'Metadata Display': []
    }
    
    for comparison, data in accumulated_results.items():
        metadata = data.get('metadata', {})
        metadata_data['Source Comparison'].append(comparison)
        metadata_data['Main Source'].append(metadata.get('main_source', 'unknown'))
        metadata_data['Reference Source'].append(metadata.get('ref_source', 'unknown'))
        metadata_data['Main Tissue'].append(metadata.get('main_tissue', 'unknown'))
        metadata_data['Reference Tissue'].append(metadata.get('ref_tissue', 'unknown'))
        metadata_data['Main Organism'].append(metadata.get('main_organism', 'unknown'))
        metadata_data['Reference Organism'].append(metadata.get('ref_organism', 'unknown'))
        metadata_data['Metadata Display'].append(metadata.get('metadata_display', 'No metadata'))
    
    return pd.DataFrame(metadata_data)


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


def reset_accumulated_results():
    """Utility function to reset/clear accumulated results."""
    files_to_remove = [
        "knn_results_accumulated.pkl",
        "knn_results_accumulated.json",
        "knn_results_comprehensive_with_metadata.pdf",
        "knn_query_transfer_results_comprehensive.csv",
        "knn_reference_cv_results_comprehensive.csv",
        "knn_metadata_summary.csv"
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
    for source_comparison, data in accumulated_results.items():
        print(f"\n{source_comparison}:")
        metadata = data.get('metadata', {})
        print(f"  Metadata: {metadata.get('metadata_display', 'No metadata')}")
        
        query_results = data.get('query_accuracies', {})
        reference_results = data.get('reference_cv_scores', {})
        
        print("  Query Transfer Accuracies:")
        for embedding, accuracy in query_results.items():
            print(f"    {embedding}: {accuracy}")
        
        print("  Reference CV Scores:")
        for embedding, score in reference_results.items():
            print(f"    {embedding}: {score}")
    
    print(f"\nTotal source comparisons: {len(accumulated_results)}")


# Example usage functions for testing
if __name__ == "__main__":
    # Test the enhanced functionality with metadata
    
    # Example metadata
    metadata_1 = {
        'main_tissue': 'pancreas',
        'ref_tissue': 'pancreas',
        'main_organism': 'human',
        'ref_organism': 'human'
    }
    
    metadata_2 = {
        'main_tissue': 'pancreas',
        'ref_tissue': 'pancreas',
        'main_organism': 'mouse',
        'ref_organism': 'human'
    }
    
    # Example 1: First run with metadata
    print("=== Example 1: First run with metadata ===")
    results_table_1 = {
        'scVI': {'Reference CV': '0.850±0.023', 'Query Transfer': '0.834'},
        'scANVI': {'Reference CV': '0.901±0.015', 'Query Transfer': '0.867'},
        'RAW data': {'Reference CV': '0.612±0.045', 'Query Transfer': '0.598'}
    }
    create_results_table(results_table_1, "xin_2016", "baron_2016h", "test_file_1", metadata=metadata_1)
    
    # Example 2: Second run with different metadata
    print("\n=== Example 2: Run with different metadata ===")
    results_table_2 = {
        'scVI': {'Reference CV': '0.782±0.028', 'Query Transfer': '0.756'},
        'scANVI': {'Reference CV': '0.823±0.022', 'Query Transfer': '0.801'},
        'UMAP': {'Reference CV': '0.689±0.038', 'Query Transfer': '0.654'}
    }
    create_results_table(results_table_2, "muraro_2016", "baron_2016h", "test_file_2", metadata=metadata_2)
    
    # Show all accumulated results
    print("\n=== Final accumulated results with metadata ===")
    show_accumulated_results()