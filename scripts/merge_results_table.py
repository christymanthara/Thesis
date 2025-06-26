def merge_results_tables(table1, table2, source_key=None):
    """
    Merge two results tables that share the same source/dataset.
    
    Parameters:
    -----------
    table1 : dict
        Results table from first pipeline
    table2 : dict  
        Results table from second pipeline
    source_key : str, optional
        Common identifier for the datasets (e.g., "baron_xin")
        
    Returns:
    --------
    dict: Merged results table
    """
    merged_table = {}
    
    # Copy all entries from first table
    for method_name, metrics in table1.items():
        merged_table[method_name] = metrics.copy()
    
    # Add entries from second table
    for method_name, metrics in table2.items():
        if method_name in merged_table:
            # If method exists in both, merge the metrics
            merged_table[method_name].update(metrics)
        else:
            # If method is new, add it
            merged_table[method_name] = metrics.copy()
    
    return merged_table

def create_unified_results_table(merged_table, source1, source2, filename_base, metadata=None):
    """
    Create a unified results table that includes results from both pipelines.
    This is an enhanced version of your existing create_results_table function.
    """
    import pandas as pd
    from datetime import datetime
    
    # Convert results to DataFrame
    df = pd.DataFrame.from_dict(merged_table, orient='index')
    
    # Add metadata information
    if metadata:
        print(f"\nDataset Information:")
        print(f"Reference: {metadata.get('ref_source', 'unknown')} ({metadata.get('ref_cell_count', 'unknown')} cells)")
        print(f"Query: {metadata.get('query_source', 'unknown')} ({metadata.get('query_cell_count', 'unknown')} cells)")
        print(f"Reference Tissue: {metadata.get('ref_tissue', 'unknown')}")
        print(f"Query Tissue: {metadata.get('query_tissue', 'unknown')}")
        print(f"Reference Organism: {metadata.get('ref_organism', 'unknown')}")
        print(f"Query Organism: {metadata.get('query_organism', 'unknown')}")
    
    # Print the unified table
    print(f"\n=== UNIFIED RESULTS TABLE ===")
    print(df.to_string())
    
    # Save to CSV
    csv_filename = f"{filename_base}_unified.csv"
    df.to_csv(csv_filename)
    print(f"\nUnified results saved to: {csv_filename}")
    
    # Save to Excel with metadata
    excel_filename = f"{filename_base}_unified.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Write results table
        df.to_excel(writer, sheet_name='Results')
        
        # Write metadata if available
        if metadata:
            metadata_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"Unified results with metadata saved to: {excel_filename}")
    
    return df