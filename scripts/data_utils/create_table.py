import pandas as pd

# Initialize the table with columns
def initialize_bioinf_table():
    columns = ['tissue', 'source 1', 'source 2', 'Pavlin', 'Pavlin_tsne', 'Pavlin_umap', 'fastscbatch']
    return pd.DataFrame(columns=columns)

# Update the table
def update_bioinf_table(df, tissue, column_name, value):
    # Check if tissue already exists
    if tissue in df['tissue'].values:
        # Update the existing row
        df.loc[df['tissue'] == tissue, column_name] = value
    else:
        # Create a new row with NaN for missing values
        new_row = {col: None for col in df.columns}
        new_row['tissue'] = tissue
        new_row[column_name] = value
        df = df.append(new_row, ignore_index=True)
    return df
