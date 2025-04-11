import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Initialize the table with specific columns
def initialize_bioinf_table():
    columns = ['tissue', 'source 1', 'source 2', 'Pavlin', 'Pavlin_tsne', 'Pavlin_umap', 'fastscbatch']
    return pd.DataFrame(columns=columns)

# Add or update a specific cell in the table
def update_bioinf_table(df, tissue, column_name, value):
    if tissue in df['tissue'].values:
        df.loc[df['tissue'] == tissue, column_name] = value
    else:
        new_row = {col: None for col in df.columns}
        new_row['tissue'] = tissue
        new_row[column_name] = value
        df = df.append(new_row, ignore_index=True)
    return df

# Save the DataFrame as CSV
def save_table_as_csv(df, filename="bioinf_table.csv"):
    df.to_csv(filename, index=False)

# Load the DataFrame from CSV
def load_table_from_csv(filename="bioinf_table.csv"):
    return pd.read_csv(filename)

# Save the table as PDF
def save_table_as_pdf(df, filename="bioinf_table.pdf"):
    fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.5 + 2))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
