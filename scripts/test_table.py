from data_utils.create_table import initialize_bioinf_table,update_bioinf_table,save_table_as_csv,save_table_as_pdf,load_table_from_csv, display_table
# Step 1: Load existing or initialize new table
try:
    my_table = load_table_from_csv()
except FileNotFoundError:
    my_table = initialize_bioinf_table()

# Step 2: Add/update data
my_table = update_bioinf_table(my_table, 'kidney', 'Pavlin', 0.88)
my_table = update_bioinf_table(my_table, 'lung', 'source 1', 'Tabula Muris')
my_table = update_bioinf_table(my_table, 'kidney', 'Pavlin_tsne', 0.95)
my_table = update_bioinf_table(my_table, 'pancreas', 'baron_', 0.95)

# Step 3: Save updated data
save_table_as_csv(my_table)
save_table_as_pdf(my_table)

display_table(my_table)
