import pandas as pd
from tabulate import tabulate

def csv_to_markdown_table(csv_file):
    df = pd.read_csv(csv_file)
    return tabulate(df, headers='keys', tablefmt='github', showindex=False)

def update_readme(csv_file, readme_file='README.md'):
    table_md = csv_to_markdown_table(csv_file)
    
    with open(readme_file, 'r') as file:
        content = file.read()

    start_marker = "<!-- TABLE_START -->"
    end_marker = "<!-- TABLE_END -->"
    
    new_content = content.split(start_marker)[0] + start_marker + "\n\n" + table_md + "\n\n" + end_marker + content.split(end_marker)[-1]

    with open(readme_file, 'w') as file:
        file.write(new_content)

update_readme("bioinf_table.csv")
