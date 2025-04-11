import pandas as pd
from tabulate import tabulate

CSV_PATH = "bioinf_table.csv"
README_PATH = "README.md"
REPO_URL = "https://github.com/christymanthara/Thesis/blob/main"  # or adjust if not main branch
VISUAL_PATH = "visuals"  # folder where files are stored

def format_links(df):
    for col in ["Pavlin_tsne", "Pavlin_umap"]:
        def make_link(val):
            if isinstance(val, str) and val.strip():
                return f"[{val}](./visuals/{val})"
            return ""
        df[col] = df[col].apply(make_link)
    return df

def update_readme_table():
    df = pd.read_csv(CSV_PATH)
    df = format_links(df)

    table_md = tabulate(df, headers="keys", tablefmt="github", showindex=False)

    with open(README_PATH, "r") as f:
        readme = f.read()

    start = readme.find("<!-- TABLE_START -->")
    end = readme.find("<!-- TABLE_END -->")

    if start == -1 or end == -1:
        raise ValueError("README must contain <!-- TABLE_START --> and <!-- TABLE_END --> markers")

    new_readme = (
        readme[:start+len("<!-- TABLE_START -->")] + "\n\n" +
        table_md + "\n\n" +
        readme[end:]
    )

    with open(README_PATH, "w") as f:
        f.write(new_readme)

if __name__ == "__main__":
    update_readme_table()
