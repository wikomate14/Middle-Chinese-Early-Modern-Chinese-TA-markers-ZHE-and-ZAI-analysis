# -*- coding: utf-8 -*-
"""Pre_Processing_splitting.ipynb


"""
# The following lines are only used on Google Colab and are commented out by default in this repository version.
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# !pip install pandas
import pandas as pd

import os

# !pip install openpyxl
import openpyxl

import csv

with open('/data/dataset/collection guideline - EMoC.txt', 'r', encoding='utf-8') as f:
    sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
    delimiter = dialect.delimiter
print("separator:", repr(delimiter))

from io import StringIO

def read_markdown_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove formatting rows (e.g. |----|----| )
    lines = [line for line in lines if not set(line.strip()) <= {'|', '-', ' '}]

    # Merge the remaining lines into a single string
    content = ''.join(lines)

    # Read the table into a DF using pandas
    df = pd.read_csv(StringIO(content), sep='|', engine='python')

    # Drop any completely empty columns
    df = df.dropna(axis=1, how='all')

    # Strip whitespace from all string cells
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df

# Read the two tables
df_mc = read_markdown_table('/data/dataset/collection guideline - MC.txt')
df_emoc = read_markdown_table('/data/dataset/collection guideline - EMoC.txt')

# Concatenate them into a single DataDrame
mapping = pd.concat([df_mc, df_emoc], ignore_index=True)

print(mapping)

# Extract the first number from the "Century" column
mapping['century'] = mapping['**世纪**'].str.extract(r'(\d+)', expand=False)

# Display the updated DataFrame
print(mapping[['**书名**', '**世纪**', 'century']])

file_path_probe = "/data/dataset/initial collection/UntaggedZHE_EMoC.xlsx"

content = pd.read_excel(file_path_probe, header=None, names=None)

print(content.head(10).to_string(index=False, header=False))

unique_names = content[1].unique()

print(f"Number of unique names in column 2: {len(unique_names)}")
print("Unique names:", unique_names)

# —— 1. Configuration: Each item contains source files and two output directories ——
configs = [
    {
        "src": "/data/dataset/initial collection/TaggedZAI_MC.xlsx",
        "transit_dir": "/data/dataset/split1/transit1",
        "output_dir": "/data/dataset/split1/zai_MC"
    },
    {
        "src": "/data/dataset/initial collection/TaggedZHE_MC.xlsx",
        "transit_dir": "/data/dataset/split1/transit2",
        "output_dir": "/data/dataset/split1/zhe_MC"
    },
    {
        "src": "/data/dataset/initial collection/TaggedZAI_EMoC.xlsx",
        "transit_dir": "/data/dataset/split1/transit3",
        "output_dir": "/data/dataset/split1/zai_EMoC"
    },
    {
        "src": "/data/dataset/initial collection/TaggedZHE_EMoC.xlsx",
        "transit_dir": "/data/dataset/split1/transit4",
        "output_dir": "/data/dataset/split1/zhe_EMoC"
    }
]

for cfg in configs:
    src_path = cfg["src"]
    transit_dir = cfg["transit_dir"]
    output_dir = cfg["output_dir"]

    # Ensure the directory exists
    os.makedirs(transit_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # —— 2. First split: Output to transit_dir by prefix ——
    df = pd.read_excel(src_path, header=None)
    prefixes = df[1].str.split("/", n=1).str[0].unique()

    for prefix in prefixes:
        sub_df = df[df[1].str.startswith(prefix)].reset_index(drop=True)
        out_path = os.path.join(transit_dir, f"{prefix}.xlsx")
        sub_df.to_excel(out_path, index=False, header=False)

    print(f"[finish splitting] {os.path.basename(src_path)} → {len(prefixes)} files")

    # —— 3. Secondary cleaning: Process each xlsx file under transit_dir again and output it to output_dir ——
    for fname in os.listdir(transit_dir):
        if not fname.lower().endswith(".xlsx"):
            continue
        in_path = os.path.join(transit_dir, fname)
        df2 = pd.read_excel(in_path, header=None)

        # Reset the index and insert a new column
        df2 = df2.reset_index(drop=True)
        df2.insert(0, 'New_Index', df2.index + 1)
        # Delete the original "Column 2" (index 1)
        df2 = df2.drop(columns=[1])

        out2_path = os.path.join(output_dir, fname)
        df2.to_excel(out2_path, index=False, header=False)

    print(f"[finishing clean] {os.path.basename(src_path)} → {len(os.listdir(output_dir))} files")

import shutil

def ordinal_suffix(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

source_groups = [
    {
        'src_dirs': [
            '/data/dataset/split1/zhe_MC',
            '/data/dataset/split1/zhe_EMoC'
        ],
        'dest_root': '/data/dataset/groups by century/Tagged_zhe'
    },
    {
        'src_dirs': [
            '/data/dataset/split1/zai_MC',
            '/data/dataset/split1/zai_EMoC'
        ],
        'dest_root': '/data/dataset/groups by century/Tagged_zai'
    }
]

# Traverse each source directory group
for group in source_groups:
    src_dirs = group['src_dirs']
    dest_root = group['dest_root']

    for src_dir in src_dirs:
        for file in os.listdir(src_dir):
            if not file.endswith('.xlsx'):
                continue

            base_name = os.path.splitext(file)[0]
            matched = mapping[mapping['**书名**'] == base_name]

            if not matched.empty:
                century_num = int(matched.iloc[0]['century'])
                century_str = f"{ordinal_suffix(century_num)} century"

                dest_dir = os.path.join(dest_root, century_str)
                os.makedirs(dest_dir, exist_ok=True)

                src_path = os.path.join(src_dir, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)
                print(f"✅ copied: {file} → {dest_dir}")
            else:
                print(f"⚠️ not found: {file} ({src_dir})")

print("\nThe operation is completed and the original file remains undeleted")

import os, re, unicodedata
import pandas as pd

# —— Tools ——
def _normalize_filecell(x):
    if pd.isna(x):
        return x
    s = str(x)
    s = unicodedata.normalize("NFKC", s)  # Full-width character → Half-width character
    s = s.strip().strip("'\"")            # Remove quotation marks/whitespace
    s = s.replace("／", "/")              # Full-width forward slash → Half-width character
    return s

def _safe_name(s):
    s = str(s) if s is not None else "EMPTY"
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "EMPTY"

def _pick_file_col_by_const_prefix(df_noidx):
    """
    In the sub-table after removing New_Index, automatically locate the "file" column:

    1) Normalize each column -> Calculate the percentage of rows containing '/' in that column (cover);

    2) Take the prefix token before '/'. If the token's uniqueness is == 1 (almost constant), the score is cover;

    3) Select the column with the highest score; if there is no column satisfying uniq=1, as a second-best option: select the column with the highest percentage of rows containing '/'.
    """
    Returns: column index or None
    best_col, best_cover = None, -1.0
    fallback_col, fallback_cover = None, -1.0

    for c in df_noidx.columns:
        s = df_noidx[c].map(_normalize_filecell).astype("string")
        has_slash = s.str.contains("/", na=False)
        cover = has_slash.mean()  # What percentage of rows resemble "paths"?
        if cover > fallback_cover:
            fallback_cover, fallback_col = cover, c

        if cover == 0:
            continue
        prefix = s.where(has_slash, None).dropna().str.split("/", n=1).str[0]
        nunq = prefix.nunique(dropna=True)

        # “The title prefix is ​​almost always constant” is a strong signal: unique==1
        if nunq == 1 and cover > best_cover:
            best_cover, best_col = cover, c

    # Define thresholds: Prioritize returning columns with "constant prefixes" (even if the cover is not 100%).
    if best_col is not None and best_cover > 0.1:  # A coverage of 10% or more can be considered a file column (sub-tables are usually much larger than this value).
        return best_col

    # Backtrack: Return to the column with the highest percentage of characters containing '/' (ZHE rarely uses this step)
    return fallback_col if (fallback_col is not None and fallback_cover > 0) else None


# —— 1. Configuration: ZHE (EMoC) ——
configs = [
    {
        "src": "/data/dataset/initial collection/UntaggedZHE_EMoC.xlsx",
        "transit_dir": "/data/dataset/split2/transit4",
        "output_dir": "/data/dataset/split2/zhe_EMoC"
    }
]

for cfg in configs:
    src_path = cfg["src"]
    transit_dir = cfg["transit_dir"]
    output_dir = cfg["output_dir"]

    # Optional: Clear the directory to avoid interference from old files.
    os.makedirs(transit_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    for _dir in (transit_dir, output_dir):
        for name in os.listdir(_dir):
            if name.lower().endswith(".xlsx") or name.startswith("~$"):
                try:
                    os.remove(os.path.join(_dir, name))
                except Exception:
                    pass

    # —— 2) First split: Write transit_dir with "/" prefix ——
    df = pd.read_excel(src_path, header=None)

    # Normalize only "column 2 (index 1)"; take the prefix list.
    s = df.iloc[:, 1].astype("string").map(_normalize_filecell)
    prefixes = s.dropna().str.split("/", n=1).str[0].unique().tolist()

    for prefix in prefixes:
        mask = s.str.startswith(prefix, na=False)
        sub_df = df[mask].reset_index(drop=True)

        out_path = os.path.join(transit_dir, f"{_safe_name(prefix)}.xlsx")
        sub_df.to_excel(out_path, index=False, header=False)

    print(f"[finish splitting] {os.path.basename(src_path)} → {len(prefixes)} files")

    # —— 3) Secondary cleaning: Insert New_Index + automatically locate and delete the "file" column ——
    total_in, total_out = 0, 0
    for fname in os.listdir(transit_dir):
        if not fname.lower().endswith(".xlsx"):
            continue
        if fname.startswith("~$"):  # Skip Excel temporary lock file
            continue

        in_path = os.path.join(transit_dir, fname)
        df2 = pd.read_excel(in_path, header=None)

        # Count the number of input lines
        total_in += len(df2)

        # Reset the index and insert a new column
        df2 = df2.reset_index(drop=True)
        df2.insert(0, 'New_Index', df2.index + 1)

        # Automatically position the file column in a view excluding New_Index
        candidate_view = df2.drop(columns=['New_Index'])
        file_col = _pick_file_col_by_const_prefix(candidate_view)

        if file_col is not None:
            df2 = df2.drop(columns=[file_col])

        out2_path = os.path.join(output_dir, fname)
        df2.to_excel(out2_path, index=False, header=False)

        total_out += len(df2)

    print(f"[finishing clean] {os.path.basename(src_path)} → files: {sum(1 for f in os.listdir(output_dir) if f.lower().endswith('.xlsx'))}, rows(in/out)={total_in}/{total_out}")

# —— 1. Configuration: Each item contains source files and two output directories ——
configs = [
    {
        "src": "/data/dataset/initial collection/UntaggedZAI_MC.xlsx",
        "transit_dir": "/data/dataset/split2/transit1",
        "output_dir": "/data/dataset/split2/zai_MC"
    },
    {
        "src": "/data/dataset/initial collection/UntaggedZHE_MC.xlsx",
        "transit_dir": "/data/dataset/split2/transit2",
        "output_dir": "/data/dataset/split2/zhe_MC"
    },
    {
        "src": "/data/dataset/initial collection/UntaggedZAI_EMoC.xlsx",
        "transit_dir": "/data/dataset/split2/transit3",
        "output_dir": "/data/dataset/split2/zai_EMoC"
    }
]

for cfg in configs:
    src_path = cfg["src"]
    transit_dir = cfg["transit_dir"]
    output_dir = cfg["output_dir"]

    # Ensure the directory exists
    os.makedirs(transit_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # —— 2. First split: Output to transit_dir by prefix ——
    df = pd.read_excel(src_path, header=None)
    prefixes = df[1].str.split("/", n=1).str[0].unique()

    for prefix in prefixes:
        sub_df = df[df[1].str.startswith(prefix)].reset_index(drop=True)
        out_path = os.path.join(transit_dir, f"{prefix}.xlsx")
        sub_df.to_excel(out_path, index=False, header=False)

    print(f"[finish splitting] {os.path.basename(src_path)} → {len(prefixes)} files")

    # —— 3. Secondary cleaning: Process each xlsx file under transit_dir again and output it to output_dir ——
    for fname in os.listdir(transit_dir):
        if not fname.lower().endswith(".xlsx"):
            continue
        in_path = os.path.join(transit_dir, fname)
        df2 = pd.read_excel(in_path, header=None)

        # Reset the index and insert a new column
        df2 = df2.reset_index(drop=True)
        df2.insert(0, 'New_Index', df2.index + 1)
        # Delete the original "Column 2" (index 1)
        df2 = df2.drop(columns=[1])

        out2_path = os.path.join(output_dir, fname)
        df2.to_excel(out2_path, index=False, header=False)

    print(f"[finishing clean] {os.path.basename(src_path)} → {len(os.listdir(output_dir))} files")

import shutil

def ordinal_suffix(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

source_groups = [
    {
        'src_dirs': [
            '/data/dataset/split2/zhe_MC',
            '/data/dataset/split2/zhe_EMoC'
        ],

        'dest_root': '/data/dataset/groups by century/Untagged_zhe'
    },
    {
        'src_dirs': [
            '/data/dataset/split2/zai_MC',
            '/data/dataset/split2/zai_EMoC'
        ],
        'dest_root': '/data/dataset/groups by century/Untagged_zai'
    }
]

# Traverse each source directory group
for group in source_groups:
    src_dirs = group['src_dirs']
    dest_root = group['dest_root']

    for src_dir in src_dirs:
        for file in os.listdir(src_dir):
            if not file.endswith('.xlsx'):
                continue

            base_name = os.path.splitext(file)[0]
            matched = mapping[mapping['**书名**'] == base_name]

            if not matched.empty:
                century_num = int(matched.iloc[0]['century'])
                century_str = f"{ordinal_suffix(century_num)} century"

                dest_dir = os.path.join(dest_root, century_str)
                os.makedirs(dest_dir, exist_ok=True)

                src_path = os.path.join(src_dir, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)
                print(f"✅ copied: {file} → {dest_dir}")
            else:
                print(f"⚠️ not found: {file} ({src_dir})")

print("\nThe operation is completed and the original file remains undeleted")