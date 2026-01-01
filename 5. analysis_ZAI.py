# -*- coding: utf-8 -*-
"""Analysis (2).ipynb

"""
# The following lines are only used on Google Colab and are commented out by default in this repository version.
#from google.colab import drive
#drive.mount('/content/drive')

#!pip -q install adjustText

#!pip install prince

#!pip install -U plotly kaleido

from pathlib import Path

import pandas as pd, numpy as np, re

def use_cjk_font():
    import matplotlib, matplotlib.pyplot as plt
    from matplotlib import font_manager as fm, rcParams
    import glob, os

    # 1) Find any Noto CJK (or Source Han Sans) font in the system
    candidates = []
    candidates += glob.glob("/usr/share/fonts/**/NotoSansCJK*.*", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/NotoSerifCJK*.*", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/Noto*Sans*CJK*.*", recursive=True)
    # can also use own TTF/OTF path:

    # candidates += ["/content/drive/MyDrive/fonts/SimHei.ttf"]

    if not candidates:
        # If not, install Colab/Ubuntu.
        import subprocess, sys
        subprocess.run(["apt","-y","install","fonts-noto-cjk"], check=False, stdout=subprocess.DEVNULL)
        candidates += glob.glob("/usr/share/fonts/**/NotoSansCJK*.*", recursive=True)

    if not candidates:
        raise RuntimeError("找不到任何 CJK 字体，请手动提供 .ttf/.otf 路径")

    path = sorted(candidates)[0]  # Choose one
    fm.fontManager.addfont(path)
    name = fm.FontProperties(fname=path).get_name()

    # 2) Set as default font (Note: This must be set before drawing)
    rcParams["font.family"] = name
    rcParams["font.sans-serif"] = [name]
    rcParams["axes.unicode_minus"] = False

    # 3) Clear cache (sometimes old cache can lock up DejaVu)
    import shutil
    cachedir = matplotlib.get_cachedir()
    shutil.rmtree(cachedir, ignore_errors=True)
    os.makedirs(cachedir, exist_ok=True)  # ✅ 不然就会出现你那个 lock 报错
    fm.fontManager = fm._load_fontmanager(try_read_cache=False)


    print("CJK font set to:", name, "\nfile:", path)

use_cjk_font()

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

CJK = fm.FontProperties(family=plt.rcParams["font.family"])

# ================= ZAI: guides + loader + mapping (century-safe) =================

# ---- Paths ----
BASE   = Path("/data/dataset/layered data")
OUTDIR = Path("/data/analysis_out/zai");OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR = OUTDIR / "figs"; FIGDIR.mkdir(parents=True, exist_ok=True)

def _sanitize(name: str) -> str:
    return (str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")
                    .replace("：", "_").replace(":", "_"))

def savefig_here(filename: str, dpi=200):
    import matplotlib.pyplot as plt
    fname = _sanitize(filename)
    plt.savefig(FIGDIR / fname, dpi=dpi, bbox_inches="tight")
    print("[saved]", FIGDIR / fname)

# ---- collection guideline (Book title - Century - Genre) ----
GL_MC   = Path("/data/dataset/collection guideline - MC.txt")
GL_EMOC = Path("/data/dataset/collection guideline - EMoC.txt")

def parse_md_table(path: Path) -> pd.DataFrame:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if "|" in ln]
    # Remove alignment lines
    lines = [ln for ln in lines if not re.match(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$', ln)]
    if not lines: raise ValueError(f"{path} 不是 Markdown 表格")
    header = [c.strip().strip("*") for c in lines[0].strip("|").split("|")]
    rows   = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[1:]]
    df = pd.DataFrame(rows, columns=header)

    def pick(pats):
        for h in header:
            if re.search(pats, str(h)): return h
        return None

    col_title = pick(r"书名")
    col_cent  = pick(r"世纪")
    col_genre = pick(r"分类")
    if not (col_title and col_cent and col_genre):
        raise ValueError(f"{path} 缺少 书名/世纪/分类 列")

    out = pd.DataFrame({
        "title": df[col_title].astype(str).str.replace(r"\*\*", "", regex=True).str.strip(),
        "century": df[col_cent].astype(str).str.strip(),
        "genre_full": df[col_genre].astype(str).str.replace(r"\*\*", "", regex=True).str.strip(),
    })

    def to_coarse(s: str) -> str:
        s = re.sub(r"（[^）]*）|\([^)]*\)", "", s or "")  # First remove the explanations in (…) / (...)
        s = s.split("/")[0] if "/" in s else s           # Change to take the part before “/”
        return re.sub(r"\s+", " ", s).strip()

    out["genre_coarse"] = out["genre_full"].apply(to_coarse).replace({"": pd.NA})
    out["title_key"]    = out["title"].str.replace(r"\s+", "", regex=True)
    return out

g_mc   = parse_md_table(GL_MC).assign(epoch="MC")
g_emoc = parse_md_table(GL_EMOC).assign(epoch="EMoC")
guides = pd.concat([g_mc, g_emoc], ignore_index=True)
print("[guides] rows:", len(guides))

# ---- Reading ZHE four-layer merge (MC / EMoC) ----
def load_epoch(epoch_prefix):  # 'MC' or 'EMoC'
    frames = []
    for L in ['layer1','layer2','layer3','layer4']:
        f = BASE / f"{epoch_prefix}_zai_{L}.txt"
        if not f.exists():
            raise FileNotFoundError(f"找不到文件: {f}")
        df = pd.read_csv(f, sep='\t', dtype=str).fillna("")
        df['layer'] = L
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['epoch'] = epoch_prefix
    return df

df_all = pd.concat([load_epoch("MC"), load_epoch("EMoC")], ignore_index=True)
print("总行数:", len(df_all),
      "  MC:", (df_all["epoch"]=="MC").sum(),
      "  EMoC:", (df_all["epoch"]=="EMoC").sum())

# ---- Align file names ----
df_all["file_base"] = (df_all["file"].astype(str)
                       .str.replace(r"\.(xls[x]?|csv|txt)$","", regex=True))
df_all["file_key"]  = (df_all["file_base"].str.split("/").str[0]
                       .str.replace(r"\s+","", regex=True))

m = df_all.merge(
    guides[["epoch","title_key","century","genre_full","genre_coarse"]],
    left_on=["epoch","file_key"], right_on=["epoch","title_key"],
    how="left"
).drop(columns=["title_key"])

# Diagnosis: No matching list (to facilitate manual mapping)
unmatched = (m[m["century"].isna()][["epoch","file","file_key"]]
             .drop_duplicates().sort_values(["epoch","file"]))
unmatched.to_csv(OUTDIR/"unmatched_files_for_genre_mapping.csv", index=False)
print(f"[merge] 匹配成功 {len(m)-len(unmatched)} 条，未匹配 {len(unmatched)} 条")
print("[merge] 未匹配清单：", OUTDIR/"unmatched_files_for_genre_mapping.csv")

df_all = m  # cover back
df_all.to_csv(OUTDIR/"zai_all_with_genre_raw.csv", index=False)

# ---- Function Mapping: Sᶻᵃᶦ = {Z-LexV, Z-P, Z-AdvProg?, Z-Prog} ----
def map_to_state(row):
    z, cond = row.get('zai', ''), row.get('condition', '')
    txt = (row.get('context','') or '') + z + (row.get('next','') or '')
    if '在(P)' in z:
        return 'Z-P'
    if '在(VCL)' in txt or '所+在' in (cond or ''):
        return 'Z-LexV'
    if any(k in (cond or '') for k in ['正/方/值/適+在+否定','正/方/值/適+在2','在+動詞+著','在+动词+著/呢','触发+在+否定']):
        return 'Z-Prog'
    if '正/方/值/適+在1' in (cond or ''):
        return 'Z-AdvProg?'
    if '在(T)' in z:
        return 'Z-P'
    return 'Z-LexV'

df_all["state"] = df_all.apply(map_to_state, axis=1)
STATE_ORDER = ['Z-LexV','Z-P','Z-AdvProg?','Z-Prog']
df_all["state"] = pd.Categorical(df_all["state"], categories=STATE_ORDER, ordered=True)

# ---- Preview & Persistence ----
preview_cols = ['epoch','century','layer','file','context','zai','next','condition','state']
df_all.to_csv(OUTDIR / "zai_all_epochs_long.csv", index=False)
df_all[preview_cols].head(30).to_csv(OUTDIR / "zai_all_preview_head30.csv", index=False)
print("[dump] written:", OUTDIR / "zai_all_epochs_long.csv")

# ---- Grouping Dimensions (Century preferred; sort key provided) ----
def _century_sort_key(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else 10**9

GROUP_COL = "century" if df_all["century"].notna().any() else "epoch"
GROUPS = sorted(df_all[GROUP_COL].dropna().astype(str).unique(), key=_century_sort_key)
print("[group] GROUP_COL =", GROUP_COL, " | #groups =", len(GROUPS))

# ==== ZAI: CI (Wilson + Jeffreys), finer by century ====
from math import sqrt
try:
    from scipy.stats import beta
except Exception:
    beta = None

# ZAI's set of states
STATES = ["Z-LexV","Z-P","Z-AdvProg?","Z-Prog"]

# Use century if possible, otherwise revert to epoch
if "century" in df_all.columns and df_all["century"].notna().any():
    GROUP_COL = "century"
else:
    GROUP_COL = "epoch"

# Sorted uniformly by century number
if "GROUPS" not in globals():
    import re
    def _century_sort_key(s):
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else 10**9
    GROUPS = sorted(df_all[GROUP_COL].dropna().astype(str).unique(), key=_century_sort_key)

def wilson_ci(k, n, z=1.96):
    if n==0: return (np.nan,np.nan,np.nan)
    p=k/n; denom=1+z*z/n
    center=(p + z*z/(2*n))/denom
    half=(z/denom)*sqrt(max(p*(1-p)/n + z*z/(4*n*n), 0))
    return (center-half, center, center+half)

def jeffreys_ci(k, n, alpha=0.05):
    if (beta is None) or n==0: return (np.nan,np.nan)
    return (beta.ppf(alpha/2, k+0.5, n-k+0.5), beta.ppf(1-alpha/2, k+0.5, n-k+0.5))

rows=[]
# observed=False, same as ZHE's analysis
for gval, g in df_all.groupby(GROUP_COL, dropna=False, observed=False):
    n=len(g)
    vc=g["state"].value_counts()
    for s in STATES:
        k=int(vc.get(s,0))
        wL,wC,wU=wilson_ci(k,n)
        jL,jU=jeffreys_ci(k,n)
        rows.append({GROUP_COL:gval,"state":s,"n":n,"k":k,"p":(k/n if n else 0.0),
                     "W_lower":wL,"W_center":wC,"W_upper":wU,"J_lower":jL,"J_upper":jU})

prop = (pd.DataFrame(rows)
          .sort_values([GROUP_COL,"state"])
          .reset_index(drop=True))

# Persistence
prop.to_csv(OUTDIR/"zai_proportions_by_group.csv", index=False)

# — Save each state as a line graph + Wilson interval to FIGDIR —
for st in STATES:
    d = (prop[prop["state"]==st]
         .set_index(GROUP_COL)
         .reindex(GROUPS))  # time order
    x = np.arange(len(GROUPS))
    y  = d["p"].values
    yL = d["W_lower"].values
    yU = d["W_upper"].values

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.fill_between(x, yL, yU, alpha=.2)
    plt.title(f"{st} over {GROUP_COL}")
    plt.xticks(x, GROUPS, rotation=45)
    plt.ylabel("Proportion")
    plt.tight_layout()
    savefig_here(f"ZAI_ci_{st}_{GROUP_COL}.png")   # ← Save as ZAI_ prefix
    plt.show()

print("[ok] ZAI CI done ->", OUTDIR/"zai_proportions_by_group.csv")

def get_p_for(gval):
    sub = prop[prop[GROUP_COL]==gval].set_index("state")["p"]
    return sub.reindex(STATES).fillna(0).to_numpy(dtype=float)

def project_row_to_simplex(row):
    row = np.clip(row.astype(float), 0, None)
    s = row.sum()
    return row/s if s > 0 else np.full_like(row, 1.0/len(row))

def estimate_T(pt, pt1, iters=300, lr=0.2):
    pt  = np.asarray(pt,  dtype=float)
    pt1 = np.asarray(pt1, dtype=float)
    # Initialize a row-random matrix with uniform row distribution# Initialize a row-random matrix with uniform row distribution
    T = np.full((len(STATES), len(STATES)), 1.0/len(STATES), dtype=float)
    for _ in range(iters):
        # grad of || pt T - pt1 ||^2  w.r.t. T  ~=  outer(pt, (pt@T - pt1))
        diff = pt @ T - pt1
        grad = np.outer(pt, diff)
        T -= lr * grad
        # Project back the probability simplex for each row
        for i in range(T.shape[0]):
            T[i] = project_row_to_simplex(T[i])
    return T

# -------- Main process of century-interval Heatmaps (does not save graphs, only displays them)--------
edge_rows = []
pairs = list(zip(GROUPS, GROUPS[1:]))
if not pairs:
    print("[warn] GROUPS 不足两个，无法估计转移。")

for g0, g1 in pairs:
    p0, p1 = get_p_for(g0), get_p_for(g1)
    T = estimate_T(p0, p1)

    # write a long list
    for i, s_from in enumerate(STATES):
        for j, s_to in enumerate(STATES):
            edge_rows.append({
                "from": g0, "to": g1,
                "from_state": s_from, "to_state": s_to,
                "prob": float(T[i, j])
            })

    # Visualization
    fig = plt.figure()
    plt.imshow(np.clip(T, 0, 1), vmin=0, vmax=1, aspect='auto')
    plt.title(f"T: {g0} → {g1}")
    plt.xticks(range(len(STATES)), STATES, rotation=45)
    plt.yticks(range(len(STATES)), STATES)
    plt.colorbar(label="P(to | from)")
    # Numerical labeling (readable)
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            plt.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()

edges = pd.DataFrame(edge_rows)
edges.to_csv(OUTDIR / "zai_transitions_long.csv", index=False)
print("[done] transitions written:", OUTDIR / "zai_transitions_long.csv")

# ==== ZAI TM analysis (only MC → EMoC) ====

# ZAI's set of states
STATES = ["Z-LexV","Z-P","Z-AdvProg?","Z-Prog"]

# --- Take the state share vectors p(MC) and p(EMoC) at both ends of MC / EMoC ---
def _p_of_epoch(df, epoch_name):
    sub = df[df["epoch"]==epoch_name]["state"].value_counts()
    sub = sub.reindex(STATES).fillna(0).to_numpy(dtype=float)
    s = sub.sum()
    return sub / s if s>0 else np.full(len(STATES), 1/len(STATES), dtype=float)

p0 = _p_of_epoch(df_all, "MC")
p1 = _p_of_epoch(df_all, "EMoC")

# --- Simple row random matrix fitting: min || p0·T - p1 ||^2, progressively projecting to the simplex ---
def _project_row_to_simplex(row):
    row = np.clip(np.asarray(row, float), 0, None)
    s = row.sum()
    return row/s if s>0 else np.full_like(row, 1.0/len(row))

def _estimate_T(pt, pt1, iters=300, lr=0.2):
    pt, pt1 = np.asarray(pt, float), np.asarray(pt1, float)
    T = np.full((len(STATES), len(STATES)), 1.0/len(STATES), dtype=float)
    for _ in range(iters):
        diff = pt @ T - pt1
        T -= lr * np.outer(pt, diff)
        for i in range(T.shape[0]):
            T[i] = _project_row_to_simplex(T[i])
    return T

T = _estimate_T(p0, p1)

# --- ① Heatmap (display only, do not save) ---
plt.figure()
plt.imshow(np.clip(T,0,1), vmin=0, vmax=1, aspect='auto')
plt.title("T: MC → EMoC")
plt.xticks(range(len(STATES)), STATES, rotation=45)
plt.yticks(range(len(STATES)), STATES)
plt.colorbar(label="P(to | from)")
for i in range(len(STATES)):
    for j in range(len(STATES)):
        plt.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.show()

# --- ② Stacking Strips (Save) ---
def _plot_T_bars(T, title, fname):
    fig, ax = plt.subplots()
    bottoms = np.zeros(T.shape[0])
    for j, lab in enumerate(STATES):
        ax.bar(np.arange(T.shape[0]), T[:, j], bottom=bottoms, label=f"→ {lab}")
        bottoms += T[:, j]
    ax.set_xticks(np.arange(T.shape[0]))
    ax.set_xticklabels([f"from {s}" for s in STATES])
    ax.set_ylim(0,1); ax.set_ylabel("Probability"); ax.set_title(title)
    ax.legend(ncol=2, frameon=False); plt.tight_layout()
    savefig_here(fname); plt.show()

_plot_T_bars(T, title="T: MC → EMoC", fname="ZAI_TM_MC_to_EMoC_stackedbar.png")

# --- ③ Arrow Diagram (Save) ---
def _plot_T_arrows(T, title, fname, thr=0.05):
    fig, ax = plt.subplots()
    xs = np.arange(len(STATES))
    ax.scatter(xs, np.zeros_like(xs)); ax.scatter(xs, np.ones_like(xs))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] < thr: continue
            dx = xs[j] - xs[i]
            ax.arrow(xs[i], 0.03, dx, 0.94,
                     width=0.003*T[i,j], head_width=0.05, head_length=0.08,
                     length_includes_head=True, alpha=0.45)
    ax.set_xticks(xs); ax.set_xticklabels(STATES, rotation=45, ha="right")
    ax.set_yticks([0,1]); ax.set_yticklabels(["from","to"])
    ax.set_xlim(-0.5, len(STATES)-0.5); ax.set_ylim(-0.1, 1.1)
    ax.set_title(title); plt.tight_layout()
    savefig_here(fname); plt.show()

_plot_T_arrows(T, title="T: MC → EMoC", fname="ZAI_TM_MC_to_EMoC_arrows.png", thr=0.05)

# — Optional: Save the T value as well as a long table (for easy reuse).
edges = []
for i, s_from in enumerate(STATES):
    for j, s_to in enumerate(STATES):
        edges.append({"from":"MC","to":"EMoC","from_state":s_from,"to_state":s_to,"prob":float(T[i,j])})
pd.DataFrame(edges).to_csv(OUTDIR/"zai_TM_MC_to_EMoC.csv", index=False)
print("[TM] written:", OUTDIR/"zai_TM_MC_to_EMoC.csv")

def make_ca_table(df: pd.DataFrame, use_century=True) -> pd.DataFrame:
    STATES = ["Z-LexV","Z-P","Z-AdvProg?","Z-Prog"]
    group_col = "century" if (use_century and "century" in df.columns and df["century"].notna().any()) else "epoch"
    have_coarse = ("genre_coarse" in df.columns) and df["genre_coarse"].notna().any()

    if have_coarse:
        tab = (df.groupby([group_col,"genre_coarse","state"], dropna=False, observed=False)
                 .size().unstack("state").reindex(columns=STATES, fill_value=0).reset_index())
        gf = (df.dropna(subset=["genre_coarse"])
                .groupby([group_col,"genre_coarse"])["genre_full"]
                .apply(lambda s: "|".join(sorted(set(map(str,s))))))
        tab = tab.merge(gf.rename("genre_full_report").reset_index(),
                        on=[group_col,"genre_coarse"], how="left")
    else:
        tab = (df.groupby([group_col,"state"], dropna=False, observed=False)
                 .size().unstack("state").reindex(columns=STATES, fill_value=0).reset_index())
    return tab

HAVE_COARSE = ("genre_coarse" in df_all.columns) and df_all["genre_coarse"].notna().any()

if HAVE_COARSE:
    ca_tab = (df_all.groupby([GROUP_COL,"genre_coarse","state"], dropna=False)
                     .size().unstack("state").fillna(0).astype(int).reset_index())
    # Report: Complete Genre Collection
    gf = (df_all.dropna(subset=["genre_coarse"])
                .groupby([GROUP_COL,"genre_coarse"])["genre_full"]
                .apply(lambda s: "|".join(sorted(set(map(str,s))))))
    ca_tab = ca_tab.merge(gf.rename("genre_full_report").reset_index(),
                          on=[GROUP_COL,"genre_coarse"], how="left")
else:
    ca_tab = (df_all.groupby([GROUP_COL,"state"], dropna=False)
                     .size().unstack("state").fillna(0).astype(int).reset_index())
ca_tab.to_csv(OUTDIR/"zai_ca_table.csv", index=False)

def run_ca_and_plot(ca_tab: pd.DataFrame, title_prefix="CA", annotate_top=30, fname_prefix="ca"):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, prince

    # ZAI's set of states
    STATES = ["Z-LexV","Z-P","Z-AdvProg?","Z-Prog"]

    # Row label column: prioritize century, then epoch if none; then add genre_coarse (if any).
    label_cols = []
    if "century" in ca_tab.columns and ca_tab["century"].notna().any():
        label_cols.append("century")
    elif "epoch" in ca_tab.columns:
        label_cols.append("epoch")
    if "genre_coarse" in ca_tab.columns:
        label_cols.append("genre_coarse")

    row_labels = (ca_tab[label_cols].astype(str).agg("_".join, axis=1)
                  if label_cols else pd.Series(range(len(ca_tab)), dtype=str))

    # Numerical matrix (only four state columns are used to ensure pure numbers)
    X = ca_tab.reindex(columns=STATES, fill_value=0)
    row_mask = X.sum(axis=1) > 0
    col_mask = X.sum(axis=0) > 0
    X = X.loc[row_mask, col_mask]
    row_labels = row_labels.loc[row_mask]

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("CA 输入全为 0，请检查体裁/时期是否过稀。")

    n_comp = max(1, min(2, min(X.shape[0]-1, X.shape[1]-1)))
    ca = prince.CA(n_components=n_comp, random_state=42).fit(X)

    rows = ca.row_coordinates(X); rows.index = row_labels.values
    cols = ca.column_coordinates(X)

    # Only the top K rows by frequency are highlighted
    freq = X.sum(axis=1).to_numpy()
    K = min(int(annotate_top), len(freq))
    keep_pos = np.argsort(-freq)[:K]
    top_mask = np.zeros(len(freq), dtype=bool); top_mask[keep_pos] = True

    if n_comp >= 2:
        plt.figure()
        plt.scatter(rows[0], rows[1], s=18, alpha=0.5)
        for name, x, y in zip(rows.index[top_mask], rows[0][top_mask], rows[1][top_mask]):
            plt.annotate(name, (x, y), fontsize=9)
        plt.title(f"{title_prefix} rows"); plt.xlabel("Dim1"); plt.ylabel("Dim2"); plt.tight_layout()
        savefig_here(f"{fname_prefix}_rows.png"); plt.show()

        plt.figure()
        plt.scatter(cols[0], cols[1], s=40)
        for name, x, y in zip(cols.index, cols[0], cols[1]):
            plt.annotate(name, (x, y), fontsize=10)
        plt.title(f"{title_prefix} columns (states)"); plt.xlabel("Dim1"); plt.ylabel("Dim2"); plt.tight_layout()
        savefig_here(f"{fname_prefix}_cols.png"); plt.show()
    else:
        # band-like misalignment based on x to avoid overlap at x
        x = rows[0].to_numpy()
        order = np.argsort(x)
        y = np.zeros_like(x, dtype=float)

        band_height = 0.06   # band spacing
        band_span   = 5

        cluster_x = None
        band_idx  = 0
        for idx in order:
            if cluster_x is None or abs(x[idx] - cluster_x) > 0.03:
                # New Clusters
                cluster_x = x[idx]
                band_idx  = 0
            else:
                band_idx += 1
            #-2, -1, 0, 1, 2... symmetrically spread
            y[idx] = ((band_idx % band_span) - (band_span-1)/2) * band_height

        plt.figure()
        plt.scatter(x, y, s=18, alpha=0.5)
        for name, xi, yi in zip(rows.index[top_mask], rows[0][top_mask], y[top_mask]):
            plt.annotate(name, (xi, yi), fontsize=9,
                        xytext=(0, 6), textcoords="offset points", ha="center")
        plt.title(f"{title_prefix} rows (1D)")
        plt.xlabel("Dim1"); plt.yticks([]); plt.tight_layout()
        savefig_here(f"{fname_prefix}_rows_1D.png"); plt.show()

        plt.title(f"{title_prefix} rows (1D)"); plt.xlabel("Dim1"); plt.yticks([]); plt.tight_layout()
        savefig_here(f"{fname_prefix}_rows_1D.png"); plt.show()

        plt.figure()
        plt.scatter(cols[0], np.zeros(len(cols)), s=40)
        for name, x in zip(cols.index, cols[0]):
            plt.annotate(name, (x, 0), fontsize=10, xytext=(0,6), textcoords="offset points", ha="center")
        plt.title(f"{title_prefix} columns (1D)"); plt.xlabel("Dim1"); plt.yticks([]); plt.tight_layout()
        savefig_here(f"{fname_prefix}_cols_1D.png"); plt.show()

from adjustText import adjust_text
import matplotlib.patheffects as pe

def _scatter_with_labels(x, y, labels, top_mask=None, title="", xlabel="Dim1", ylabel="Dim2",
                         repel_iters=300, point_size_bg=14, point_size_fg=24, fontsize=9):
    x = np.asarray(x); y = np.asarray(y); labels = np.asarray(labels)
    if top_mask is None: top_mask = np.ones_like(x, dtype=bool)

    plt.figure()
    # Background plots
    plt.scatter(x[~top_mask], y[~top_mask], s=point_size_bg, alpha=0.35)
    # Foreground points (will be marked)
    plt.scatter(x[top_mask], y[top_mask], s=point_size_fg, alpha=0.8)

    texts = []
    for xi, yi, lab in zip(x[top_mask], y[top_mask], labels[top_mask]):
        t = plt.text(xi, yi, lab, fontsize=fontsize, ha="center", va="center",
                     path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
        texts.append(t)

    if texts:
        adjust_text(
            texts,
            # Allow bidirectional movement of dots/words; increase the "weight" setting.
            only_move={'points':'xy','text':'xy'},
            force_points=0.6, force_text=0.6,
            expand_points=(1.6,1.6), expand_text=(1.6,1.6),
            autoalign='xy',
            lim=repel_iters,  # More iterations
            arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.35, shrinkA=6, shrinkB=6)
        )

    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.show()


def run_ca_and_plot(ca_tab: pd.DataFrame, title_prefix="CA", annotate_top=30, fname_prefix="ca"):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, prince
    STATES = ["Z-LexV","Z-P","Z-AdvProg?","Z-Prog"]

    # Labels (prefer century, epoch if none; with genre_coarse)
    label_cols = []
    if "century" in ca_tab.columns and ca_tab["century"].notna().any():
        label_cols.append("century")
    elif "epoch" in ca_tab.columns:
        label_cols.append("epoch")
    if "genre_coarse" in ca_tab.columns:
        label_cols.append("genre_coarse")

    row_labels = (ca_tab[label_cols].astype(str).agg("_".join, axis=1)
                  if label_cols else pd.Series(range(len(ca_tab)), dtype=str))

    # Numeric matrix
    X = ca_tab.reindex(columns=STATES, fill_value=0)
    row_mask = X.sum(axis=1) > 0
    col_mask = X.sum(axis=0) > 0
    X = X.loc[row_mask, col_mask]
    row_labels = row_labels.loc[row_mask]

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("CA 输入全为 0，请检查体裁/时期是否过稀。")

    n_comp = max(1, min(2, min(X.shape[0]-1, X.shape[1]-1)))
    ca = prince.CA(n_components=n_comp, random_state=42).fit(X)

    rows = ca.row_coordinates(X); rows.index = row_labels.values
    cols = ca.column_coordinates(X)

    # Only the top K rows are marked by frequency.
    freq = X.sum(axis=1).to_numpy()
    K = min(int(annotate_top), len(freq))
    keep_pos = np.argsort(-freq)[:K]
    top_mask = np.zeros(len(freq), dtype=bool); top_mask[keep_pos] = True

    # === Perform micro-clustering on overlapping/nearest points (initially with small radial divergence to reduce the load on adjustText) ===
    def _micro_defuse(xy, eps=1e-8, radius=0.012):
        # Assign points with identical/very similar coordinates to smaller circles.
        xy = np.asarray(xy)
        out = xy.copy()
        key = np.round(xy, 6)   # Merging nearly overlapping points with 6 decimal places
        # Finding a cluster
        uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        for i, c in enumerate(counts):
            if c <= 1: continue
            idx = np.where(inv == i)[0]
            # Evenly distributed on the small circumference
            ang = np.linspace(0, 2*np.pi, c, endpoint=False)
            out[idx, 0] += radius * np.cos(ang)
            out[idx, 1] += radius * np.sin(ang)
        return out

    row_xy = np.c_[rows[0].to_numpy(), rows[1].to_numpy()] if n_comp >= 2 else np.c_[rows[0].to_numpy(), np.zeros(len(rows))]
    col_xy = np.c_[cols[0].to_numpy(), cols[1].to_numpy()] if n_comp >= 2 else np.c_[cols[0].to_numpy(), np.zeros(len(cols))]

    row_xy = _micro_defuse(row_xy, radius=0.015)
    col_xy = _micro_defuse(col_xy, radius=0.015)

    # --- Drawing ---
    if n_comp >= 2:
        _scatter_with_labels(
            x=row_xy[:,0], y=row_xy[:,1],
            labels=rows.index.to_numpy(),
            top_mask=top_mask,
            title=f"{title_prefix} rows"
        )
        _scatter_with_labels(
            x=col_xy[:,0], y=col_xy[:,1],
            labels=cols.index.to_numpy(),
            top_mask=np.ones(len(cols), dtype=bool),
            title=f"{title_prefix} columns (states)"
        )
    else:
        # 1D: Add minimal jitter to the y-axis while limiting the main vertical text avoidance
        jitter_rows = (np.random.rand(len(rows))*0.06 - 0.03)
        _scatter_with_labels(
            x=row_xy[:,0], y=jitter_rows,
            labels=rows.index.to_numpy(),
            top_mask=top_mask,
            title=f"{title_prefix} rows (1D)", ylabel=""
        )
        _scatter_with_labels(
            x=col_xy[:,0], y=np.zeros(len(cols)),
            labels=cols.index.to_numpy(),
            top_mask=np.ones(len(cols), dtype=bool),
            title=f"{title_prefix} columns (1D)", ylabel=""
        )

df_mc = df_all[df_all["epoch"]=="MC"].copy()
df_emoc = df_all[df_all["epoch"]=="EMoC"].copy()
ca_mc = make_ca_table(df_mc, use_century=True)
ca_emoc = make_ca_table(df_emoc, use_century=True)
run_ca_and_plot(ca_mc, title_prefix="CA (MC: century×genre)", annotate_top=30, fname_prefix="CA_MC")
run_ca_and_plot(ca_emoc, title_prefix="CA (EMoC: century×genre)", annotate_top=30, fname_prefix="CA_EMoC")

# ================== CA: Interactive + PNG rollback save: One-time output of MC/EMoC ==================
import plotly.express as px
import plotly.io as pio
from prince import CA

STATES = ["Z-LexV","Z-P","Z-AdvProg?","Z-Prog"]

def ca_interactive_scatter(rows, cols, title_prefix="CA"):
    has_dim2_rows = 1 in rows.columns
    has_dim2_cols = 1 in cols.columns
    rows_df = pd.DataFrame({
        "x": rows[0].to_numpy(),
        "y": rows[1].to_numpy() if has_dim2_rows else np.zeros(len(rows)),
        "label": rows.index.astype(str),
        "type": "row",
    })
    cols_df = pd.DataFrame({
        "x": cols[0].to_numpy(),
        "y": cols[1].to_numpy() if has_dim2_cols else np.zeros(len(cols)),
        "label": cols.index.astype(str),
        "type": "column",
    })
    all_df = pd.concat([rows_df, cols_df], ignore_index=True)

    fig = px.scatter(
        all_df, x="x", y="y",
        color="type", symbol="type",
        hover_name="label",
        title=title_prefix,
    )
    fig.update_traces(selector=dict(name="column"), marker=dict(size=10))
    fig.update_traces(selector=dict(name="row"),    marker=dict(size=6, opacity=0.7))
    fig.update_xaxes(title_text="Dim1")
    fig.update_yaxes(title_text=("Dim2" if (has_dim2_rows or has_dim2_cols) else ""))
    return fig

def _fit_rows_cols_from_ca_tab(ca_tab: pd.DataFrame):
    label_cols = []
    if "century" in ca_tab.columns and ca_tab["century"].notna().any():
        label_cols.append("century")
    elif "epoch" in ca_tab.columns:
        label_cols.append("epoch")
    if "genre_coarse" in ca_tab.columns:
        label_cols.append("genre_coarse")
    row_labels = (ca_tab[label_cols].astype(str).agg("_".join, axis=1)
                  if label_cols else pd.Series(range(len(ca_tab)), dtype=str))
    X = ca_tab.reindex(columns=STATES, fill_value=0)
    row_mask = X.sum(axis=1) > 0
    col_mask = X.sum(axis=0) > 0
    X = X.loc[row_mask, col_mask]
    row_labels = row_labels.loc[row_mask]
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("CA 输入全为 0。")
    n_comp = max(1, min(2, min(X.shape[0]-1, X.shape[1]-1)))
    ca = CA(n_components=n_comp, random_state=42).fit(X)
    rows = ca.row_coordinates(X); rows.index = row_labels.values
    cols = ca.column_coordinates(X)
    return rows, cols, X

# —— Matplotlib rollback: plot rows/cols as a static PNG (optionally only label the first _k rows).
def save_png_fallback(rows, cols, png_path, title, top_k=30):
    # 2D or 1D
    rows_x = rows[0].to_numpy()
    rows_y = rows[1].to_numpy() if (1 in rows.columns) else np.zeros(len(rows))
    cols_x = cols[0].to_numpy()
    cols_y = cols[1].to_numpy() if (1 in cols.columns) else np.zeros(len(cols))

    plt.figure(figsize=(8.5, 6.0), dpi=150)
    # Plots
    plt.scatter(rows_x, rows_y, s=16, alpha=0.55, label="row")
    plt.scatter(cols_x, cols_y, s=36, alpha=0.9, marker="^", label="column")
    # Only add tags to the top_k most frequent elements in the row (to avoid Chinese characters being squeezed out).
    freq = np.zeros(len(rows))
    if hasattr(rows, "index"):
        # If the upstream retains the frequency, it can be passed in from X.sum(axis=1); here we use a simplification to "be consistent with the original implementation": omit the ones closer to the origin
        # A more stable approach: also return X.sum(axis=1), which is omitted here.
        pass
    # Sort by approximate magnitude of x+y (all-purposed)
    approx = np.abs(rows_x) + np.abs(rows_y)
    keep = np.argsort(-approx)[: min(top_k, len(rows_x))]
    for i in keep:
        plt.annotate(str(rows.index[i]), (rows_x[i], rows_y[i]), fontsize=9,
                     xytext=(0, 6), textcoords="offset points", ha="center")

    plt.title(title)
    plt.xlabel("Dim1"); plt.ylabel("Dim2" if (1 in rows.columns or 1 in cols.columns) else "")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

# ========= Simultaneously output MC / EMoC: HTML + PNG =========
for name, ca_source in [("MC", ca_mc), ("EMoC", ca_emoc)]:
    rows, cols, X = _fit_rows_cols_from_ca_tab(ca_source)
    fig = ca_interactive_scatter(rows, cols, title_prefix=f"CA ({name}: century×genre)")

    html_path = FIGDIR / f"CA_{name}.html"
    png_path  = FIGDIR / f"CA_{name}.png"

    # Interactive version (no additional dependencies required)
    pio.write_html(fig, str(html_path))
    print(f"[ok] HTML saved -> {html_path}")

    # Static PNG (no longer depends on kaleido; use Matplotlib as a fallback)
    save_png_fallback(rows, cols, png_path, title=f"CA ({name}: century×genre)", top_k=30)
    print(f"[ok] PNG saved  -> {png_path}")

    # Show us if you want to see the interaction
    fig.show()

# ==== ZAI: dispersion（period ×[genre] × state）====

# ZAI's set of states
STATES = ["Z-LexV","Z-P","Z-AdvProg?","Z-Prog"]

HAVE_COARSE = ("genre_coarse" in df_all.columns) and df_all["genre_coarse"].notna().any()
group_keys  = [GROUP_COL] + (["genre_coarse"] if HAVE_COARSE else [])

disp_rows = []
for keys, sub_all in df_all.groupby(group_keys, dropna=False, observed=False):
    total_texts = sub_all["file"].nunique()  # the total number of texts in the period×[genre] bucket

    for st, sub in sub_all.groupby("state", dropna=False):
        tokens  = int(len(sub))
        n_texts = int(sub["file"].nunique()) if tokens > 0 else 0

        if tokens > 0:
            by_file = sub.groupby("file").size()
            top_text_share = float(by_file.max() / by_file.sum())

            p = (by_file / by_file.sum()).to_numpy(float)
            H = float((p * p).sum())      # Herfindahl (the larger the number, the more concentrated it becomes)
            U = 1.0 - H                   # Uniformity
            N = max(1, n_texts)
            dp = 1.0 - (np.abs(p - 1.0/N).sum() / (2.0 - 2.0/N))  # Gries DP (0–1, the larger the value, the more uniform the distribution)
        else:
            top_text_share = float("nan"); H = float("nan"); U = float("nan"); dp = float("nan")

        row = {
            "state": st,
            "tokens": tokens,
            "n_texts": n_texts,
            "top_text_share": top_text_share,
            "doc_coverage": (n_texts / total_texts) if total_texts > 0 else float("nan"),
            "H_concentration": H,
            "U_evenness": U,
            "DP_Gries": dp,
        }
        # Write back the group key (period / [genre])
        if isinstance(keys, tuple):
            for k, v in zip(group_keys, keys): row[k] = v
        else:
            row[group_keys[0]] = keys

        disp_rows.append(row)

disp_zhe = (pd.DataFrame(disp_rows)
            .astype({"tokens":"Int64","n_texts":"Int64"})
            .sort_values(group_keys + ["state"])
           )

# —— store and print ——
disp_zhe.to_csv(OUTDIR / "zai_dispersion_report.csv", index=False)
print("[ZAI-dispersion] saved:", OUTDIR / "zai_dispersion_report.csv")

# ==== Lightweight Visualization: DP and top_text_share Heatmap ====
# 1) DP(Gries) by group × state
dp_tab = (disp_zhe.pivot_table(index=group_keys, columns="state", values="DP_Gries", aggfunc="mean")
                    .reindex(columns=STATES))
# Convert index to string for easier display
if len(group_keys) == 2:
    dp_tab.index = dp_tab.index.map(lambda t: f"{t[0]}_{t[1]}")
else:
    dp_tab.index = dp_tab.index.astype(str)

plt.figure(figsize=(max(6, 0.6*len(STATES)+3), 1.0 + 0.35*len(dp_tab)))
im = plt.imshow(dp_tab.to_numpy(), aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, label="DP (Gries, higher = more even)")
plt.xticks(range(len(STATES)), STATES, rotation=45, ha="right")
plt.yticks(range(len(dp_tab.index)), dp_tab.index)
plt.title("ZAI dispersion (DP) by period × state")
plt.tight_layout(); savefig_here("ZAI_dispersion_DP_heat.png"); plt.show()

# 2) Top_text_share by group × state (to check if it is "piled up by individual texts")
ts_tab = (disp_zhe.pivot_table(index=group_keys, columns="state", values="top_text_share", aggfunc="mean")
                    .reindex(columns=STATES))
if len(group_keys) == 2:
    ts_tab.index = ts_tab.index.map(lambda t: f"{t[0]}_{t[1]}")
else:
    ts_tab.index = ts_tab.index.astype(str)

plt.figure(figsize=(max(6, 0.6*len(STATES)+3), 1.0 + 0.35*len(ts_tab)))
im = plt.imshow(ts_tab.to_numpy(), aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, label="top_text_share (0–1, lower = less piled)")
plt.xticks(range(len(STATES)), STATES, rotation=45, ha="right")
plt.yticks(range(len(ts_tab.index)), ts_tab.index)
plt.title("ZAI top_text_share by period × state")
plt.tight_layout(); savefig_here("ZAI_dispersion_topshare_heat.png"); plt.show()

"""
Layer 1 (l1)"""

# —— Paths re-confirmed ——

FIGDIR = OUTDIR / "figs"; FIGDIR.mkdir(parents=True, exist_ok=True)
def savefig_here(name):
    import matplotlib.pyplot as plt
    plt.savefig(FIGDIR / name, dpi=180, bbox_inches="tight")

# Take only Layer1
l1 = df_all[df_all["layer"].str.lower().eq("layer1")].copy()

# Specify `cond_sub` and map it to 6 classes.
l1["cond_sub"] = l1.get("cond_sub", "").fillna("").astype(str).str.strip()

BUCKETS = [
    "L1_trigZai_V",        # Triggered by + 在 + V（V is adjacent)
    "L1_trigZai_neg",      # Triggered by + 在 + negation ...
    "L1_zai_V_zhe",        # 在 + V (+0–3 characters/words) + 著/呢
    "L1_zhengzai_V",       # 正在 + V
    "L1_trigZai_V_loose",  # Triggered by + 在 … + V（loosen 1 character/word）
    "L1_other",            # The rest Layer1
]

def map_bucket(s: str) -> str:
    if   s.startswith("L1_trigZai_V_loose"): return "L1_trigZai_V_loose"
    elif s.startswith("L1_trigZai_V"):       return "L1_trigZai_V"
    elif s.startswith("L1_trigZai_neg"):     return "L1_trigZai_neg"
    elif s.startswith("L1_zai_V_zhe"):       return "L1_zai_V_zhe"
    elif s.startswith("L1_zhengzai_V"):      return "L1_zhengzai_V"
    else:                                    return "L1_other"

l1["l1_bucket"] = l1["cond_sub"].apply(map_bucket)

# Merge and label the candidate entities of the progressive aspect
PROG_CAND = {"L1_trigZai_V","L1_trigZai_neg","L1_zai_V_zhe","L1_zhengzai_V","L1_trigZai_V_loose"}
l1["is_prog_cand"] = l1["l1_bucket"].isin(PROG_CAND)

# ==== ZAI-Layer1: CI ====
def wilson_ci(k, n, z=1.96):
    if n==0: return (np.nan,np.nan,np.nan)
    p=k/n; denom=1+z*z/n
    center=(p + z*z/(2*n))/denom
    half=(z/denom)*sqrt(max(p*(1-p)/n + z*z/(4*n*n), 0))
    return (center-half, center, center+half)

def jeffreys_ci(k, n, alpha=0.05):
    if (beta is None) or n==0: return (np.nan,np.nan)
    return (beta.ppf(alpha/2, k+0.5, n-k+0.5), beta.ppf(1-alpha/2, k+0.5, n-k+0.5))

# Bucketed CI
rows=[]
for gval, g in l1.groupby(GROUP_COL, dropna=False, observed=False):
    n = len(g)  # Based on l1's share
    vc = g["l1_bucket"].value_counts()
    for b in BUCKETS:
        k = int(vc.get(b, 0))
        wL,wC,wU = wilson_ci(k,n)
        jL,jU    = jeffreys_ci(k,n)
        rows.append({GROUP_COL:gval,"bucket":b,"n":n,"k":k,"p":k/n,
                     "W_lower":wL,"W_center":wC,"W_upper":wU,
                     "J_lower":jL,"J_upper":jU})
l1_prop = pd.DataFrame(rows).sort_values([GROUP_COL,"bucket"]).reset_index(drop=True)
l1_prop.to_csv(OUTDIR/"zai_L1_proportions_by_group.csv", index=False)

# merge to a line: candidate entities vs others
rows=[]
for gval, g in l1.groupby(GROUP_COL, dropna=False, observed=False):
    n = len(g)
    k = int(g["is_prog_cand"].sum())
    wL,wC,wU = wilson_ci(k,n); jL,jU = jeffreys_ci(k,n)
    rows.append({GROUP_COL:gval,"series":"Prog-candidate", "n":n,"k":k,"p":k/n,
                 "W_lower":wL,"W_center":wC,"W_upper":wU,
                 "J_lower":jL,"J_upper":jU})
l1_progline = pd.DataFrame(rows)
l1_progline.to_csv(OUTDIR/"zai_L1_prog_candidate_line.csv", index=False)

# Uniform horizontal axis order
def _century_key(s):
    m = re.search(r'(\d+)', str(s));   return int(m.group(1)) if m else 10**9
GROUPS = sorted(l1[GROUP_COL].dropna().astype(str).unique(), key=_century_key)

# One per bucket
for b in BUCKETS:
    d = (l1_prop[l1_prop["bucket"]==b]
         .set_index(GROUP_COL).reindex(GROUPS))
    x = np.arange(len(GROUPS))
    y  = d["p"].to_numpy()
    yL = d["W_lower"].to_numpy()
    yU = d["W_upper"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.fill_between(x, yL, yU, alpha=.2)
    plt.title(f"Layer1 {b} over {GROUP_COL}")
    plt.xticks(x, GROUPS, rotation=45)
    plt.ylabel("Proportion (within Layer1)")
    plt.tight_layout()
    savefig_here(f"L1_{b}_over_{GROUP_COL}.png")
    plt.show()

# Merge "candidate entities" into one line
d = l1_progline.set_index(GROUP_COL).reindex(GROUPS)
x = np.arange(len(GROUPS))
y, yL, yU = d["p"].to_numpy(), d["W_lower"].to_numpy(), d["W_upper"].to_numpy()

plt.figure()
plt.plot(x, y, marker="o")
plt.fill_between(x, yL, yU, alpha=.2)
plt.title(f"Layer1 Prog-candidate over {GROUP_COL}")
plt.xticks(x, GROUPS, rotation=45)
plt.ylabel("Proportion (within Layer1)")
plt.tight_layout()
savefig_here(f"L1_ProgCandidate_over_{GROUP_COL}.png")
plt.show()

# --- Dependencies: there already are obtained l1 (Layer1 sub-table), BUCKETS, GROUP_COL, GROUPS, OUTDIR ---
FIGDIR = OUTDIR / "figs"; FIGDIR.mkdir(parents=True, exist_ok=True)
def savefig_here(name): plt.savefig(FIGDIR / name, dpi=180, bbox_inches="tight")

# 1) create the share table first for l1 (equivalent to the prop in the above large TM analysis, but with column names changed to bucket).
rows=[]
for gval, g in l1.groupby(GROUP_COL, dropna=False, observed=False):
    n = len(g)
    vc = g["l1_bucket"].value_counts()
    for b in BUCKETS:
        k = int(vc.get(b, 0))
        rows.append({GROUP_COL:gval, "bucket": b, "n": n, "k": k, "p": k/n if n else 0.0})
prop_L1 = pd.DataFrame(rows).sort_values([GROUP_COL,"bucket"]).reset_index(drop=True)

# 2) TM tool (make a reusable, general-purpose version of the above TM analysis)
def _project_row_to_simplex(row):
    row = np.clip(np.asarray(row, float), 0, None)
    s = row.sum()
    return row/s if s>0 else np.full_like(row, 1.0/len(row))

def _estimate_T(pt, pt1, iters=300, lr=0.2):
    pt, pt1 = np.asarray(pt, float), np.asarray(pt1, float)
    T = np.full((len(pt), len(pt)), 1.0/len(pt))
    for _ in range(iters):
        diff = pt @ T - pt1
        T   -= lr * np.outer(pt, diff)
        for i in range(T.shape[0]):
            T[i] = _project_row_to_simplex(T[i])
    return T

def _get_p_for_bucket(prop_df, group_value, buckets):
    sub = prop_df[prop_df[GROUP_COL]==group_value].set_index("bucket")["p"]
    return sub.reindex(buckets).fillna(0).to_numpy(float)

def _plot_T_heat(T, labels, title, fname=None):
    plt.figure()
    plt.imshow(np.clip(T,0,1), vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="P(to | from)")
    plt.title(title); plt.tight_layout()
    if fname: savefig_here(fname)
    plt.show()

def _plot_T_bars_arrows(T, labels, title, fname_bar=None, fname_arr=None, thr=0.05):
    # stacking bars
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    left = np.zeros(T.shape[0])
    for j, lab in enumerate(labels):
        ax.barh(np.arange(T.shape[0]), T[:, j], left=left, label=f"→ {lab}")
        left += T[:, j]
    ax.set_yticks(np.arange(T.shape[0]))
    ax.set_yticklabels([f"from {lab}" for lab in labels])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.legend(ncol=2, frameon=False, fontsize=9)
    plt.tight_layout()
    if fname_bar: savefig_here(fname_bar.replace("_stackedbar","_stackedbarH"))
    plt.show()


    # Arrows
    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    ax.scatter(xs, np.zeros_like(xs)); ax.scatter(xs, np.ones_like(xs))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] < thr: continue
            dx = xs[j] - xs[i]
            ax.arrow(xs[i], 0.03, dx, 0.94, width=0.003*T[i,j],
                     head_width=0.05, head_length=0.08, length_includes_head=True, alpha=0.45)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks([0,1]); ax.set_yticklabels(["from","to"])
    ax.set_xlim(-0.5, len(labels)-0.5); ax.set_ylim(-0.1, 1.1)
    ax.set_title(title); plt.tight_layout()
    if fname_arr: savefig_here(fname_arr)
    plt.show()

# 3) Estimate the transition matrix of adjacent periods in l1 and plot it
edge_rows=[]
pairs = list(zip(GROUPS, GROUPS[1:]))
for g0, g1 in pairs:
    p0 = _get_p_for_bucket(prop_L1, g0, BUCKETS)
    p1 = _get_p_for_bucket(prop_L1, g1, BUCKETS)
    T  = _estimate_T(p0, p1, iters=300, lr=0.2)

    # long list
    for i, b_from in enumerate(BUCKETS):
        for j, b_to in enumerate(BUCKETS):
            edge_rows.append({"from":g0,"to":g1,"from_bucket":b_from,"to_bucket":b_to,"prob":float(T[i,j])})

    title = f"L1-TM: {g0} → {g1}"
    _plot_T_heat(T, BUCKETS, title, fname=f"L1_TM_{g0}_to_{g1}_heat.png")
    _plot_T_bars_arrows(T, BUCKETS, title,
                        fname_bar=f"L1_TM_{g0}_to_{g1}_stackedbar.png",
                        fname_arr=f"L1_TM_{g0}_to_{g1}_arrows.png", thr=0.05)

pd.DataFrame(edge_rows).to_csv(OUTDIR/"zai_L1_transitions_long.csv", index=False)
print("[L1-TM] written:", OUTDIR/"zai_L1_transitions_long.csv")

# ca_L1 is the wide table you get from make_ca_table_L1.
num_cols = [c for c in ca_L1.columns if c not in {"epoch","century","genre_coarse","genre_full_report"}]
X = ca_L1[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()

fig, ax = plt.subplots(figsize=(8, 0.4*len(ca_L1)))  # Highly follow-up adjustment
im = ax.imshow(X, aspect="auto")
ax.set_yticks(range(len(ca_L1)))

row_lab_cols = [c for c in ["century","epoch","genre_coarse"] if c in ca_L1.columns]
row_labels = ca_L1[row_lab_cols].astype(str).agg("_".join, axis=1) if row_lab_cols else pd.Series(range(len(ca_L1)))
ax.set_yticklabels(row_labels)
ax.set_xticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha="right")
fig.colorbar(im, ax=ax, label="count")
plt.title("L1 wide matrix preview (rows = period×genre; cols = buckets)")
plt.tight_layout(); plt.show()

# — Table structure: Period × Genre (if applicable) × l1_bucket → Counting matrix
def make_ca_table_L1(df_l1: pd.DataFrame, use_century=True) -> pd.DataFrame:
    group_col   = "century" if (use_century and "century" in df_l1.columns and df_l1["century"].notna().any()) else "epoch"
    have_coarse = ("genre_coarse" in df_l1.columns) and df_l1["genre_coarse"].notna().any()

    if have_coarse:
        tab = (df_l1.groupby([group_col,"genre_coarse","l1_bucket"], dropna=False, observed=False)
                     .size()
                     .unstack("l1_bucket")
                     .reindex(columns=BUCKETS, fill_value=0)
                     .fillna(0)                               # ★ 新增：抹平已有 NaN
                     .reset_index())
        gf  = (df_l1.dropna(subset=["genre_coarse"])
                    .groupby([group_col,"genre_coarse"])["genre_full"]
                    .apply(lambda s: "|".join(sorted(set(map(str,s))))))
        tab = tab.merge(gf.rename("genre_full_report").reset_index(),
                        on=[group_col,"genre_coarse"], how="left")
    else:
        tab = (df_l1.groupby([group_col,"l1_bucket"], dropna=False, observed=False)
                     .size()
                     .unstack("l1_bucket")
                     .reindex(columns=BUCKETS, fill_value=0)
                     .fillna(0)                               # new added
                     .reset_index())
    return tab


ca_L1 = make_ca_table_L1(l1, use_century=True)
ca_L1.to_csv(OUTDIR/"zai_L1_ca_table.csv", index=False)

def _scatter_with_labels(ax, x, y, labels, top_mask=None, point_bg=14, point_fg=22, fontsize=9):
    x = np.asarray(x); y = np.asarray(y); labels = np.asarray(labels)
    if top_mask is None: top_mask = np.ones_like(x, dtype=bool)

    # Background plots
    ax.scatter(x[~top_mask], y[~top_mask], s=point_bg, alpha=0.35)
    # Foreground plots (need to be marked)
    ax.scatter(x[top_mask], y[top_mask], s=point_fg, alpha=0.8)

    texts = []
    for xi, yi, lab in zip(x[top_mask], y[top_mask], labels[top_mask]):
        t = ax.text(xi, yi, lab, fontsize=fontsize, ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        texts.append(t)

    if texts:
        adjust_text(
            texts,
            only_move={'points':'xy','text':'xy'},
            force_points=0.6, force_text=0.6,
            expand_points=(1.6,1.6), expand_text=(1.6,1.6),
            autoalign='xy',
            arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.4, shrinkA=6, shrinkB=6),
        )


def run_ca_and_plot_generic(ca_tab: pd.DataFrame, title_prefix="CA", annotate_top=30, fname_prefix="ca"):
    non_num = {"epoch","century","genre_coarse","genre_full_report"}
    num_cols = [c for c in ca_tab.columns if c not in non_num]

    X = (ca_tab[num_cols]
         .apply(pd.to_numeric, errors="coerce")   # Forced transfer value
         .fillna(0)                                # Smooth it out again
         .astype(float))                           # Prince 'likes' float

    # Labels: century/epoch + genre_coarse (if any)
    label_cols=[]
    if "century" in ca_tab.columns and ca_tab["century"].notna().any(): label_cols.append("century")
    elif "epoch" in ca_tab.columns: label_cols.append("epoch")
    if "genre_coarse" in ca_tab.columns: label_cols.append("genre_coarse")
    row_labels = (ca_tab[label_cols].astype(str).agg("_".join, axis=1)
                  if label_cols else pd.Series(range(len(ca_tab)), dtype=str))

    # Remove all rows/columns with a value of 0
    row_mask = X.sum(axis=1) > 0; col_mask = X.sum(axis=0) > 0
    X = X.loc[row_mask, col_mask]; row_labels = row_labels.loc[row_mask]
    if X.shape[0]==0 or X.shape[1]==0:
        print("[warn] CA 输入全为 0"); return

    n_comp = max(1, min(2, min(X.shape[0]-1, X.shape[1]-1)))
    ca = prince.CA(n_components=n_comp, random_state=42).fit(X)
    rows = ca.row_coordinates(X); rows.index = row_labels.values
    cols = ca.column_coordinates(X); cols.index = X.columns  # column name = bucket name

    # Only mark the first K rows
    freq = X.sum(axis=1).to_numpy()
    keep = np.argsort(-freq)[:min(int(annotate_top), len(freq))]
    top_mask = np.zeros(len(freq), bool); top_mask[keep] = True

    # --- Drawing ---
    if n_comp >= 2:
        fig, ax = plt.subplots()
        _scatter_with_labels(ax,
            x=rows[0].to_numpy(), y=rows[1].to_numpy(),
            labels=rows.index.to_numpy(), top_mask=top_mask
        )
        ax.set_title(f"{title_prefix} rows"); ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
        plt.tight_layout(); savefig_here(f"{fname_prefix}_rows.png"); plt.show()

        fig, ax = plt.subplots()
        _scatter_with_labels(ax,
            x=cols[0].to_numpy(), y=cols[1].to_numpy(),
            labels=cols.index.to_numpy(), top_mask=np.ones(len(cols), dtype=bool)
        )
        ax.set_title(f"{title_prefix} columns"); ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
        plt.tight_layout(); savefig_here(f"{fname_prefix}_cols.png"); plt.show()
    else:
        # 1D: Add minimal jitter to the y-axis while limiting the main vertical text avoidance
        jitter = (np.random.rand(len(rows))*0.06 - 0.03)
        fig, ax = plt.subplots()
        _scatter_with_labels(ax,
            x=rows[0].to_numpy(), y=jitter,
            labels=rows.index.to_numpy(), top_mask=top_mask
        )
        ax.set_title(f"{title_prefix} rows (1D)"); ax.set_xlabel("Dim1"); ax.set_yticks([])
        plt.tight_layout(); savefig_here(f"{fname_prefix}_rows_1D.png"); plt.show()

        fig, ax = plt.subplots()
        _scatter_with_labels(ax,
            x=cols[0].to_numpy(), y=np.zeros(len(cols)),
            labels=cols.index.to_numpy(), top_mask=np.ones(len(cols), dtype=bool)
        )
        ax.set_title(f"{title_prefix} columns (1D)"); ax.set_xlabel("Dim1"); ax.set_yticks([])
        plt.tight_layout(); savefig_here(f"{fname_prefix}_cols_1D.png"); plt.show()

# —— CA graphs of l1 ——
run_ca_and_plot_generic(ca_L1, title_prefix="CA (Layer1: century×genre × buckets)",
                        annotate_top=30, fname_prefix="CA_L1")

# ==== Layer 1: Interactive CA (one MC / one EMoC), and save HTML + PNG. fallback ====
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px
import plotly.io as pio
from prince import CA

FIGDIR = Path(OUTDIR) / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

# —— Tool: Fit CA from the ca table in L1 and return rows/cols coordinates (automatically recognizes numeric columns = BUCKETS) ——
def _fit_rows_cols_from_ca_tab_generic(ca_tab: pd.DataFrame):
    # Numerical column = All values ​​after removing the grouping/reporting columns (i.e., each bucket)
    non_num = {"epoch","century","genre_coarse","genre_full_report"}
    num_cols = [c for c in ca_tab.columns if c not in non_num]
    X = (ca_tab[num_cols]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0)
         .astype(float))

    # Labels: prioritize century, then epoch if none; add genre_coarse (if applicable).
    label_cols = []
    if "century" in ca_tab.columns and ca_tab["century"].notna().any():
        label_cols.append("century")
    elif "epoch" in ca_tab.columns:
        label_cols.append("epoch")
    if "genre_coarse" in ca_tab.columns:
        label_cols.append("genre_coarse")

    row_labels = (ca_tab[label_cols].astype(str).agg("_".join, axis=1)
                  if label_cols else pd.Series(range(len(ca_tab)), dtype=str))

    row_mask = X.sum(axis=1) > 0
    col_mask = X.sum(axis=0) > 0
    X = X.loc[row_mask, col_mask]
    row_labels = row_labels.loc[row_mask]
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("CA 输入全为 0（Layer1×buckets）。")

    n_comp = max(1, min(2, min(X.shape[0]-1, X.shape[1]-1)))
    ca = CA(n_components=n_comp, random_state=42).fit(X)

    rows = ca.row_coordinates(X); rows.index = row_labels.values
    cols = ca.column_coordinates(X); cols.index = X.columns  # 列轴=各 bucket 名
    return rows, cols, X

# —— interactive scatters ——
def ca_interactive_scatter_generic(rows, cols, title_prefix="CA (L1×buckets)"):
    has_dim2_rows = 1 in rows.columns
    has_dim2_cols = 1 in cols.columns
    rows_df = pd.DataFrame({
        "x": rows[0].to_numpy(),
        "y": rows[1].to_numpy() if has_dim2_rows else np.zeros(len(rows)),
        "label": rows.index.astype(str),
        "type": "row",
    })
    cols_df = pd.DataFrame({
        "x": cols[0].to_numpy(),
        "y": cols[1].to_numpy() if has_dim2_cols else np.zeros(len(cols)),
        "label": cols.index.astype(str),
        "type": "column",
    })
    all_df = pd.concat([rows_df, cols_df], ignore_index=True)

    fig = px.scatter(
        all_df, x="x", y="y",
        color="type", symbol="type",
        hover_name="label",
        title=title_prefix,
    )
    fig.update_traces(selector=dict(name="column"), marker=dict(size=10))
    fig.update_traces(selector=dict(name="row"),    marker=dict(size=6, opacity=0.7))
    fig.update_xaxes(title_text="Dim1")
    fig.update_yaxes(title_text=("Dim2" if (has_dim2_rows or has_dim2_cols) else ""))
    return fig

# —— Matplotlib rollback PNG (without Kaleido) ——
def save_png_fallback(rows, cols, png_path, title, top_k=30):
    rx = rows[0].to_numpy()
    ry = rows[1].to_numpy() if (1 in rows.columns) else np.zeros(len(rows))
    cx = cols[0].to_numpy()
    cy = cols[1].to_numpy() if (1 in cols.columns) else np.zeros(len(cols))

    plt.figure(figsize=(8.5, 6.0), dpi=150)
    plt.scatter(rx, ry, s=16, alpha=0.55, label="row")
    plt.scatter(cx, cy, s=36, alpha=0.9, marker="^", label="column")

    # Only the top_k labels with "large amplitude" in the row points are given
    approx = np.abs(rx) + np.abs(ry)
    keep = np.argsort(-approx)[: min(top_k, len(rx))]
    for i in keep:
        plt.annotate(str(rows.index[i]), (rx[i], ry[i]), fontsize=9,
                     xytext=(0, 6), textcoords="offset points", ha="center")

    plt.title(title); plt.xlabel("Dim1");
    if (1 in rows.columns) or (1 in cols.columns): plt.ylabel("Dim2")
    else: plt.gca().set_yticks([])
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

# — For each of MC/EMoC: first filter the epochs in l1, then construct the table, and then draw —
for name, sub in [("MC", l1[l1["epoch"]=="MC"]), ("EMoC", l1[l1["epoch"]=="EMoC"])]:
    ca_l1_epoch = make_ca_table_L1(sub, use_century=True)  # Still in the century × (genre) × bucket
    rows, cols, X = _fit_rows_cols_from_ca_tab_generic(ca_l1_epoch)

    html_path = FIGDIR / f"CA_L1_{name}.html"
    png_path  = FIGDIR / f"CA_L1_{name}.png"

    fig = ca_interactive_scatter_generic(rows, cols, title_prefix=f"CA (Layer1 {name}: century×genre×buckets)")
    pio.write_html(fig, str(html_path))
    print(f"[ok] L1 HTML -> {html_path}")

    save_png_fallback(rows, cols, png_path, title=f"CA (Layer1 {name}: century×genre×buckets)", top_k=30)
    print(f"[ok] L1 PNG  -> {png_path}")

    # Interaction can be displayed directly
    fig.show()

# ==== Layer1 dispersion（period ×[genre] × l1_bucket）====
FIGDIR = OUTDIR / "figs"; FIGDIR.mkdir(parents=True, exist_ok=True)
def savefig_here(name): plt.savefig(FIGDIR / name, dpi=180, bbox_inches="tight")

# group keys
group_keys_base = [GROUP_COL] + (["genre_coarse"] if (("genre_coarse" in l1.columns) and l1["genre_coarse"].notna().any()) else [])
group_keys = group_keys_base + ["l1_bucket"]

# The "total number of texts" in the bucket (regardless of the bucket itself) is used to calculate coverage.
parent_totals = (l1.groupby(group_keys_base, dropna=False)["file"]
                   .nunique()
                   .rename("parent_total_texts"))

disp_rows = []
for keys, sub in l1.groupby(group_keys, dropna=False):
    # key (period × [genre])
    parent_key = keys[:-1] if isinstance(keys, tuple) else [keys][:-1]
    if not isinstance(keys, tuple):  # When using a single key, groupby will provide a scalar value.
        parent_key = (keys,)

    # Total number of texts under this period × [genre] (bucket-insensitive)
    total_texts = parent_totals.loc[parent_key] if len(group_keys_base) else l1["file"].nunique()

    tokens  = len(sub)                              # Number of tokens in this bucket
    n_texts = sub["file"].nunique() if tokens>0 else 0  # Number of texts that have triggered this bucket

    if tokens > 0:
        by_file = sub.groupby("file").size()
        top_text_share = float(by_file.max() / by_file.sum())
        p = (by_file / by_file.sum()).to_numpy(float)
        H = float((p * p).sum())      # Herfindahl (the larger the number, the more concentrated it becomes)
        U = float(1.0 - H)            # Uniformity
        N = int(n_texts)
        if N > 0:
            uniform = 1.0 / N
            dp = float(1.0 - (np.abs(p - uniform).sum() / (2.0 - 2.0/N)))  # Gries DP (0–1, the larger the value, the more uniform the distribution)
        else:
            dp = float("nan")
    else:
        top_text_share = float("nan"); H = float("nan"); U = float("nan"); dp = float("nan")

    row = {
        GROUP_COL: keys[0] if len(group_keys)==2 else keys[0],   # Compatible with/without genre
        "l1_bucket": keys[-1],
        "tokens": int(tokens),
        "n_texts": int(n_texts),
        "top_text_share": top_text_share,
        "doc_coverage": float(n_texts / total_texts) if total_texts>0 else float("nan"),
        "H_concentration": H,
        "U_evenness": U,
        "DP_Gries": dp,
    }
    if len(group_keys_base)==2:
        row["genre_coarse"] = keys[1]
    disp_rows.append(row)

disp_L1 = (pd.DataFrame(disp_rows)
             .astype({"tokens":"Int64","n_texts":"Int64"})
             .sort_values(group_keys_base + ["l1_bucket"]))

# store and print
disp_L1.to_csv(OUTDIR / "zai_L1_dispersion_report.csv", index=False)
print("[L1-dispersion] saved:", OUTDIR / "zai_L1_dispersion_report.csv")

# ===== Optional: Two lightweight visualizations (for quick and easy visualization) =====
# 1) DP Heatmap (by period × bucket; if there is a genre, aggregate and take the average first)
heat = (disp_L1.groupby([GROUP_COL, "l1_bucket"], dropna=False)["DP_Gries"]
               .mean().unstack("l1_bucket"))
plt.figure(figsize=(max(6, 1.2*len(heat.columns)), 0.9*len(heat.index)))
plt.imshow(heat.to_numpy(), aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
plt.yticks(range(len(heat.index)), heat.index)
plt.colorbar(label="DP (Gries, higher=more even)")
plt.title("Layer1 dispersion (DP) by period × bucket")
plt.tight_layout(); savefig_here("L1_dispersion_DP_heatmap.png"); plt.show()

# 2) Top-level concentration bar chart: Select the bucket with the highest top_text_share for each period (you can see whether it was boosted by individual texts).
peak = (disp_L1.sort_values(["top_text_share"], ascending=False)
               .groupby(GROUP_COL, as_index=False).first())
plt.figure(figsize=(8, max(3, 0.5*len(peak))))
plt.barh(peak[GROUP_COL].astype(str), peak["top_text_share"])
for y, (g, b, v) in enumerate(zip(peak[GROUP_COL].astype(str), peak["l1_bucket"], peak["top_text_share"])):
    plt.text(v+0.01, y, f"{b}  {v:.2f}", va="center", fontsize=9)
plt.xlim(0, 1); plt.xlabel("top_text_share (max share within bucket)")
plt.title("Layer1: max concentration per period (bucket label at right)")
plt.tight_layout(); savefig_here("L1_dispersion_topshare_by_period.png"); plt.show()