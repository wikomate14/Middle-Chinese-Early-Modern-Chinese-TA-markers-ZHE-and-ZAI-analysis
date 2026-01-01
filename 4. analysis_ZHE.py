# -*- coding: utf-8 -*-
"""Analysis (1).ipynb

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

# ---- I/O basic path ----
BASE   = Path("/data/dataset/layered data")
OUTDIR = Path("/data/analysis_out/zhe"); OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR = OUTDIR / "figs"; FIGDIR.mkdir(parents=True, exist_ok=True)

def _sanitize(name: str) -> str:
    return (str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")
                    .replace("：", "_").replace(":", "_"))

def savefig_here(filename: str, dpi=200):
    import matplotlib.pyplot as plt
    fname = _sanitize(filename)
    plt.savefig(FIGDIR / fname, dpi=dpi, bbox_inches="tight")
    print("[saved]", FIGDIR / fname)

# ---- Collection guideline (Book title - Century - Genre) ----
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
        f = BASE / f"{epoch_prefix}_zhe_{L}.txt"
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

# ---- Align file names (ignore anything after the backslash "/"; remove extensions and spaces) ----
df_all["file_base"] = (df_all["file"].astype(str)
                       .str.replace(r"\.(xls[x]?|csv|txt)$","", regex=True))
df_all["file_key"]  = (df_all["file_base"].str.split("/").str[0]
                       .str.replace(r"\s+","", regex=True))

m = df_all.merge(
    guides[["epoch","title_key","century","genre_full","genre_coarse"]],
    left_on=["epoch","file_key"], right_on=["epoch","title_key"],
    how="left"
).drop(columns=["title_key"])

unmatched = (m[m["genre_full"].isna()][["epoch","file","file_base","file_key"]]
             .drop_duplicates().sort_values(["epoch","file"]))
unmatched.to_csv(OUTDIR/"unmatched_files_for_genre_mapping.csv", index=False)
print(f"[merge] 匹配成功 {len(m)-len(unmatched)} 条，未匹配 {len(unmatched)} 条")
print("[merge] 未匹配清单：", OUTDIR/"unmatched_files_for_genre_mapping.csv")

df_all = m  # cover back
df_all.to_csv(OUTDIR/"zhe_all_with_genre_raw.csv", index=False)

# ---- Function Mapping: Sᶻʰᵉ = {Zh-LexV, Zh-P, Zh-Dur?, Zh-Dur} ----
# 1) Prefer using the label/condition from the layer export (if it exists, your ZHE extraction script will provide it).
# 2) If there is no label, use the condition + context for conservative fallback.
def map_zhe_state(row) -> str:
    lab = (row.get("label","") or "").strip()
    cond = (row.get("condition","") or "").strip()
    ctx  = (row.get("context","") or "")
    zhe  = (row.get("zhe","") or "")
    nxt  = (row.get("next","") or "")
    full = f"{ctx} {zhe} {nxt}"

    # --- Prioritize options with labels (directly reuse my layer extraction results) ---
    if lab in {"T1","U1"}:
        if "T1: 著(Di/T)" in cond or cond.startswith("U1: 汉字+著"):
            return "Zh-Dur"       # strong durative markers
        if re.search(r"著\(V", cond):  # explicit verbs "着"
            return "Zh-LexV"
        if "汉字+著+汉字(1)" in cond:
            return "Zh-Dur?"      # optional weak cases
        return "Zh-LexV"

    if lab in {"T2","U2"}:
        return "Zh-P"              # Orientation/Location Structure

    if lab in {"T3","U3"}:
        return "Zh-LexV"           # New L3 returns to LexV

    if lab in {"T4","U4"}:
        return "Zh-LexV"           # 所+着 → verb (default)

    # --- A conservative fallback without a label (only a fallback, it won't override your explicit label) ---
    # L2 Common Location/Site Signs
    if re.search(r"著\(V", zhe) and re.search(r"[上下内中前後东西南北於于在間]\(", full):
        return "Zh-P"
    # L1 contains (Di/T) → aspect
    if re.search(r"著\((Di|T)\)", full):
        return "Zh-Dur"
    # 所+著（at any layer）→ LexV
    if re.search(r"所[\u4e00-\u9fff]*著", f"{ctx}{zhe}"):
        return "Zh-LexV"
    # Default return LexV
    return "Zh-LexV"

df_all["state"] = df_all.apply(map_zhe_state, axis=1)
STATES = ["Zh-LexV","Zh-P","Zh-Dur?","Zh-Dur"]
df_all["state"] = pd.Categorical(df_all["state"], categories=STATES, ordered=True)

# ---- Preview & Persistence ----
preview_cols = ["epoch","century","layer","file","context","zhe","next","condition","state"]
df_all.to_csv(OUTDIR / "zhe_all_epochs_long.csv", index=False)
df_all[preview_cols].head(30).to_csv(OUTDIR / "zhe_all_preview_head30.csv", index=False)
print("[dump] written:", OUTDIR / "zhe_all_epochs_long.csv")

# ---- Grouping Dimension ----
def _century_sort_key(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else 10**9

GROUP_COL = "century" if (df_all["century"].notna().any()) else "epoch"
GROUPS = sorted(df_all[GROUP_COL].dropna().astype(str).unique(), key=_century_sort_key)
print("[group] GROUP_COL =", GROUP_COL, " | #groups =", len(GROUPS))

# ==== ZHE: CI (Wilson + Jeffreys) ====
from math import sqrt
try:
    from scipy.stats import beta
except Exception:
    beta = None

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
for gval, g in df_all.groupby(GROUP_COL, dropna=False, observed=False):
    n=len(g)
    vc=g["state"].value_counts()
    for s in STATES:
        k=int(vc.get(s,0))
        wL,wC,wU=wilson_ci(k,n)
        jL,jU=jeffreys_ci(k,n)
        rows.append({GROUP_COL:gval,"state":s,"n":n,"k":k,"p":k/n if n else 0.0,
                     "W_lower":wL,"W_center":wC,"W_upper":wU,"J_lower":jL,"J_upper":jU})
prop = (pd.DataFrame(rows)
          .sort_values([GROUP_COL,"state"])
          .reset_index(drop=True))

# Persistence
prop.to_csv(OUTDIR/"zhe_proportions_by_group.csv", index=False)

# — Save each state as a line graph + Wilson interval to FIGDIR —
for st in STATES:
    d = prop[prop["state"]==st].set_index(GROUP_COL).reindex(GROUPS)
    x = np.arange(len(GROUPS)); y=d["p"].values; yL=d["W_lower"].values; yU=d["W_upper"].values
    plt.figure()
    plt.plot(x,y,marker="o")
    plt.fill_between(x,yL,yU,alpha=.2)
    plt.title(f"{st} over {GROUP_COL}")
    plt.xticks(x, GROUPS, rotation=45)
    plt.ylabel("Proportion")
    plt.tight_layout()
    savefig_here(f"ZHE_ci_{st}_{GROUP_COL}.png")   # ← save PNG
    plt.show()

print("[ok] CI done ->", OUTDIR/"zhe_proportions_by_group.csv")

# ==== ZHE TM (only MC → EMoC) ====

# ZHE's set of states
STATES = ["Zh-LexV","Zh-P","Zh-Dur?","Zh-Dur"]

# --- Take the state share vectors p(MC), p(EMoC) at both ends of MC / EMoC ---
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

_plot_T_bars(T, title="T: MC → EMoC", fname="ZHE_TM_MC_to_EMoC_stackedbar.png")

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

_plot_T_arrows(T, title="T: MC → EMoC", fname="ZHE_TM_MC_to_EMoC_arrows.png", thr=0.05)

# — Optional: Save the T value as well as a long table (for easy reuse).
edges = []
for i, s_from in enumerate(STATES):
    for j, s_to in enumerate(STATES):
        edges.append({"from":"MC","to":"EMoC","from_state":s_from,"to_state":s_to,"prob":float(T[i,j])})
pd.DataFrame(edges).to_csv(OUTDIR/"zhe_TM_MC_to_EMoC.csv", index=False)
print("[TM] written:", OUTDIR/"zhe_TM_MC_to_EMoC.csv")

# ==== ZHE | CA (Standard Model Diagram + Interaction Diagram) =====

import prince

from adjustText import adjust_text

# —— Construct CA Input Table (Period × Genre × State Count) ——
HAVE_COARSE = ("genre_coarse" in df_all.columns) and df_all["genre_coarse"].notna().any()
if HAVE_COARSE:
    ca_tab = (df_all.groupby([GROUP_COL,"genre_coarse","state"], dropna=False)
                     .size().unstack("state").fillna(0).astype(float).reset_index())
    # Summary of full names of genres (for report purposes)
    gf = (df_all.dropna(subset=["genre_coarse"])
                .groupby([GROUP_COL,"genre_coarse"])["genre_full"]
                .apply(lambda s: "|".join(sorted(set(map(str,s))))))
    ca_tab = ca_tab.merge(gf.rename("genre_full_report").reset_index(),
                          on=[GROUP_COL,"genre_coarse"], how="left")
else:
    ca_tab = (df_all.groupby([GROUP_COL,"state"], dropna=False)
                     .size().unstack("state").fillna(0).astype(float).reset_index())

# — Standard model diagram —

def savefig_here(name, dpi=200):
    p = FIGDIR / name
    plt.savefig(p, dpi=dpi, bbox_inches="tight"); print("[saved]", p)

# ========= General: Scattered Points + Avoidance & CA Map Generation =========
import matplotlib.patheffects as pe

def _scatter_with_labels(ax, x, y, labels, top_mask=None,
                         point_bg=14, point_fg=22, fontsize=9):
    x = np.asarray(x); y = np.asarray(y); labels = np.asarray(labels)
    if top_mask is None: top_mask = np.ones_like(x, dtype=bool)
    ax.scatter(x[~top_mask], y[~top_mask], s=point_bg, alpha=0.35)
    ax.scatter(x[top_mask], y[top_mask], s=point_fg, alpha=0.85)
    texts=[]
    for xi, yi, lab in zip(x[top_mask], y[top_mask], labels[top_mask]):
        texts.append(ax.text(xi, yi, lab, fontsize=fontsize, ha="center", va="center",
                             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]))
    if texts:
        adjust_text(texts, only_move={'points':'xy','text':'xy'},
                    force_points=0.6, force_text=0.6,
                    expand_points=(1.6,1.6), expand_text=(1.6,1.6),
                    arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.4, shrinkA=6, shrinkB=6))

def make_ca_table_generic(df: pd.DataFrame, states, use_century=True) -> pd.DataFrame:
    """Counted by (century/epoch) × genre_coarse (optional) × state, listed as states."""
    group_col = "century" if (use_century and "century" in df.columns and df["century"].notna().any()) else "epoch"
    have_coarse = ("genre_coarse" in df.columns) and df["genre_coarse"].notna().any()
    if have_coarse:
        tab = (df.groupby([group_col,"genre_coarse","state"], dropna=False, observed=False)
                 .size().unstack("state").reindex(columns=states, fill_value=0).reset_index())
    else:
        tab = (df.groupby([group_col,"state"], dropna=False, observed=False)
                 .size().unstack("state").reindex(columns=states, fill_value=0).reset_index())
    return tab

def run_ca_and_plot_generic(ca_tab: pd.DataFrame, states,
                            title_prefix="CA", annotate_top=30, fname_prefix="CA"):
    # Numeric matrix
    X = ca_tab.reindex(columns=states, fill_value=0).astype(float)
    # Row tags: century/epoch + genre_coarse (if any)
    label_cols=[]
    if "century" in ca_tab.columns and ca_tab["century"].notna().any(): label_cols.append("century")
    elif "epoch" in ca_tab.columns: label_cols.append("epoch")
    if "genre_coarse" in ca_tab.columns: label_cols.append("genre_coarse")
    row_labels = (ca_tab[label_cols].astype(str).agg("_".join, axis=1)
                  if label_cols else pd.Series(range(len(ca_tab)), dtype=str))

    # Remove all rows/columns with 0
    row_mask = X.sum(axis=1)>0; col_mask = X.sum(axis=0)>0
    X = X.loc[row_mask, col_mask]; row_labels = row_labels.loc[row_mask]
    if X.shape[0]==0 or X.shape[1]==0:
        print("[CA] empty input."); return

    n_comp = max(1, min(2, min(X.shape[0]-1, X.shape[1]-1)))
    ca = prince.CA(n_components=n_comp, random_state=42).fit(X)
    rows = ca.row_coordinates(X); rows.index = row_labels.values
    cols = ca.column_coordinates(X); cols.index = X.columns

    # with the highest frequency (K)
    freq = X.sum(axis=1).to_numpy()
    K = min(int(annotate_top), len(freq))
    keep = np.argsort(-freq)[:K]
    top_mask = np.zeros(len(freq), dtype=bool); top_mask[keep]=True

    # Micro-clusters are dispersed to avoid overlap
    def _defuse(xy, radius=0.015):
        xy = np.asarray(xy); out = xy.copy()
        key = np.round(xy, 6)
        uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        for i, c in enumerate(cnt):
            if c<=1: continue
            idx = np.where(inv==i)[0]
            ang = np.linspace(0, 2*np.pi, c, endpoint=False)
            out[idx,0] += radius*np.cos(ang); out[idx,1] += radius*np.sin(ang)
        return out

    if n_comp >= 2:
        rxy = _defuse(np.c_[rows[0].to_numpy(), rows[1].to_numpy()])
        cxy = _defuse(np.c_[cols[0].to_numpy(), cols[1].to_numpy()])

        fig, ax = plt.subplots()
        _scatter_with_labels(ax, rxy[:,0], rxy[:,1], rows.index.to_numpy(), top_mask=top_mask)
        ax.set_title(f"{title_prefix} rows"); ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
        plt.tight_layout(); savefig_here(f"{fname_prefix}_rows.png"); plt.show()

        fig, ax = plt.subplots()
        _scatter_with_labels(ax, cxy[:,0], cxy[:,1], cols.index.to_numpy(),
                             top_mask=np.ones(len(cols), dtype=bool))
        ax.set_title(f"{title_prefix} columns (states)"); ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
        plt.tight_layout(); savefig_here(f"{fname_prefix}_cols.png"); plt.show()
    else:
        jitter = (np.random.rand(len(rows))*0.06 - 0.03)
        fig, ax = plt.subplots()
        _scatter_with_labels(ax, rows[0].to_numpy(), jitter, rows.index.to_numpy(), top_mask=top_mask)
        ax.set_title(f"{title_prefix} rows (1D)"); ax.set_xlabel("Dim1"); ax.set_yticks([])
        plt.tight_layout(); savefig_here(f"{fname_prefix}_rows_1D.png"); plt.show()

        fig, ax = plt.subplots()
        _scatter_with_labels(ax, cols[0].to_numpy(), np.zeros(len(cols)), cols.index.to_numpy(),
                             top_mask=np.ones(len(cols), dtype=bool))
        ax.set_title(f"{title_prefix} columns (1D)"); ax.set_xlabel("Dim1"); ax.set_yticks([])
        plt.tight_layout(); savefig_here(f"{fname_prefix}_cols_1D.png"); plt.show()

# ZHE's set of states
STATES_ZHE = ["Zh-LexV","Zh-P","Zh-Dur?","Zh-Dur"]

df_all_zhe_mc   = df_all[df_all["epoch"]=="MC"].copy()
df_all_zhe_emoc = df_all[df_all["epoch"]=="EMoC"].copy()

ca_zhe_mc   = make_ca_table_generic(df_all_zhe_mc,   STATES_ZHE, use_century=True)
ca_zhe_emoc = make_ca_table_generic(df_all_zhe_emoc, STATES_ZHE, use_century=True)

run_ca_and_plot_generic(ca_zhe_mc,   STATES_ZHE,
                        title_prefix="CA (ZHE: MC century×genre)",
                        annotate_top=30, fname_prefix="CA_ZHE_MC")

run_ca_and_plot_generic(ca_zhe_emoc, STATES_ZHE,
                        title_prefix="CA (ZHE: EMoC century×genre)",
                        annotate_top=30, fname_prefix="CA_ZHE_EMoC")

# —— Interactive Images: Two (MC / EMoC), saved to disk HTML+PNG Back ——
import plotly.express as px, plotly.io as pio

def _fit_rows_cols(ca_tab_src: pd.DataFrame):
    X = ca_tab_src.reindex(columns=STATES, fill_value=0).astype(float)
    label_cols=[]
    if "century" in ca_tab_src.columns and ca_tab_src["century"].notna().any(): label_cols.append("century")
    elif "epoch" in ca_tab_src.columns: label_cols.append("epoch")
    if "genre_coarse" in ca_tab_src.columns: label_cols.append("genre_coarse")
    row_labels = (ca_tab_src[label_cols].astype(str).agg("_".join, axis=1)
                  if label_cols else pd.Series(range(len(ca_tab_src)), dtype=str))
    row_mask = X.sum(axis=1)>0; col_mask = X.sum(axis=0)>0
    X = X.loc[row_mask, col_mask]; row_labels = row_labels.loc[row_mask]
    n_comp = max(1, min(2, min(X.shape[0]-1, X.shape[1]-1)))
    ca = prince.CA(n_components=n_comp, random_state=42).fit(X)
    rows = ca.row_coordinates(X); rows.index = row_labels.values
    cols = ca.column_coordinates(X);
    return rows, cols

def ca_interactive(rows, cols, title):
    has_dim2_rows = 1 in rows.columns
    has_dim2_cols = 1 in cols.columns
    rows_df = pd.DataFrame({"x":rows[0].to_numpy(),
                            "y":rows[1].to_numpy() if has_dim2_rows else np.zeros(len(rows)),
                            "label":rows.index.astype(str), "type":"row"})
    cols_df = pd.DataFrame({"x":cols[0].to_numpy(),
                            "y":cols[1].to_numpy() if has_dim2_cols else np.zeros(len(cols)),
                            "label":cols.index.astype(str), "type":"column"})
    all_df = pd.concat([rows_df, cols_df], ignore_index=True)
    fig = px.scatter(all_df, x="x", y="y", color="type", symbol="type", hover_name="label", title=title)
    fig.update_traces(selector=dict(name="column"), marker=dict(size=10))
    fig.update_traces(selector=dict(name="row"),    marker=dict(size=6, opacity=0.75))
    fig.update_xaxes(title_text="Dim1"); fig.update_yaxes(title_text=("Dim2" if (has_dim2_rows or has_dim2_cols) else ""))
    return fig

# Disassembling the two interactive diagrams of MC / EMoC
df_mc   = df_all[df_all["epoch"]=="MC"].copy()
df_emoc = df_all[df_all["epoch"]=="EMoC"].copy()

def _make_ca_tab(df_src):
    if HAVE_COARSE:
        t = (df_src.groupby([GROUP_COL,"genre_coarse","state"], dropna=False)
                    .size().unstack("state").fillna(0).astype(float).reset_index())
    else:
        t = (df_src.groupby([GROUP_COL,"state"], dropna=False)
                    .size().unstack("state").fillna(0).astype(float).reset_index())
    return t

ca_mc   = _make_ca_tab(df_mc)
ca_emoc = _make_ca_tab(df_emoc)

for name, ca_src in [("MC", ca_mc), ("EMoC", ca_emoc)]:
    rows, cols = _fit_rows_cols(ca_src)
    fig = ca_interactive(rows, cols, title=f"CA (ZHE · {name})")
    html_path = FIGDIR / f"CA_ZHE_{name}.html"
    pio.write_html(fig, str(html_path)); print("[saved]", html_path)
    # Back up a static PNG image (using plotly's to_image requires kaleido; skip this step if kaleido is not available in your environment)
    try:
        import kaleido  # noqa
        png_path = FIGDIR / f"CA_ZHE_{name}.png"
        fig.write_image(str(png_path), scale=2); print("[saved]", png_path)
    except Exception:
        pass
    fig.show()

# ==== ZHE: dispersion (period × [genre] × state) ====

# ZHE's set of states
STATES = ["Zh-LexV","Zh-P","Zh-Dur?","Zh-Dur"]

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
disp_zhe.to_csv(OUTDIR / "zhe_dispersion_report.csv", index=False)
print("[ZHE-dispersion] saved:", OUTDIR / "zhe_dispersion_report.csv")

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
plt.title("ZHE dispersion (DP) by period × state")
plt.tight_layout(); savefig_here("ZHE_dispersion_DP_heat.png"); plt.show()

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
plt.title("ZHE top_text_share by period × state")
plt.tight_layout(); savefig_here("ZHE_dispersion_topshare_heat.png"); plt.show()