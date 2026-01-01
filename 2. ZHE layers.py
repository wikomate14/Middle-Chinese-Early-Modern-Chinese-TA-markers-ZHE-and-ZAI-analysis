# -*- coding: utf-8 -*-
"""ZHE.ipynb

"""

# The following lines are only used on Google Colab and are commented out by default in this repository version.
#!fusermount -u /content/drive 2>/dev/null || true
#!rm -rf /content/drive
#!mkdir -p /content/drive

#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

# =========================================
# Shared header (imports, config, paths, utils)
# =========================================
import re
import pandas as pd
from pathlib import Path

class Config:
    ID_COL      = 1
    TARGET_COL  = 4
    CONTEXT_COL = 3
    NEXT_COL    = 5

    @staticmethod
    def ordinal_suffix(n: int) -> str:
        if 10 <= n % 100 <= 20:
            return f"{n}th"
        return f"{n}{ {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th') }"

BASE_DIR     = Path("/data/dataset/groups by century")
TAGGED_DIR   = BASE_DIR / "Tagged_zhe"
UNTAGGED_DIR = BASE_DIR / "Untagged_zhe"
CENTURIES    = [f"{Config.ordinal_suffix(i)} century" for i in range(3, 10)]

OUT_DIR      = Path("/data/dataset/layered data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_rows():
    """Scan both Tagged/Untagged data by century; return unified dataframe."""
    records = []
    for is_tagged, root in [(True, TAGGED_DIR), (False, UNTAGGED_DIR)]:
        for label in CENTURIES:
            cdir = root / label
            if not cdir.exists():
                continue
            for fpath in cdir.glob("*.xlsx"):
                try:
                    df = pd.read_excel(fpath, header=None, engine="openpyxl")
                except Exception as e:
                    print(f"[warn] read fail: {fpath} -> {e}")
                    continue
                for _, row in df.iterrows():
                    records.append({
                        "file":      fpath.stem,
                        "ID":        str(row.iloc[Config.ID_COL-1]).strip(),
                        "context":   str(row.iloc[Config.CONTEXT_COL-1]),
                        "zhe":       str(row.iloc[Config.TARGET_COL-1]),
                        "next":      str(row.iloc[Config.NEXT_COL-1]),
                        "is_tagged": is_tagged,
                        "source_dir":"Tagged" if is_tagged else "Untagged",
                    })
    cols = ["file","ID","context","zhe","next","is_tagged","source_dir"]
    df = pd.DataFrame.from_records(records, columns=cols)
    for c in ["file","ID","context","zhe","next"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def safe_read_xlsx(path: Path, columns=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
    return pd.read_excel(path, dtype=str, engine="openpyxl")

# =========================================================
# ---------- Part 1: old L3 & old L1  →  New L1 & New L2
# =========================================================
# Mapping in this part:
# - old L3 (T3/U3) becomes New Layer1 (T1/U1)  → outputs: zhe_L1_blend_T1_U1.*
# - old L1 (T1/U1) becomes New Layer2 (T2/U2)  → outputs: zhe_L2_blend_T2_U2.*

# ---- old L3 rules (now New Layer1; relabel T3/U3 → T1/U1)
def _oldL3_tagged(ctx: str, zhe: str, nxt: str):
    full_text = f"{ctx} {zhe} {nxt}"
    if re.search(r'(?:^|\s)著\((Di|T)\)', full_text):
        return True, 'T1: 著(Di/T)'
    patterns = [
        (r'\w+\(N[^)]*\)\s+\w+\(V[^)]*\)\s+\w+\(N[^)]*\)\s+著\([^)]*\)', 'T1: N-V-N 著(V)'),
        (r'\w+\(N[^)]*\)\s+\w+\(V[^)]*\)\s+\w+\((Dg|VCL|Ng|Nc)[^)]*\)\s+著\(V[^)]*\)', 'T1: N-V-(Dg|VCL|Ng|Nc) 著(V)'),
        (r'\w+\(V[^)]*\)\s+\w+\(V[^)]*\)\s+著\(V[^)]*\)', 'T1: V-V 著(V)'),
    ]
    for pat, label in patterns:
        if re.search(pat, full_text):
            return True, label
    return False, ''

def _oldL3_untagged(ctx: str, zhe: str, nxt: str):
    combo_prev = f"{ctx}{zhe}"
    zhenxt = f"{zhe}{nxt}"
    if re.search(r'所[\u4e00-\u9fff]*著', combo_prev):
        return False, ''
    if re.search(r'著[上下内中前後东西南北於于在間]', zhenxt):
        return False, ''
    if re.search(r'著[\u4e00-\u9fff]{2,}', zhenxt):
        return False, ''
    txt = f"{ctx}{zhe}{nxt}"
    if re.search(r'[\u4e00-\u9fff]著(?![\u4e00-\u9fff])', txt):
        return True, 'U1: 汉字+著 + 非汉字/句边'
    if re.search(r'[\u4e00-\u9fff]著[\u4e00-\u9fff](?=[^\u4e00-\u9fff]|$)', txt):
        return True, 'U1: 汉字+著+汉字(1) + 非汉字/句边'
    return False, ''

# ---- old L1 rules (now New Layer2; relabel T1/U1 → T2/U2)
def _oldL1_tagged(ctx: str, zhe: str, nxt: str):
    full_text = f"{ctx} {zhe} {nxt}"
    if re.search(r'著\(V[^)]*\)(?:\s+\w+\([^)]*\))*\s+([上下内中前後东西南北於于在間])\([^)]*\)', full_text):
        return True, 'T2: 著(V...) ... 方位词'
    if re.search(r'著\(V[^)]*\)\s+([上下内中前後东西南北於于在間])\([^)]*\)', full_text):
        return True, 'T2: 著(V...) 方位词'
    if re.search(r'著\(V[^)]*\)\s+\w+\(N[^)]*\)\w+\(Ng\)', f"{zhe} {nxt}"):
        return True, 'T2: 著(V) + N + Ng'
    if re.search(r'著\(V[^)]*\)\s+\w+\(Na\)\w+\(Na\)', f"{zhe} {nxt}"):
        return True, 'T2: 著(V) + Na + Na'
    if re.search(r'\w+\(V[^)]*\).*?著\([^)]*\).*?\w+\(N[cdg][^)]*\)', full_text):
        return True, 'T2: V ... 著(...) ... N[cdg]'
    if re.search(r'著\(V[^)]*\)\s+\w+\(N[cdg][^)]*\)', f"{zhe} {nxt}"):
        return True, 'T2: 著(V) + N[cdg]'
    if re.search(r'\w+\(V[^)]*\).*?著\([^)]*\).*?[上下内中前後东西南北於于在間]', full_text):
        return True, 'T2: V ... 著(...) ... 方位字'
    return False, ''

def _oldL1_untagged(ctx: str, zhe: str, nxt: str):
    zhenxt = f"{zhe}{nxt}"
    if re.search(r'著[上下内中前後东西南北於于在間]', zhenxt):
        return True, 'U2: 著 + 方位字'
    if re.search(r'著[\u4e00-\u9fff][上下内中前後东西南北於于在間]', zhenxt):
        return True, 'U2: 著 + 任一字 + 方位字'
    return False, ''

def main_part1():
    df_all = load_all_rows()
    if df_all.empty:
        print("[warn] no rows found.")
        return

    # ----- New Layer1 from old L3 (T1/U1) -----
    l1_rows = []
    for _, r in df_all.iterrows():
        ctx, zhe, nxt = r["context"], r["zhe"], r["next"]
        if r["is_tagged"]:
            hit, cond = _oldL3_tagged(ctx, zhe, nxt)
            if hit:
                l1_rows.append({"file": r["file"], "ID": r["ID"], "label": "T1", "condition": cond,
                                "context": ctx, "zhe": zhe, "next": nxt, "source_dir": r["source_dir"]})
        else:
            hit, cond = _oldL3_untagged(ctx, zhe, nxt)
            if hit:
                l1_rows.append({"file": r["file"], "ID": r["ID"], "label": "U1", "condition": cond,
                                "context": ctx, "zhe": zhe, "next": nxt, "source_dir": r["source_dir"]})
    df_l1 = pd.DataFrame(l1_rows, columns=["file","ID","label","condition","context","zhe","next","source_dir"])
    out_l1_xlsx = OUT_DIR / "zhe_L1_blend_T1_U1.xlsx"
    out_l1_csv  = OUT_DIR / "zhe_L1_blend_T1_U1.csv"
    if not df_l1.empty:
        df_l1.to_excel(out_l1_xlsx, index=False, engine="openpyxl")
        df_l1.to_csv(out_l1_csv, index=False, encoding="utf-8-sig")
    print(f"[L1-blend] rows: {len(df_l1)} -> {out_l1_xlsx}")

    # ----- New Layer2 from old L1 (T2/U2), prefer T2 then U2 per (file,ID) -----
    hits = []
    for _, r in df_all.iterrows():
        ctx, zhe, nxt = r["context"], r["zhe"], r["next"]
        if r["is_tagged"]:
            t2, t2c = _oldL1_tagged(ctx, zhe, nxt)
            hits.append((t2, t2c, False, ""))      # (hit_T2, cond_T2, hit_U2, cond_U2)
        else:
            u2, u2c = _oldL1_untagged(ctx, zhe, nxt)
            hits.append((False, "", u2, u2c))
    df_all[["hit_T2","cond_T2","hit_U2","cond_U2"]] = pd.DataFrame(hits, index=df_all.index)

    def choose_L2(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        g["__ord"] = range(len(g))
        t2 = g[g["hit_T2"]]
        if not t2.empty:
            pick = t2.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "T2",
                              "condition": pick["cond_T2"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"]})
        u2 = g[g["hit_U2"]]
        if not u2.empty:
            pick = u2.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "U2",
                              "condition": pick["cond_U2"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"]})
        return pd.Series({"file": g.iloc[0]["file"], "ID": g.iloc[0]["ID"],
                          "label": "", "condition": "", "context": "", "zhe": "", "next": "", "source_dir": ""})

    l2_result = (df_all.groupby(["file","ID"], as_index=False)
                        .apply(choose_L2)
                        .reset_index(drop=True))
    df_l2 = l2_result[l2_result["label"]!=""].copy()
    out_l2_xlsx = OUT_DIR / "zhe_L2_blend_T2_U2.xlsx"
    out_l2_csv  = OUT_DIR / "zhe_L2_blend_T2_U2.csv"
    if not df_l2.empty:
        df_l2.to_excel(out_l2_xlsx, index=False, engine="openpyxl")
        df_l2.to_csv(out_l2_csv, index=False, encoding="utf-8-sig")
    print(f"[L2-blend] rows: {len(df_l2)} -> {out_l2_xlsx}")

    n_keys = df_all[["file","ID"]].drop_duplicates().shape[0]
    print(f"[scan] unique (file,ID): {n_keys}")
    print(f"[done] New L1: {len(df_l1)} | New L2: {len(df_l2)}")

if __name__ == "__main__":
    main_part1()

# =====================================
# ---------- Part 2: Layer4 (unchanged)
# =====================================
L3_BLEND_XLSX = OUT_DIR / "zhe_L1_blend_T1_U1.xlsx"  # note: now New L1
L1_BLEND_XLSX = OUT_DIR / "zhe_L2_blend_T2_U2.xlsx"  # note: now New L2
L4_OUT_XLSX   = OUT_DIR / "zhe_L4_from_U4_T4.xlsx"
L4_OUT_CSV    = OUT_DIR / "zhe_L4_from_U4_T4.csv"

def _untagged_U4(ctx: str, zhe: str, nxt: str):
    if re.search(r'所[\u4e00-\u9fff]*著', f"{ctx}{zhe}"):
        return True, "U4: 所+著"
    return False, ""

def _tagged_T4(ctx: str, zhe: str, nxt: str):
    if re.search(r'所[\u4e00-\u9fff]*\s+著', f"{ctx} {zhe}"):
        return True, "T4: 所+著 (non-default)"
    return False, ""

def main_part2():
    df_all = load_all_rows()
    if df_all.empty:
        print("[warn] no rows found.")
        return

    flags = []
    for _, r in df_all.iterrows():
        ctx, zhe, nxt = r["context"], r["zhe"], r["next"]
        u4_hit, u4_cond = _untagged_U4(ctx, zhe, nxt)
        t4_hit, t4_cond = (False, "")
        if r["is_tagged"]:
            t4_hit, t4_cond = _tagged_T4(ctx, zhe, nxt)
        flags.append((u4_hit, u4_cond, t4_hit, t4_cond))
    df_all[["hit_U4","cond_U4","hit_T4","cond_T4"]] = pd.DataFrame(flags, index=df_all.index)

    def decide_L4(g: pd.DataFrame) -> pd.Series:
        g = g.copy(); g["__ord"] = range(len(g))
        u4 = g[g["hit_U4"]]
        if not u4.empty:
            pick = u4.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "U4",
                              "condition": pick["cond_U4"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"],
                              "audit_T4": "1" if g["hit_T4"].any() else "0"})
        t4 = g[g["hit_T4"]]
        if not t4.empty:
            pick = t4.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "T4",
                              "condition": pick["cond_T4"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"],
                              "audit_T4": "1"})
        return pd.Series({"file": g.iloc[0]["file"], "ID": g.iloc[0]["ID"],
                          "label": "", "condition": "", "context": "", "zhe": "", "next": "",
                          "source_dir": "", "audit_T4": "0"})

    l4_result = (df_all.groupby(["file","ID"], as_index=False, group_keys=False)
                        .apply(decide_L4)
                        .reset_index(drop=True))
    df_l4 = l4_result[l4_result["label"]!=""].copy()

    if not df_l4.empty:
        df_l4.to_excel(L4_OUT_XLSX, index=False, engine="openpyxl")
        df_l4.to_csv(L4_OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[L4] rows: {len(df_l4)} -> {L4_OUT_XLSX}")

    # Remove these keys from New L1 / New L2 outputs
    keys_l4 = set(df_l4[["file","ID"]].itertuples(index=False, name=None))

    def rewrite_without_keys(path: Path, keys: set):
        df = safe_read_xlsx(path, columns=["file","ID","label","condition","context","zhe","next","source_dir"])
        if df.empty: return
        keep = ~df.apply(lambda r: (str(r["file"]), str(r["ID"])) in keys, axis=1)
        df_new = df.loc[keep].copy()
        df_new.to_excel(path, index=False, engine="openpyxl")
        df_new.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        print(f"[rewrite] {path.name}: {len(df)} -> {len(df_new)} (removed {len(df)-len(df_new)})")

    rewrite_without_keys(L3_BLEND_XLSX, keys_l4)  # actually New L1 file
    rewrite_without_keys(L1_BLEND_XLSX, keys_l4)  # actually New L2 file

if __name__ == "__main__":
    main_part2()

# ===================================================
# ---------- Part 3: old L2  →  New Layer3
# ===================================================
# Mapping here:
# - old L2 (T2/U2) → New Layer3 (T3/U3)
#   So: relabel outputs to zhe_L3_from_U3_T3.*

L4_FILE   = OUT_DIR / "zhe_L4_from_U4_T4.xlsx"
NEW_L1    = OUT_DIR / "zhe_L1_blend_T1_U1.xlsx"  # produced in Part 1 (from old L3)
NEW_L2    = OUT_DIR / "zhe_L2_blend_T2_U2.xlsx"  # produced in Part 1 (from old L1)
L3_OUT_X  = OUT_DIR / "zhe_L3_from_U3_T3.xlsx"
L3_OUT_C  = OUT_DIR / "zhe_L3_from_U3_T3.csv"

# old L2 rules → relabel to T3/U3
def _oldL2_untagged(ctx: str, zhe: str, nxt: str):
    combo_all = f"{ctx}{zhe}{nxt}"
    zhenxt = f"{zhe}{nxt}"
    if re.search(r'[不弗勿毋未非莫没無]著(?![上下内中前後东西南北於于在間])(?![\u4e00-\u9fff][上下内中前後东西南北於于在間])', combo_all):
        return True, 'U3: Untagged Layer2 cond2'
    if re.search(r'著[\u4e00-\u9fff]{2,}', zhenxt):
        return True, 'U3: Untagged Layer2 cond1'
    if re.search(r'著[不弗勿毋未非莫没無]', zhenxt):
        return True, 'U3: Untagged Layer2 cond3'
    return False, ''

def _oldL2_tagged(ctx: str, zhe: str, nxt: str):
    if re.search(r'著\(V[^)]*\)', zhe):
        return True, 'T3: Tagged Layer2 default'
    return False, ''

def main_part3():
    df_all = load_all_rows()
    if df_all.empty:
        print("[warn] no rows found.")
        return

    # keys already produced by New L4, New L1, New L2
    produced = set()
    for f in [L4_FILE, NEW_L1, NEW_L2]:
        dfk = safe_read_xlsx(f)
        if not dfk.empty and {'file','ID'}.issubset(dfk.columns):
            produced |= set(dfk[['file','ID']].itertuples(index=False, name=None))

    df_remaining = df_all[~df_all.apply(lambda r: (r['file'], r['ID']) in produced, axis=1)].copy()

    def choose_L3(g: pd.DataFrame) -> pd.Series:
        g = g.copy(); g['__ord'] = range(len(g))
        # Prefer U3 (old untagged L2) then T3 (old tagged L2)
        u3 = []
        for i, r in g.iterrows():
            hit, cond = _oldL2_untagged(r['context'], r['zhe'], r['next'])
            if hit: u3.append((i, cond))
        if u3:
            idx, cond = sorted(u3, key=lambda x: x[0])[0]
            pick = g.loc[idx]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "U3",
                              "condition": cond, "context": pick["context"], "zhe": pick["zhe"],
                              "next": pick["next"], "source_dir": pick["source_dir"]})
        t3 = []
        for i, r in g.iterrows():
            if r['is_tagged']:
                hit, cond = _oldL2_tagged(r['context'], r['zhe'], r['next'])
                if hit: t3.append((i, cond))
        if t3:
            idx, cond = sorted(t3, key=lambda x: x[0])[0]
            pick = g.loc[idx]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "T3",
                              "condition": cond, "context": pick["context"], "zhe": pick["zhe"],
                              "next": pick["next"], "source_dir": pick["source_dir"]})
        return pd.Series({"file": g.iloc[0]["file"], "ID": g.iloc[0]["ID"], "label": "",
                          "condition": "", "context": "", "zhe": "", "next": "", "source_dir": ""})

    l3_from_remaining = (df_remaining.groupby(['file','ID'], as_index=False, group_keys=False)
                                     .apply(choose_L3)
                                     .reset_index(drop=True))
    l3_from_remaining = l3_from_remaining[l3_from_remaining['label']!=''].copy()

    # Optionally siphon U3 from New L1 (if any patterns qualify) – not required,
    # but keeping parity with your previous “backfill from L3” idea.
    # Here we *don’t* cannibalize earlier layers unless you explicitly want it.
    df_l3_all = l3_from_remaining

    if not df_l3_all.empty:
        df_l3_all["__ord"] = range(len(df_l3_all))
        df_l3_all = (df_l3_all.sort_values(["file","ID","__ord"])
                                .drop_duplicates(subset=["file","ID"], keep="first")
                                .drop(columns=["__ord"]))
        df_l3_all.to_excel(L3_OUT_X, index=False, engine="openpyxl")
        df_l3_all.to_csv(L3_OUT_C, index=False, encoding="utf-8-sig")

    n_all  = df_all[['file','ID']].drop_duplicates().shape[0]
    n_rem  = df_remaining[['file','ID']].drop_duplicates().shape[0]
    print(f"[scan] unique (file,ID): {n_all}")
    print(f"[scan] remaining after L4/L1/L2: {n_rem}")
    print(f"[L3] total written: {len(df_l3_all)} -> {L3_OUT_X}")

if __name__ == "__main__":
    main_part3()

# ==================================================
# ---------- Part 4: Remaining Export (unchanged)
# ==================================================
# This part just collects still-unassigned (file,ID) and exports a preference list.

def _read_layer_keys(path: Path):
    if path.exists():
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    elif path.with_suffix(".csv").exists():
        df = pd.read_csv(path.with_suffix(".csv"), dtype=str)
    else:
        return set()
    if not {'file','ID'}.issubset(df.columns):
        return set()
    df = df[['file','ID']].fillna("").astype(str)
    return set(df.itertuples(index=False, name=None))

def main_part4():
    layer_files = [
        OUT_DIR / "zhe_L1_blend_T1_U1.xlsx",   # New L1 (from old L3)
        OUT_DIR / "zhe_L2_blend_T2_U2.xlsx",   # New L2 (from old L1)
        OUT_DIR / "zhe_L3_from_U3_T3.xlsx",    # New L3 (from old L2)
        OUT_DIR / "zhe_L4_from_U4_T4.xlsx",    # L4
    ]

    produced_keys = set()
    for p in layer_files:
        produced_keys |= _read_layer_keys(p)

    def scan_dir(root: Path, src_label: str):
        rows = []
        for label in CENTURIES:
            cdir = root / label
            if not cdir.exists():
                continue
            for f in cdir.glob("*.xlsx"):
                try:
                    df = pd.read_excel(f, header=None, engine="openpyxl", dtype=str)
                except Exception:
                    continue
                tmp = pd.DataFrame({
                    "file": f.stem,
                    "ID":   df.iloc[:, 0].astype(str).str.strip(),
                    "context": df.iloc[:, 2].astype(str),
                    "zhe":     df.iloc[:, 3].astype(str),
                    "next":    df.iloc[:, 4].astype(str),
                    "source_dir": src_label
                })
                rows.append(tmp[["file","ID","context","zhe","next","source_dir"]])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["file","ID","context","zhe","next","source_dir"]
        )

    df_t = scan_dir(TAGGED_DIR, "Tagged")
    df_u = scan_dir(UNTAGGED_DIR, "Untagged")
    df_all = pd.concat([df_t, df_u], ignore_index=True)

    for c in ["file","ID","context","zhe","next","source_dir"]:
        df_all[c] = df_all[c].fillna("").astype(str).str.strip()

    all_keys = set(df_all[["file","ID"]].itertuples(index=False, name=None))
    remaining_keys = all_keys - produced_keys

    df_tag = (df_all[df_all["source_dir"]=="Tagged"]
              .drop_duplicates(subset=["file","ID"], keep="last"))
    df_unt = (df_all[df_all["source_dir"]=="Untagged"]
              .drop_duplicates(subset=["file","ID"], keep="last"))

    df_tag_rem = df_tag[df_tag.apply(lambda r: (r["file"], r["ID"]) in remaining_keys, axis=1)]
    picked = set(df_tag_rem[["file","ID"]].itertuples(index=False, name=None))
    df_unt_rem = df_unt[df_unt.apply(lambda r: (r["file"], r["ID"]) in (remaining_keys - picked), axis=1)]

    df_rem_pref = pd.concat([df_tag_rem, df_unt_rem], ignore_index=True)
    df_rem_pref = df_rem_pref.drop_duplicates(subset=["file","ID"], keep="first")

    out_path = OUT_DIR / "zhe_remaining_pref.xlsx"
    df_rem_pref.to_excel(out_path, index=False, engine="openpyxl")
    print(f"[done] remaining list -> {out_path} (rows: {len(df_rem_pref)})")

if __name__ == "__main__":
    main_part4()


# ==================================================
# ---------- Part 5: MC ZHE clean TXT export
# ==================================================
# Produce 6-column, tab-separated files to mirror the MC_zai_* style:
#   file   ID   context   zhe   next   condition

def _read_any_layer(path: Path) -> pd.DataFrame:
    """Read a layer table from .xlsx if present; else from .csv; else empty."""
    if path.exists():
        return pd.read_excel(path, dtype=str, engine="openpyxl").fillna("")
    csvp = path.with_suffix(".csv")
    if csvp.exists():
        return pd.read_csv(csvp, dtype=str).fillna("")
    return pd.DataFrame()

def _ensure_six_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Select and coerce the exact six columns in the required order."""
    need = ["file","ID","context","zhe","next","condition"]
    for c in need:
        if c not in df.columns:
            df[c] = ""
    return df[need].astype(str)

def main_part5_export_mc_zhe_txt():
    layer_map = {
        "layer1": OUT_DIR / "zhe_L1_blend_T1_U1.xlsx",
        "layer2": OUT_DIR / "zhe_L2_blend_T2_U2.xlsx",
        "layer3": OUT_DIR / "zhe_L3_from_U3_T3.xlsx",
        "layer4": OUT_DIR / "zhe_L4_from_U4_T4.xlsx",
    }
    for lname, in_path in layer_map.items():
        df = _read_any_layer(in_path)
        if df.empty:
            print(f"[MC-ZHE] skip {lname}: no data in {in_path.name} (xlsx/csv)")
            continue

        clean = _ensure_six_cols(df)
        out_txt = OUT_DIR / f"MC_zhe_{lname}.txt"
        clean.to_csv(out_txt, sep="\t", index=False, encoding="utf-8")
        print(f"[MC-ZHE] wrote {len(clean)} rows -> {out_txt}")

if __name__ == "__main__":
    main_part5_export_mc_zhe_txt()

# =========================================
# Shared header (imports, config, paths, utils)
# =========================================
import re
import pandas as pd
from pathlib import Path

class Config:
    ID_COL      = 1
    TARGET_COL  = 4
    CONTEXT_COL = 3
    NEXT_COL    = 5

    @staticmethod
    def ordinal_suffix(n: int) -> str:
        if 10 <= n % 100 <= 20:
            return f"{n}th"
        return f"{n}{ {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th') }"

BASE_DIR     = Path("/data/dataset/groups by century")
TAGGED_DIR   = BASE_DIR / "Tagged_zhe"
UNTAGGED_DIR = BASE_DIR / "Untagged_zhe"

def _centuries_10_to_19(base_dir: Path):
    """Return ['10th century', ..., '19th century'] filtered by existing dirs."""
    candidates = [f"{Config.ordinal_suffix(i)} century" for i in range(10, 20)]
    kept = []
    for name in candidates:
        if ((base_dir / "Tagged_zhe" / name).exists()
            or (base_dir / "Untagged_zhe" / name).exists()
            or (base_dir / name).exists()):
            kept.append(name)
    return kept if kept else candidates  # fallback: avoid accidental empty scan

CENTURIES    = _centuries_10_to_19(BASE_DIR)

OUT_DIR      = Path("/data/dataset/layered data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_rows():
    """Scan both Tagged/Untagged data by century; return unified dataframe."""
    records = []
    for is_tagged, root in [(True, TAGGED_DIR), (False, UNTAGGED_DIR)]:
        for label in CENTURIES:
            cdir = root / label
            if not cdir.exists():
                continue
            for fpath in cdir.glob("*.xlsx"):
                try:
                    df = pd.read_excel(fpath, header=None, engine="openpyxl")
                except Exception as e:
                    print(f"[warn] read fail: {fpath} -> {e}")
                    continue
                for _, row in df.iterrows():
                    records.append({
                        "file":      fpath.stem,
                        "ID":        str(row.iloc[Config.ID_COL-1]).strip(),
                        "context":   str(row.iloc[Config.CONTEXT_COL-1]),
                        "zhe":       str(row.iloc[Config.TARGET_COL-1]),
                        "next":      str(row.iloc[Config.NEXT_COL-1]),
                        "is_tagged": is_tagged,
                        "source_dir":"Tagged" if is_tagged else "Untagged",
                    })
    cols = ["file","ID","context","zhe","next","is_tagged","source_dir"]
    df = pd.DataFrame.from_records(records, columns=cols)
    for c in ["file","ID","context","zhe","next"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def safe_read_xlsx(path: Path, columns=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
    return pd.read_excel(path, dtype=str, engine="openpyxl")

# =========================================================
# ---------- Part 1: old L3 & old L1  →  New L1 & New L2
# =========================================================
# Mapping:
# - old L3 (T3/U3) → New Layer1 (T1/U1)  → zhe_L1_blend_T1_U1.*
# - old L1 (T1/U1) → New Layer2 (T2/U2)  → zhe_L2_blend_T2_U2.*

# old L3 → New L1
def _oldL3_tagged(ctx: str, zhe: str, nxt: str):
    full_text = f"{ctx} {zhe} {nxt}"
    if re.search(r'(?:^|\s)著\((Di|T)\)', full_text):
        return True, 'T1: 著(Di/T)'
    patterns = [
        (r'\w+\(N[^)]*\)\s+\w+\(V[^)]*\)\s+\w+\(N[^)]*\)\s+著\([^)]*\)', 'T1: N-V-N 著(V)'),
        (r'\w+\(N[^)]*\)\s+\w+\(V[^)]*\)\s+\w+\((Dg|VCL|Ng|Nc)[^)]*\)\s+著\(V[^)]*\)', 'T1: N-V-(Dg|VCL|Ng|Nc) 著(V)'),
        (r'\w+\(V[^)]*\)\s+\w+\(V[^)]*\)\s+著\(V[^)]*\)', 'T1: V-V 著(V)'),
    ]
    for pat, label in patterns:
        if re.search(pat, full_text):
            return True, label
    return False, ''

def _oldL3_untagged(ctx: str, zhe: str, nxt: str):
    combo_prev = f"{ctx}{zhe}"
    zhenxt = f"{zhe}{nxt}"
    if re.search(r'所[\u4e00-\u9fff]*著', combo_prev):
        return False, ''
    if re.search(r'著[上下内中前後东西南北於于在間]', zhenxt):
        return False, ''
    if re.search(r'著[\u4e00-\u9fff]{2,}', zhenxt):
        return False, ''
    txt = f"{ctx}{zhe}{nxt}"
    if re.search(r'[\u4e00-\u9fff]著(?![\u4e00-\u9fff])', txt):
        return True, 'U1: 汉字+著 + 非汉字/句边'
    if re.search(r'[\u4e00-\u9fff]著[\u4e00-\u9fff](?=[^\u4e00-\u9fff]|$)', txt):
        return True, 'U1: 汉字+著+汉字(1) + 非汉字/句边'
    return False, ''

# old L1 → New L2
def _oldL1_tagged(ctx: str, zhe: str, nxt: str):
    full_text = f"{ctx} {zhe} {nxt}"
    if re.search(r'著\(V[^)]*\)(?:\s+\w+\([^)]*\))*\s+([上下内中前後东西南北於于在間])\([^)]*\)', full_text):
        return True, 'T2: 著(V...) ... 方位词'
    if re.search(r'著\(V[^)]*\)\s+([上下内中前後东西南北於于在間])\([^)]*\)', full_text):
        return True, 'T2: 著(V...) 方位词'
    if re.search(r'著\(V[^)]*\)\s+\w+\(N[^)]*\)\w+\(Ng\)', f"{zhe} {nxt}"):
        return True, 'T2: 著(V) + N + Ng'
    if re.search(r'著\(V[^)]*\)\s+\w+\(Na\)\w+\(Na\)', f"{zhe} {nxt}"):
        return True, 'T2: 著(V) + Na + Na'
    if re.search(r'\w+\(V[^)]*\).*?著\([^)]*\).*?\w+\(N[cdg][^)]*\)', full_text):
        return True, 'T2: V ... 著(...) ... N[cdg]'
    if re.search(r'著\(V[^)]*\)\s+\w+\(N[cdg][^)]*\)', f"{zhe} {nxt}"):
        return True, 'T2: 著(V) + N[cdg]'
    if re.search(r'\w+\(V[^)]*\).*?著\([^)]*\).*?[上下内中前後东西南北於于在間]', full_text):
        return True, 'T2: V ... 著(...) ... 方位字'
    return False, ''

def _oldL1_untagged(ctx: str, zhe: str, nxt: str):
    zhenxt = f"{zhe}{nxt}"
    if re.search(r'著[上下内中前後东西南北於于在間]', zhenxt):
        return True, 'U2: 著 + 方位字'
    if re.search(r'著[\u4e00-\u9fff][上下内中前後东西南北於于在間]', zhenxt):
        return True, 'U2: 著 + 任一字 + 方位字'
    return False, ''

def main_part1():
    df_all = load_all_rows()
    if df_all.empty:
        print("[warn] no rows found.")
        return

    # New L1 from old L3 (T1/U1)
    l1_rows = []
    for _, r in df_all.iterrows():
        ctx, zhe, nxt = r["context"], r["zhe"], r["next"]
        if r["is_tagged"]:
            hit, cond = _oldL3_tagged(ctx, zhe, nxt)
            if hit:
                l1_rows.append({"file": r["file"], "ID": r["ID"], "label": "T1", "condition": cond,
                                "context": ctx, "zhe": zhe, "next": nxt, "source_dir": r["source_dir"]})
        else:
            hit, cond = _oldL3_untagged(ctx, zhe, nxt)
            if hit:
                l1_rows.append({"file": r["file"], "ID": r["ID"], "label": "U1", "condition": cond,
                                "context": ctx, "zhe": zhe, "next": nxt, "source_dir": r["source_dir"]})
    df_l1 = pd.DataFrame(l1_rows, columns=["file","ID","label","condition","context","zhe","next","source_dir"])
    out_l1_xlsx = OUT_DIR / "zhe_L1_blend_T1_U1.xlsx"
    out_l1_csv  = OUT_DIR / "zhe_L1_blend_T1_U1.csv"
    if not df_l1.empty:
        df_l1.to_excel(out_l1_xlsx, index=False, engine="openpyxl")
        df_l1.to_csv(out_l1_csv, index=False, encoding="utf-8-sig")
    print(f"[L1-blend] rows: {len(df_l1)} -> {out_l1_xlsx}")

    # New L2 from old L1 (T2/U2), prefer T2 then U2
    hits = []
    for _, r in df_all.iterrows():
        ctx, zhe, nxt = r["context"], r["zhe"], r["next"]
        if r["is_tagged"]:
            t2, t2c = _oldL1_tagged(ctx, zhe, nxt)
            hits.append((t2, t2c, False, ""))      # (hit_T2, cond_T2, hit_U2, cond_U2)
        else:
            u2, u2c = _oldL1_untagged(ctx, zhe, nxt)
            hits.append((False, "", u2, u2c))
    df_all[["hit_T2","cond_T2","hit_U2","cond_U2"]] = pd.DataFrame(hits, index=df_all.index)

    def choose_L2(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        g["__ord"] = range(len(g))
        t2 = g[g["hit_T2"]]
        if not t2.empty:
            pick = t2.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "T2",
                              "condition": pick["cond_T2"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"]})
        u2 = g[g["hit_U2"]]
        if not u2.empty:
            pick = u2.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "U2",
                              "condition": pick["cond_U2"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"]})
        return pd.Series({"file": g.iloc[0]["file"], "ID": g.iloc[0]["ID"],
                          "label": "", "condition": "", "context": "", "zhe": "", "next": "", "source_dir": ""})

    l2_result = (df_all.groupby(["file","ID"], as_index=False)
                        .apply(choose_L2)
                        .reset_index(drop=True))
    df_l2 = l2_result[l2_result["label"]!=""].copy()
    out_l2_xlsx = OUT_DIR / "zhe_L2_blend_T2_U2.xlsx"
    out_l2_csv  = OUT_DIR / "zhe_L2_blend_T2_U2.csv"
    if not df_l2.empty:
        df_l2.to_excel(out_l2_xlsx, index=False, engine="openpyxl")
        df_l2.to_csv(out_l2_csv, index=False, encoding="utf-8-sig")
    print(f"[L2-blend] rows: {len(df_l2)} -> {out_l2_xlsx}")

    n_keys = df_all[["file","ID"]].drop_duplicates().shape[0]
    print(f"[scan] unique (file,ID): {n_keys}")
    print(f"[done] New L1: {len(df_l1)} | New L2: {len(df_l2)}")

if __name__ == "__main__":
    main_part1()

# =====================================
# ---------- Part 2: Layer4 (unchanged)
# =====================================
L3_BLEND_XLSX = OUT_DIR / "zhe_L1_blend_T1_U1.xlsx"  # New L1
L1_BLEND_XLSX = OUT_DIR / "zhe_L2_blend_T2_U2.xlsx"  # New L2
L4_OUT_XLSX   = OUT_DIR / "zhe_L4_from_U4_T4.xlsx"
L4_OUT_CSV    = OUT_DIR / "zhe_L4_from_U4_T4.csv"

def _untagged_U4(ctx: str, zhe: str, nxt: str):
    if re.search(r'所[\u4e00-\u9fff]*著', f"{ctx}{zhe}"):
        return True, "U4: 所+著"
    return False, ""

def _tagged_T4(ctx: str, zhe: str, nxt: str):
    if re.search(r'所[\u4e00-\u9fff]*\s+著', f"{ctx} {zhe}"):
        return True, "T4: 所+著 (non-default)"
    return False, ""

def main_part2():
    df_all = load_all_rows()
    if df_all.empty:
        print("[warn] no rows found.")
        return

    flags = []
    for _, r in df_all.iterrows():
        ctx, zhe, nxt = r["context"], r["zhe"], r["next"]
        u4_hit, u4_cond = _untagged_U4(ctx, zhe, nxt)
        t4_hit, t4_cond = (False, "")
        if r["is_tagged"]:
            t4_hit, t4_cond = _tagged_T4(ctx, zhe, nxt)
        flags.append((u4_hit, u4_cond, t4_hit, t4_cond))
    df_all[["hit_U4","cond_U4","hit_T4","cond_T4"]] = pd.DataFrame(flags, index=df_all.index)

    def decide_L4(g: pd.DataFrame) -> pd.Series:
        g = g.copy(); g["__ord"] = range(len(g))
        u4 = g[g["hit_U4"]]
        if not u4.empty:
            pick = u4.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "U4",
                              "condition": pick["cond_U4"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"],
                              "audit_T4": "1" if g["hit_T4"].any() else "0"})
        t4 = g[g["hit_T4"]]
        if not t4.empty:
            pick = t4.sort_values("__ord").iloc[0]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "T4",
                              "condition": pick["cond_T4"], "context": pick["context"],
                              "zhe": pick["zhe"], "next": pick["next"], "source_dir": pick["source_dir"],
                              "audit_T4": "1"})
        return pd.Series({"file": g.iloc[0]["file"], "ID": g.iloc[0]["ID"],
                          "label": "", "condition": "", "context": "", "zhe": "", "next": "",
                          "source_dir": "", "audit_T4": "0"})

    l4_result = (df_all.groupby(["file","ID"], as_index=False, group_keys=False)
                        .apply(decide_L4)
                        .reset_index(drop=True))
    df_l4 = l4_result[l4_result["label"]!=""].copy()

    if not df_l4.empty:
        df_l4.to_excel(L4_OUT_XLSX, index=False, engine="openpyxl")
        df_l4.to_csv(L4_OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[L4] rows: {len(df_l4)} -> {L4_OUT_XLSX}")

    # Remove keys from New L1/New L2
    keys_l4 = set(df_l4[["file","ID"]].itertuples(index=False, name=None))

    def rewrite_without_keys(path: Path, keys: set):
        df = safe_read_xlsx(path, columns=["file","ID","label","condition","context","zhe","next","source_dir"])
        if df.empty: return
        keep = ~df.apply(lambda r: (str(r["file"]), str(r["ID"])) in keys, axis=1)
        df_new = df.loc[keep].copy()
        df_new.to_excel(path, index=False, engine="openpyxl")
        df_new.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        print(f"[rewrite] {path.name}: {len(df)} -> {len(df_new)} (removed {len(df)-len(df_new)})")

    rewrite_without_keys(L3_BLEND_XLSX, keys_l4)  # New L1 file
    rewrite_without_keys(L1_BLEND_XLSX, keys_l4)  # New L2 file

if __name__ == "__main__":
    main_part2()

# ===================================================
# ---------- Part 3: old L2  →  New Layer3
# ===================================================
# Mapping:
# - old L2 (T2/U2) → New Layer3 (T3/U3) → zhe_L3_from_U3_T3.*

L4_FILE   = OUT_DIR / "zhe_L4_from_U4_T4.xlsx"
NEW_L1    = OUT_DIR / "zhe_L1_blend_T1_U1.xlsx"
NEW_L2    = OUT_DIR / "zhe_L2_blend_T2_U2.xlsx"
L3_OUT_X  = OUT_DIR / "zhe_L3_from_U3_T3.xlsx"
L3_OUT_C  = OUT_DIR / "zhe_L3_from_U3_T3.csv"

def _oldL2_untagged(ctx: str, zhe: str, nxt: str):
    combo_all = f"{ctx}{zhe}{nxt}"
    zhenxt = f"{zhe}{nxt}"
    if re.search(r'[不弗勿毋未非莫没無]著(?![上下内中前後东西南北於于在間])(?![\u4e00-\u9fff][上下内中前後东西南北於于在間])', combo_all):
        return True, 'U3: Untagged Layer2 cond2'
    if re.search(r'著[\u4e00-\u9fff]{2,}', zhenxt):
        return True, 'U3: Untagged Layer2 cond1'
    if re.search(r'著[不弗勿毋未非莫没無]', zhenxt):
        return True, 'U3: Untagged Layer2 cond3'
    return False, ''

def _oldL2_tagged(ctx: str, zhe: str, nxt: str):
    if re.search(r'著\(V[^)]*\)', zhe):
        return True, 'T3: Tagged Layer2 default'
    return False, ''

def main_part3():
    df_all = load_all_rows()
    if df_all.empty:
        print("[warn] no rows found.")
        return

    produced = set()
    for f in [L4_FILE, NEW_L1, NEW_L2]:
        dfk = safe_read_xlsx(f)
        if not dfk.empty and {'file','ID'}.issubset(dfk.columns):
            produced |= set(dfk[['file','ID']].itertuples(index=False, name=None))

    df_remaining = df_all[~df_all.apply(lambda r: (r['file'], r['ID']) in produced, axis=1)].copy()

    def choose_L3(g: pd.DataFrame) -> pd.Series:
        g = g.copy(); g['__ord'] = range(len(g))
        u3 = []
        for i, r in g.iterrows():
            hit, cond = _oldL2_untagged(r['context'], r['zhe'], r['next'])
            if hit: u3.append((i, cond))
        if u3:
            idx, cond = sorted(u3, key=lambda x: x[0])[0]
            pick = g.loc[idx]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "U3",
                              "condition": cond, "context": pick["context"], "zhe": pick["zhe"],
                              "next": pick["next"], "source_dir": pick["source_dir"]})
        t3 = []
        for i, r in g.iterrows():
            if r['is_tagged']:
                hit, cond = _oldL2_tagged(r['context'], r['zhe'], r['next'])
                if hit: t3.append((i, cond))
        if t3:
            idx, cond = sorted(t3, key=lambda x: x[0])[0]
            pick = g.loc[idx]
            return pd.Series({"file": pick["file"], "ID": pick["ID"], "label": "T3",
                              "condition": cond, "context": pick["context"], "zhe": pick["zhe"],
                              "next": pick["next"], "source_dir": pick["source_dir"]})
        return pd.Series({"file": g.iloc[0]["file"], "ID": g.iloc[0]["ID"], "label": "",
                          "condition": "", "context": "", "zhe": "", "next": "", "source_dir": ""})

    l3_from_remaining = (df_remaining.groupby(['file','ID'], as_index=False, group_keys=False)
                                     .apply(choose_L3)
                                     .reset_index(drop=True))
    l3_from_remaining = l3_from_remaining[l3_from_remaining['label']!=''].copy()

    df_l3_all = l3_from_remaining

    if not df_l3_all.empty:
        df_l3_all["__ord"] = range(len(df_l3_all))
        df_l3_all = (df_l3_all.sort_values(["file","ID","__ord"])
                                .drop_duplicates(subset=["file","ID"], keep="first")
                                .drop(columns=["__ord"]))
        df_l3_all.to_excel(L3_OUT_X, index=False, engine="openpyxl")
        df_l3_all.to_csv(L3_OUT_C, index=False, encoding="utf-8-sig")

    n_all  = df_all[['file','ID']].drop_duplicates().shape[0]
    n_rem  = df_remaining[['file','ID']].drop_duplicates().shape[0]
    print(f"[scan] unique (file,ID): {n_all}")
    print(f"[scan] remaining after L4/L1/L2: {n_rem}")
    print(f"[L3] total written: {len(df_l3_all)} -> {L3_OUT_X}")

if __name__ == "__main__":
    main_part3()

# ==================================================
# ---------- Part 4: Remaining Export (unchanged)
# ==================================================
def _read_layer_keys(path: Path):
    if path.exists():
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    elif path.with_suffix(".csv").exists():
        df = pd.read_csv(path.with_suffix(".csv"), dtype=str)
    else:
        return set()
    if not {'file','ID'}.issubset(df.columns):
        return set()
    df = df[['file','ID']].fillna("").astype(str)
    return set(df.itertuples(index=False, name=None))

def main_part4():
    layer_files = [
        OUT_DIR / "zhe_L1_blend_T1_U1.xlsx",
        OUT_DIR / "zhe_L2_blend_T2_U2.xlsx",
        OUT_DIR / "zhe_L3_from_U3_T3.xlsx",
        OUT_DIR / "zhe_L4_from_U4_T4.xlsx",
    ]

    produced_keys = set()
    for p in layer_files:
        produced_keys |= _read_layer_keys(p)

    def scan_dir(root: Path, src_label: str):
        rows = []
        for label in CENTURIES:
            cdir = root / label
            if not cdir.exists():
                continue
            for f in cdir.glob("*.xlsx"):
                try:
                    df = pd.read_excel(f, header=None, engine="openpyxl", dtype=str)
                except Exception:
                    continue
                tmp = pd.DataFrame({
                    "file": f.stem,
                    "ID":   df.iloc[:, 0].astype(str).str.strip(),
                    "context": df.iloc[:, 2].astype(str),
                    "zhe":     df.iloc[:, 3].astype(str),
                    "next":    df.iloc[:, 4].astype(str),
                    "source_dir": src_label
                })
                rows.append(tmp[["file","ID","context","zhe","next","source_dir"]])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["file","ID","context","zhe","next","source_dir"]
        )

    df_t = scan_dir(TAGGED_DIR, "Tagged")
    df_u = scan_dir(UNTAGGED_DIR, "Untagged")
    df_all = pd.concat([df_t, df_u], ignore_index=True)

    for c in ["file","ID","context","zhe","next","source_dir"]:
        df_all[c] = df_all[c].fillna("").astype(str).str.strip()

    all_keys = set(df_all[["file","ID"]].itertuples(index=False, name=None))
    remaining_keys = all_keys - produced_keys

    df_tag = (df_all[df_all["source_dir"]=="Tagged"]
              .drop_duplicates(subset=["file","ID"], keep="last"))
    df_unt = (df_all[df_all["source_dir"]=="Untagged"]
              .drop_duplicates(subset=["file","ID"], keep="last"))

    df_tag_rem = df_tag[df_tag.apply(lambda r: (r["file"], r["ID"]) in remaining_keys, axis=1)]
    picked = set(df_tag_rem[["file","ID"]].itertuples(index=False, name=None))
    df_unt_rem = df_unt[df_unt.apply(lambda r: (r["file"], r["ID"]) in (remaining_keys - picked), axis=1)]

    df_rem_pref = pd.concat([df_tag_rem, df_unt_rem], ignore_index=True)
    df_rem_pref = df_rem_pref.drop_duplicates(subset=["file","ID"], keep="first")

    out_path = OUT_DIR / "zhe_remaining_pref.xlsx"
    df_rem_pref.to_excel(out_path, index=False, engine="openpyxl")
    print(f"[done] remaining list -> {out_path} (rows: {len(df_rem_pref)})")

if __name__ == "__main__":
    main_part4()

# ==================================================
# ---------- Part 5 (EMoC): EMoC ZHE clean TXT export
# ==================================================
# Produce 6-column, tab-separated files to mirror the EMoC_zhe_* style:
#   file   ID   context   zhe   next   condition

def _read_any_layer_emoc(path: Path) -> pd.DataFrame:
    """Read a layer table from .xlsx if present; else from .csv; else empty."""
    if path.exists():
        return pd.read_excel(path, dtype=str, engine="openpyxl").fillna("")
    csvp = path.with_suffix(".csv")
    if csvp.exists():
        return pd.read_csv(csvp, dtype=str).fillna("")
    return pd.DataFrame()

def _ensure_six_cols_emoc(df: pd.DataFrame) -> pd.DataFrame:
    """Select and coerce the exact six columns in the required order."""
    need = ["file","ID","context","zhe","next","condition"]
    for c in need:
        if c not in df.columns:
            df[c] = ""
    return df[need].astype(str)

def main_part5_export_emoc_zhe_txt():
    # Map each EMoC layer's table to its exported TXT name.
    layer_map = {
        "layer1": OUT_DIR / "zhe_L1_blend_T1_U1.xlsx",
        "layer2": OUT_DIR / "zhe_L2_blend_T2_U2.xlsx",
        "layer3": OUT_DIR / "zhe_L3_from_U3_T3.xlsx",
        "layer4": OUT_DIR / "zhe_L4_from_U4_T4.xlsx",
    }
    for lname, in_path in layer_map.items():
        df = _read_any_layer_emoc(in_path)
        if df.empty:
            print(f"[EMoC-ZHE] skip {lname}: no data in {in_path.name} (xlsx/csv)")
            continue

        clean = _ensure_six_cols_emoc(df)
        out_txt = OUT_DIR / f"EMoC_zhe_{lname}.txt"
        clean.to_csv(out_txt, sep="\t", index=False, encoding="utf-8")
        print(f"[EMoC-ZHE] wrote {len(clean)} rows -> {out_txt}")

if __name__ == "__main__":
    main_part5_export_emoc_zhe_txt()