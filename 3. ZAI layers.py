# -*- coding: utf-8 -*-
"""ZAI.ipynb

"""

# The following lines are only used on Google Colab and are commented out by default in this repository version.
#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

# -*- coding: utf-8 -*-
"""
============================================================
Keep-last Dedup + Audit + Alignment with Initial (cols 3/4/5)
============================================================
"""

# ====================
# Imports
# ====================
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import pandas as pd


# ====================
# Global Config
# ====================
class Config:
    """
    Centralized configuration.
    Column indices are 1-based to mirror the original spreadsheets.
    """
    ID_COL: int = 1       # unique row identifier (1-based)
    TARGET_COL: int = 4   # column of token '在'
    CONTEXT_COL: int = 3  # left context
    NEXT_COL: int = 5     # right context

    # Deduplication key mode:
    # - 'text_cond': cross-file/ID dedup; key = (context, zai, next, condition)
    # - 'file_id_text_cond': conservative; includes (file, id)
    DEDUP_KEY_MODE: str = 'text_cond'

    # Whether to preserve multiplicity from the initial sheet. When False, we fold by (c,z,n).
    PRESERVE_INITIAL_MULTIPLICITY: bool = True

    # Top-N for overwritten report
    TOP_N: int = 100

    @staticmethod
    def ordinal_suffix(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"


# ====================
# IO Paths
# ====================
BASE_DIR   = Path("/data/dataset/groups by century")
TAGGED_DIR = BASE_DIR / "Tagged_zai"
CENTURIES  = [f"{Config.ordinal_suffix(i)} century" for i in range(3, 10)]

OUT_DIR    = Path("/data/dataset/layered data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Read-only initial sheet (for final alignment; we only read cols 3/4/5)
INIT_XLSX  = Path("/data/dataset/initial collection/TaggedZAI_MC.xlsx")

# Constant layer order
LAYER_ORDER: Tuple[str, ...] = ("Layer1", "Layer2", "Layer3", "Layer4")


# ====================
# Tagger (Layer 1 subdivision + original L2–L4)
# ====================
class TaggedProcessor:
    """
    Layer1 subdivision（cond_sub）：
      L1_zhengzai_V        : 正在 + V
      L1_zai_V_zhe         : 在 + V (+0–3) + 著/呢
      L1_trigZai_neg       : 正/方/值/適 (+Light adv.) + 在 + negation(+0–3)
      L1_trigZai_V         : 正/方/值/適 (+Light adv.) + 在(V...) + V  （near V）
      L1_trigZai_V_loose   : 正/方/值/適 (+Light adv.) + 在 + 1 word + V  （loosen an one-word position）
      L1_other             : the remaining cases fall into Layer 1
    The rules for the remaining layers remain unchanged.
    """
    TOKEN   = r'[^\s()]+'                            # Non-blank and not parentheses
    TOK     = rf'{TOKEN}\([^)]*\)'                   # Forms like: characters (...)
    VERB    = rf'{TOKEN}\(V[^)]*\)'
    TRIGGER = re.compile(r'[正方值適]\([^)]*\)')     # Trigger: 正、方、值、適
    LIGHT   = r'(?:[只便又還皆都且將]\([^)]*\)\s*)?' # Light adv. 0–1
    ZAI_ANY = r'在\([^)]*\)'
    ZAI_V   = r'在\(V[^)]*\)'
    NEGCHR  = r'[不弗勿毋未非莫沒無]'

    # —— Layer 1 subdivision using regular expressions (order: strong → wide) ——
    PAT_ZHENGZAI_V   = re.compile(rf'正在\([^)]*\)\s*{VERB}')
    PAT_ZAI_V_ZHE    = re.compile(rf'{ZAI_ANY}\s*{VERB}(?:\s*{TOK}){{0,3}}(?:著|呢)')
    PAT_TRIG_NEG     = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{NEGCHR}{TOK}(?:\s*{TOK}){{0,3}}')
    PAT_TRIG_V       = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_V}\s*{VERB}')
    PAT_TRIG_V_LOOSE = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{TOK}\s*{VERB}')

    # —— Original Layer 2–4 Decision ——
    _re_l2_1 = re.compile(r'在\(P\)')
    _re_l2_2 = re.compile(r"\w+\(V[^)]*\)在(?!\w+\(V\))\s*\w+\(N[cdg][^)]*\)")
    _re_l2_3 = re.compile(r"\w+\(V[^)]*\)在\w+\(Na\)\s+\w+\(Na\)")

    _re_l3_1 = re.compile(r'\w+\(VCL\)(?:\s+\w+\(N[cdg][^)]*\)|\s+\w+\(Na\)\s+\w+\(Ng\))')
    _re_l3_2 = re.compile(r"\w*\(N[^)]*\)在\(V[^)]*\)\w+\(N[^)]*\)")
    _re_l3_3 = re.compile(r"\w*\(N[^)]*\)在\(V[^)]*\)(此|是)")

    _re_l4_1 = re.compile(r'\w+\(T8\)\s+在\([^)]*\)(?:\s+\w+\([^)]*\))*\s+\w+\(V[^)]*\)')

    @staticmethod
    def classify(record):
        """
        Return (layer, condition, cond_sub)
        """
        ctx, zai, nxt = record['context'], record['zai'], record['next']
        text = f"{ctx}{zai}{nxt}"

        # ===== Layer1 Subdivision =====
        if TaggedProcessor.PAT_ZHENGZAI_V.search(text):
            return 'Layer1', 'Tagged 正在+V', 'L1_zhengzai_V'
        if TaggedProcessor.PAT_ZAI_V_ZHE.search(text):
            return 'Layer1', 'Tagged 在+动词+著/呢', 'L1_zai_V_zhe'
        if TaggedProcessor.PAT_TRIG_NEG.search(text):
            return 'Layer1', 'Tagged 触发+在+否定', 'L1_trigZai_neg'
        if TaggedProcessor.PAT_TRIG_V.search(text):
            return 'Layer1', 'Tagged 触发+在(动词)+动词', 'L1_trigZai_V'
        if TaggedProcessor.PAT_TRIG_V_LOOSE.search(text):
            return 'Layer1', 'Tagged 触发+在+1词+动词', 'L1_trigZai_V_loose'
        if re.search(r'在\(T\)', zai):
            return 'Layer1', 'Tagged 在(T)', 'L1_other'

        # ===== Layer2 =====
        if TaggedProcessor._re_l2_1.search(zai):
            return 'Layer2', 'Tagged Layer2 cond3', None
        if TaggedProcessor._re_l2_2.search(text):
            return 'Layer2', 'Tagged Layer2 cond1', None
        if TaggedProcessor._re_l2_3.search(text):
            return 'Layer2', 'Tagged Layer2 cond2', None

        # ===== Layer3 =====
        if TaggedProcessor._re_l3_1.search(text):
            return 'Layer3', 'Tagged Layer3 cond3 VCL+N结构', None
        if TaggedProcessor._re_l3_2.search(text):
            return 'Layer3', 'Tagged Layer3 cond1', None
        if TaggedProcessor._re_l3_3.search(text):
            return 'Layer3', 'Tagged Layer3 cond2', None

        # ===== Layer4 =====
        if TaggedProcessor._re_l4_1.search(text):
            return 'Layer4', 'Tagged 所+在', None
        return 'Layer4', 'Tagged default', None


# ====================
# Processing System (keep-last; audit; initial alignment)
# ====================
class EnhancedProcessingSystem:
    """
    - In-file dedup by (context, zai, next), keep='last'
    - Cross-file keep-last by key (depends on Config.DEDUP_KEY_MODE)
    - Final alignment to initial sheet keys (context, zai, next)
    - Export per layer + logs + counts
    """
    def __init__(self) -> None:
        # store: unique_key -> (layer, entry_tuple)
        # entry: (file, id, context, zai, next, condition, cond_sub)
        self.store: Dict[Tuple[Any, ...], Tuple[str, Tuple[str, Any, str, str, str, str, Any]]] = {}
        self.results_by_layer: Dict[str, List[List[str]]] = defaultdict(list)

        self.overwritten: List[Tuple[str, Any, str, str, str, str, Any, str, str]] = []
        self.infile_dedup_count: int = 0
        self.scanned_rows: int = 0

    @staticmethod
    def _make_key(record, condition):
        # Note: Deduplication keys do not include cond_sub
        ctx = (record['context'] or "").strip()
        zai = (record['zai'] or "").strip()
        nxt = (record['next'] or "").strip()
        cond = (condition or "").strip()
        if Config.DEDUP_KEY_MODE == 'file_id_text_cond':
            file_ = (record['file'] or "").strip()
            id_   = str(record['id']).strip()
            return ('file_id_text_cond', file_, id_, ctx, zai, nxt, cond)
        return ('text_cond', ctx, zai, nxt, cond)

    def _upsert(self, layer, record, condition, cond_sub=None):
        key = self._make_key(record, condition)
        entry = (record['file'], record['id'], record['context'],
                 record['zai'], record['next'], condition, cond_sub)
        if key in self.store:
            prev_layer, prev_entry = self.store[key]
            self.overwritten.append((*prev_entry, prev_layer, 'OVERWRITTEN_KEEP_LAST'))
        self.store[key] = (layer, entry)

    def process_files(self, base_dir: Path) -> None:
        for century in CENTURIES:
            dir_path = base_dir / century
            if not dir_path.exists():
                print(f"[warn] Directory does not exist: {dir_path}")
                continue

            for file in dir_path.glob('*.xlsx'):
                df = pd.read_excel(file, header=None, engine='openpyxl')
                self.scanned_rows += len(df)

                # In-file dedup: (context, zai, next), keep last
                before = len(df)
                df = df.drop_duplicates(
                    subset=[Config.CONTEXT_COL-1, Config.TARGET_COL-1, Config.NEXT_COL-1],
                    keep='last'
                )
                removed = before - len(df)
                if removed > 0:
                    self.infile_dedup_count += removed
                    print(f"[dedup-file] {file.name}: {removed} duplicate rows removed (keep=last)")

                # Classify and upsert
                for _, row in df.iterrows():
                    record = {
                        'file': Path(file.name).stem,
                        'id': row.iloc[Config.ID_COL-1],
                        'context': str(row.iloc[Config.CONTEXT_COL-1]),
                        'zai': str(row.iloc[Config.TARGET_COL-1]),
                        'next': str(row.iloc[Config.NEXT_COL-1]),
                    }
                    layer, cond, cond_sub = TaggedProcessor.classify(record)
                    self._upsert(layer, record, cond, cond_sub)

    def _collect_all_kept(self) -> pd.DataFrame:
        rows: List[Tuple[Any, ...]] = []
        for layer, entry in self.store.values():
            rows.append((*entry, layer))
        if not rows:
            return pd.DataFrame(columns=['file','ID','context','zai','next','condition','cond_sub','layer'])
        return pd.DataFrame(rows, columns=['file','ID','context','zai','next','condition','cond_sub','layer'])

    def _filter_by_initial(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retain only rows whose (context,zai,next) appear in INIT_XLSX.
        Then (optionally) fold by (c,z,n) with keep-last unless PRESERVE_INITIAL_MULTIPLICITY is True.
        """
        init_raw = pd.read_excel(
            INIT_XLSX, header=None, dtype=str, engine='openpyxl', usecols=[2,3,4]
        )
        init_raw.columns = ['context','zai','next']
        init_raw = init_raw.fillna("").astype(str)
        init_keys = set(map(tuple, init_raw[['context','zai','next']].itertuples(index=False, name=None)))

        mask = df.apply(lambda r: (r['context'], r['zai'], r['next']) in init_keys, axis=1)
        df2 = df.loc[mask].copy()

        if not Config.PRESERVE_INITIAL_MULTIPLICITY:
            df2 = df2.drop_duplicates(subset=['context','zai','next'], keep='last')
        return df2

    def export_results(self, output_dir: Path) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Gather kept
        all_kept = self._collect_all_kept()

        # 2) Align to initial + optional (c,z,n) folding
        all_kept = self._filter_by_initial(all_kept)

        # 3) Final keep-last by dedup subset（not including 'cond_sub'）
        if Config.DEDUP_KEY_MODE == 'file_id_text_cond':
            subset = ['file','ID','context','zai','next','condition']
        else:
            subset = ['context','zai','next','condition']
        all_kept = all_kept.drop_duplicates(subset=subset, keep='last')

        # 4) Write per layer（Export containing 'cond_sub'）
        total_written = 0
        for layer in LAYER_ORDER:
            dfl = all_kept[all_kept['layer'] == layer].copy()
            dfl = dfl[['file','ID','context','zai','next','condition','cond_sub']]
            total_written += len(dfl)
            dfl.to_csv(
                output_dir / f"MC_zai_{layer.lower()}.txt",
                sep='\t',
                index=False,
                encoding='utf-8'
            )

        # 5) Write overwritten log (if any) + inside-file dedup info
        if self.overwritten or self.infile_dedup_count > 0:
            dup_cols = ['file','ID','context','zai','next','condition','cond_sub','pred_layer','reason']
            pd.DataFrame(self.overwritten, columns=dup_cols).to_csv(
                output_dir / "duplicates_log.tsv",
                sep='\t',
                index=False,
                encoding='utf-8'
            )
            print(f"[log] duplicates_log.tsv written with {len(self.overwritten)} overwritten rows "
                  f"+ {self.infile_dedup_count} removed inside files")

        # Cache for summary + reports
        self.results_by_layer.clear()
        for layer in LAYER_ORDER:
            self.results_by_layer[layer] = all_kept[all_kept['layer']==layer][
                ['file','ID','context','zai','next','condition','cond_sub']
            ].values.tolist()

        return total_written

    def print_summary(self) -> None:
        print("\n=== Summary (keep-last, after initial-filter) ===")
        total_kept = 0
        for layer in LAYER_ORDER:
            count = len(self.results_by_layer.get(layer, []))
            print(f"{layer}: {count} rows")
            total_kept += count
        print(f"Total kept (exported): {total_kept}")
        print(f"Scanned raw rows                 : {self.scanned_rows}")
        print(f"Removed inside files (keep=last) : {self.infile_dedup_count}")
        print(f"Overwritten by key (keep-last)   : {len(self.overwritten)}")

        # Uniqueness audit
        all_rows: List[Tuple[Any, ...]] = []
        for layer in LAYER_ORDER:
            for r in self.results_by_layer.get(layer, []):
                all_rows.append((*r, layer))
        if all_rows:
            df = pd.DataFrame(all_rows, columns=['file','ID','context','zai','next','condition','cond_sub','layer'])
            n_ctn   = df.drop_duplicates(subset=['context','zai','next']).shape[0]
            n_ctnc  = df.drop_duplicates(subset=['context','zai','next','condition']).shape[0]
            print("\n--- Audit ---")
            print(f"Unique (context,zai,next)        : {n_ctn}")
            print(f"Unique (context,zai,next,cond)   : {n_ctnc}")
        else:
            print("\n--- Audit ---\n(no rows kept)")

    def export_count_report(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Overview
        kept_rows = sum(len(v) for v in self.results_by_layer.values())
        overview = {
            "scanned_raw_rows": self.scanned_rows,
            "removed_inside_files_keep_last": self.infile_dedup_count,
            "overwritten_keep_last": len(self.overwritten),
            "kept_rows_after_initial_filter": kept_rows,
        }

        # Uniqueness (global)
        all_rows: List[Tuple[Any, ...]] = []
        for layer in LAYER_ORDER:
            for r in self.results_by_layer.get(layer, []):
                all_rows.append((*r, layer))
        if all_rows:
            df = pd.DataFrame(all_rows, columns=['file','ID','context','zai','next','condition','cond_sub','layer'])
            overview["unique_ctn"]  = df.drop_duplicates(subset=['context','zai','next']).shape[0]
            overview["unique_ctnc"] = df.drop_duplicates(subset=['context','zai','next','condition']).shape[0]

        pd.DataFrame([overview]).to_csv(output_dir / "counts_overview.tsv", sep="\t", index=False)

        # Per-layer stats
        per_layer = []
        for layer in LAYER_ORDER:
            data = self.results_by_layer.get(layer, [])
            if not data:
                per_layer.append({"layer": layer, "kept": 0, "unique_ctn": 0, "unique_ctnc": 0})
            else:
                dfl = pd.DataFrame(data, columns=['file','ID','context','zai','next','condition','cond_sub'])
                per_layer.append({
                    "layer": layer,
                    "kept": len(dfl),
                    "unique_ctn":  dfl.drop_duplicates(subset=['context','zai','next']).shape[0],
                    "unique_ctnc": dfl.drop_duplicates(subset=['context','zai','next','condition']).shape[0],
                })
        pd.DataFrame(per_layer).to_csv(output_dir / "counts_per_layer.tsv", sep="\t", index=False)

        # Top overwritten keys + reason counts
        if self.overwritten:
            dropped_summary = pd.DataFrame(
              self.overwritten,
              columns=['file','ID','context','zai','next','condition','cond_sub','pred_layer','reason']
            )

            grp = (dropped_summary
                   .groupby(['context','zai','next','condition'], as_index=False)
                   .agg(count=('condition','size')))
            grp = grp.sort_values("count", ascending=False).head(Config.TOP_N)
            grp.to_csv(output_dir / "top_overwritten.tsv", sep="\t", index=False)

            dropped_summary.groupby("reason").size().reset_index(name="count") \
                .to_csv(output_dir / "overwritten_reason_counts.tsv", sep="\t", index=False)
        else:
            pd.DataFrame(columns=['context','zai','next','condition','count']).to_csv(
                output_dir / "top_overwritten.tsv", sep="\t", index=False
            )


# ====================
# Main
# ====================
def main() -> None:
    system = EnhancedProcessingSystem()
    system.process_files(TAGGED_DIR)
    written = system.export_results(OUT_DIR)
    system.print_summary()
    system.export_count_report(OUT_DIR)
    print(f"\n[done] Exported to: {OUT_DIR} (total rows written across layers: {written})")
    print(f"[dedup mode] {Config.DEDUP_KEY_MODE}  "
          f"({'cross-file strong dedup' if Config.DEDUP_KEY_MODE=='text_cond' else 'conservative dedup incl. file/ID'})")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Compare initial unique keys (context,zai,next) vs. kept layered outputs.
List keys present in initial but missing from layered results.
"""

# ====================
# Imports
# ====================
from pathlib import Path
import pandas as pd

# ====================
# IO Paths
# ====================
INIT_XLSX = Path("/data/dataset/initial collection/TaggedZAI_MC.xlsx")
OUT_DIR   = Path("/data/dataset/layered data")

# ====================
# 1) Read initial: use cols [1]=file, [2]=context, [3]=zai, [4]=next
# ====================
init_raw = pd.read_excel(INIT_XLSX, header=None, dtype=str, engine="openpyxl")
sub = init_raw.iloc[:, [1,2,3,4]].fillna("")
sub.columns = ["file","context","zai","next"]
init_unique = sub.drop_duplicates(subset=["context","zai","next"], keep="last")

# ====================
# 2) Read layered files; keep (3/4/5) as key + carry file for reference
# ====================
frames = []
for layer in ["layer1","layer2","layer3","layer4"]:
    p = OUT_DIR / f"MC_zai_{layer}.txt"
    if not p.exists():
        continue
    df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    frames.append(df[["context","zai","next","file"]])
if not frames:
    raise SystemExit("[ERR] No MC_zai_layer*.txt found in OUT_DIR")

kept = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["context","zai","next"], keep="last")

# ====================
# 3) Compute missing
# ====================
missing = init_unique.merge(
    kept[["context","zai","next"]],
    on=["context","zai","next"],
    how="left",
    indicator=True
)
missing = missing[missing["_merge"]=="left_only"].drop(columns=["_merge"]).copy()

print(f"[DIFF] initial unique = {len(init_unique)}")
print(f"[DIFF] kept (layered) unique = {kept[['context','zai','next']].drop_duplicates().shape[0]}")
print(f"[DIFF] missing (initial but not kept) = {len(missing)}\n")

for r in missing[["file","context","zai","next"]].itertuples(index=False):
    print("\t".join(map(str, r)))

# ====================
# 4) Save TSV (file, context, zai, next)
# ====================
out_path = OUT_DIR / "MC_missing_from_layered.tsv"
missing[["file","context","zai","next"]].to_csv(out_path, sep="\t", index=False, encoding="utf-8")
print(f"\n[Saved] {out_path}")

# -*- coding: utf-8 -*-
"""
Backfill rows (present in initial but missing from layered outputs) into the
four layer files by reclassifying with the same rules and appending
(without introducing exact duplicates on (context,zai,next,condition)).
"""

# ====================
# Imports
# ====================
from pathlib import Path
import re
import pandas as pd

# ====================
# IO Paths
# ====================
OUT_DIR = Path("/data/dataset/layered data")
MISSING_PATH = OUT_DIR / "MC_missing_from_layered.tsv"

# ====================
# Read missing list (expects: file, context, zai, next)
# ====================
missing = pd.read_csv(MISSING_PATH, sep="\t", dtype=str).fillna("")
missing = missing[["file", "context", "zai", "next"]].copy()

# ====================
# Classifier (identical to main; returns cond_sub)
# ====================
class TaggedProcessor:
    """
    Layer1 subdivision（cond_sub）：
      L1_zhengzai_V        : 正在 + V
      L1_zai_V_zhe         : 在 + V (+0–3) + 著/呢
      L1_trigZai_neg       : 正/方/值/適 (+Light adv.) + 在 + neagtion(+0–3)
      L1_trigZai_V         : 正/方/值/適 (+Light adv.) + 在(V...) + V  （near V）
      L1_trigZai_V_loose   : 正/方/值/適 (+Light adv.) + 在 + 1词 + V  （loosen an one-word position）
      L1_other             : The remaining cases fall into Layer 1
    The rules for the remaining layers remain unchanged.
    """
    TOKEN   = r'[^\s()]+'                            # Non-blank and not parentheses
    TOK     = rf'{TOKEN}\([^)]*\)'                   # Forms like characters (...)
    VERB    = rf'{TOKEN}\(V[^)]*\)'
    TRIGGER = re.compile(r'[正方值適]\([^)]*\)')     # Trigger: 正、方、值、適
    LIGHT   = r'(?:[只便又還皆都且將]\([^)]*\)\s*)?' # Light adv. 0-1
    ZAI_ANY = r'在\([^)]*\)'
    ZAI_V   = r'在\(V[^)]*\)'
    NEGCHR  = r'[不弗勿毋未非莫沒無]'

    PAT_ZHENGZAI_V   = re.compile(rf'正在\([^)]*\)\s*{VERB}')
    PAT_ZAI_V_ZHE    = re.compile(rf'{ZAI_ANY}\s*{VERB}(?:\s*{TOK}){{0,3}}(?:著|呢)')
    PAT_TRIG_NEG     = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{NEGCHR}{TOK}(?:\s*{TOK}){{0,3}}')
    PAT_TRIG_V       = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_V}\s*{VERB}')
    PAT_TRIG_V_LOOSE = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{TOK}\s*{VERB}')

    _re_l2_1 = re.compile(r'在\(P\)')
    _re_l2_2 = re.compile(r"\w+\(V[^)]*\)在(?!\w+\(V\))\s*\w+\(N[cdg][^)]*\)")
    _re_l2_3 = re.compile(r"\w+\(V[^)]*\)在\w+\(Na\)\s+\w+\(Na\)")

    _re_l3_1 = re.compile(r'\w+\(VCL\)(?:\s+\w+\(N[cdg][^)]*\)|\s+\w+\(Na\)\s+\w+\(Ng\))')
    _re_l3_2 = re.compile(r"\w*\(N[^)]*\)在\(V[^)]*\)\w+\(N[^)]*\)")
    _re_l3_3 = re.compile(r"\w*\(N[^)]*\)在\(V[^)]*\)(此|是)")

    _re_l4_1 = re.compile(r'\w+\(T8\)\s+在\([^)]*\)(?:\s+\w+\([^)]*\))*\s+\w+\(V[^)]*\)')

    @staticmethod
    def classify(record):
        ctx, zai, nxt = record['context'], record['zai'], record['next']
        text = f"{ctx}{zai}{nxt}"

        if TaggedProcessor.PAT_ZHENGZAI_V.search(text):
            return 'Layer1', 'Tagged 正在+V', 'L1_zhengzai_V'
        if TaggedProcessor.PAT_ZAI_V_ZHE.search(text):
            return 'Layer1', 'Tagged 在+动词+著/呢', 'L1_zai_V_zhe'
        if TaggedProcessor.PAT_TRIG_NEG.search(text):
            return 'Layer1', 'Tagged 触发+在+否定', 'L1_trigZai_neg'
        if TaggedProcessor.PAT_TRIG_V.search(text):
            return 'Layer1', 'Tagged 触发+在(动词)+动词', 'L1_trigZai_V'
        if TaggedProcessor.PAT_TRIG_V_LOOSE.search(text):
            return 'Layer1', 'Tagged 触发+在+1词+动词', 'L1_trigZai_V_loose'
        if re.search(r'在\(T\)', zai):
            return 'Layer1', 'Tagged 在(T)', 'L1_other'

        if TaggedProcessor._re_l2_1.search(zai):
            return 'Layer2', 'Tagged Layer2 cond3', None
        if TaggedProcessor._re_l2_2.search(text):
            return 'Layer2', 'Tagged Layer2 cond1', None
        if TaggedProcessor._re_l2_3.search(text):
            return 'Layer2', 'Tagged Layer2 cond2', None

        if TaggedProcessor._re_l3_1.search(text):
            return 'Layer3', 'Tagged Layer3 cond3 VCL+N结构', None
        if TaggedProcessor._re_l3_2.search(text):
            return 'Layer3', 'Tagged Layer3 cond1', None
        if TaggedProcessor._re_l3_3.search(text):
            return 'Layer3', 'Tagged Layer3 cond2', None

        if TaggedProcessor._re_l4_1.search(text):
            return 'Layer4', 'Tagged 所+在', None
        return 'Layer4', 'Tagged default', None


# ====================
# Classify missing and bucket by layer
# ====================
rows_by_layer = {"Layer1": [], "Layer2": [], "Layer3": [], "Layer4": []}

for _, r in missing.iterrows():
    record = {
        "file": str(r["file"]).strip(),
        "context": str(r["context"]),
        "zai": str(r["zai"]),
        "next": str(r["next"]),
    }
    layer, cond, cond_sub = TaggedProcessor.classify(record)
    rows_by_layer[layer].append({
        "file": record["file"],
        "ID": "",  # cannot recover original ID; keep empty
        "context": record["context"],
        "zai": record["zai"],
        "next": record["next"],
        "condition": cond,
        "cond_sub": cond_sub or ""
    })

# ====================
# Append into layer files (avoid exact dup on c/z/n/cond)
# ====================
added_total = 0
report = []

for layer in ["Layer1", "Layer2", "Layer3", "Layer4"]:
    add_df = pd.DataFrame(rows_by_layer[layer],
                          columns=["file","ID","context","zai","next","condition","cond_sub"]).fillna("")
    out_path = OUT_DIR / f"MC_zai_{layer.lower()}.txt"

    if out_path.exists():
        cur = pd.read_csv(out_path, sep="\t", dtype=str).fillna("")
        # Standardize columns (fill in empty columns if cond_sub is missing in older files)
        for col in ["file","ID","context","zai","next","condition","cond_sub"]:
            if col not in cur.columns:
                cur[col] = ""
        cur = cur[["file","ID","context","zai","next","condition","cond_sub"]]
    else:
        cur = pd.DataFrame(columns=["file","ID","context","zai","next","condition","cond_sub"])

    before = len(cur)
    merged = pd.concat([cur, add_df], ignore_index=True)

    # Avoid complete duplication: Remove duplicates by (context, zai, next, condition) (excluding cond_sub)
    merged = merged.drop_duplicates(subset=["context","zai","next","condition"], keep="last")

    after = len(merged)
    added = max(0, after - before)
    added_total += added
    report.append((layer, added, after))

    merged.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

# ====================
# Summary + optional log
# ====================
print("=== Backfill from missing — summary ===")
for layer, added, final_rows in report:
    print(f"{layer}: +{added} appended, now {final_rows} rows")
print(f"TOTAL appended: {added_total}")

log_path = OUT_DIR / "MC_backfilled_from_missing_log.tsv"
all_added = []
for layer in ["Layer1","Layer2","Layer3","Layer4"]:
    for d in rows_by_layer[layer]:
        all_added.append({**d, "layer": layer})
if all_added:
    pd.DataFrame(all_added).to_csv(log_path, sep="\t", index=False, encoding="utf-8")
    print("Log saved:", log_path)
else:
    print("No rows to append; nothing changed.")

# -*- coding: utf-8 -*-
# ============================================================
# EMoC Improved Full Version: Preserves final deduplication + Initial (3/4/5) alignment + Auditing and Reporting
# ============================================================

import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

class Config:
    ID_COL = 1
    TARGET_COL = 4
    CONTEXT_COL = 3
    NEXT_COL = 5

    # Deduplicated keys: 'text_cond' (strong across files/IDs) or 'file_id_text_cond' (conservative)
    DEDUP_KEY_MODE = 'text_cond'
    TOP_N = 100
    PRESERVE_INITIAL_MULTIPLICITY = True  # Whether to preserve multiple occurrences in the initial state after alignment

    @staticmethod
    def ordinal_suffix(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

# Path
BASE_DIR   = Path("/data/dataset/groups by century")
TAGGED_DIR = BASE_DIR / "Tagged_zai"  # If the EMoC directory is different, please modify it.
CENTURIES  = [f"{Config.ordinal_suffix(i)} century" for i in range(10, 18)]
OUT_DIR    = Path("/data/dataset/layered data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# EMoC initial table (for 3/4/5 alignment only)
INIT_XLSX_EMOC = Path("/data/dataset/initial collection/TaggedZAI_EMoC.xlsx")

# ========= DROP-IN REPLACEMENT =========
class TaggedProcessor:
    """
    Layer 1 (ZAI) Subdivision (Traditional Chinese) + Four-layer Judgment:

    Trigger words: positive / square / value / suitable; adverbs: only / convenient / again / still / all / all / and / will; negation: not / not / do not / not / not / not / not / not

    Assume the input is "character (tag)".
    """
    TOKEN   = r'[^\s()]+'
    TOK     = rf'{TOKEN}\([^)]*\)'
    VERB    = rf'{TOKEN}\(V[^)]*\)'
    TRIGGER = re.compile(r'[正方值適]\([^)]*\)')
    LIGHT   = r'(?:[只便又還皆都且將]\([^)]*\)\s*)?'
    ZAI_ANY = r'在\([^)]*\)'
    ZAI_V   = r'在\(V[^)]*\)'
    NEGCHR  = r'[不弗勿毋未非莫沒無]'

    # —— Layer1 Subdivision (Strong → Weak) # >>> CHANGED: Add Sub-tag Rule
    PAT_ZHENGZAI_V   = re.compile(rf'正在\([^)]*\)\s*{VERB}')
    PAT_ZAI_V_ZHE    = re.compile(rf'{ZAI_ANY}\s*{VERB}(?:\s*{TOK}){{0,3}}(?:著|呢)')
    PAT_TRIG_NEG     = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{NEGCHR}{TOK}(?:\s*{TOK}){{0,3}}')
    PAT_TRIG_V       = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_V}\s*{VERB}')
    PAT_TRIG_V_LOOSE = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{TOK}\s*{VERB}')

    @staticmethod
    def classify(record):
        """
        Return (layer, condition, cond_sub)  # >>> CHANGED: The third return value is cond_sub
        """
        ctx, zai, nxt = record['context'], record['zai'], record['next']
        text = f"{ctx}{zai}{nxt}"

        # ===== Layer1 Subdivision ===== # >>> CHANGED
        if TaggedProcessor.PAT_ZHENGZAI_V.search(text):
            return 'Layer1', 'Tagged 正在+V', 'L1_zhengzai_V'
        if TaggedProcessor.PAT_ZAI_V_ZHE.search(text):
            return 'Layer1', 'Tagged 在+动词+著/呢', 'L1_zai_V_zhe'
        if TaggedProcessor.PAT_TRIG_NEG.search(text):
            return 'Layer1', 'Tagged 触发+在+否定', 'L1_trigZai_neg'
        if TaggedProcessor.PAT_TRIG_V.search(text):
            return 'Layer1', 'Tagged 触发+在(动词)+动词', 'L1_trigZai_V'
        if TaggedProcessor.PAT_TRIG_V_LOOSE.search(text):
            return 'Layer1', 'Tagged 触发+在+1词+动词', 'L1_trigZai_V_loose'
        if re.search(r'在\(T\)', zai):
            return 'Layer1', 'Tagged 在(T)', 'L1_other'

        # ===== Layer2 =====
        if re.search(r'在\(P\)', zai):
            return 'Layer2', 'Tagged Layer2 cond3', None
        if re.search(r"\w+\(V[^)]*\)在(?!\w+\(V\))\s*\w+\(N[cdg][^)]*\)", text):
            return 'Layer2', 'Tagged Layer2 cond1', None
        if re.search(r"\w+\(V[^)]*\)在\w+\(Na\)\s+\w+\(Na\)", text):
            return 'Layer2', 'Tagged Layer2 cond2', None

        # ===== Layer3 =====
        if re.search(r'\w+\(VCL\)(?:\s+\w+\(N[cdg][^)]*\)|\s+\w+\(Na\)\s+\w+\(Ng\))', text):
            return 'Layer3', 'Tagged Layer3 cond3 VCL+N结构', None
        if re.search(r"\w*\(N[^)]*\)在\(V[^)]*\)\w+\(N[^)]*\)", text):
            return 'Layer3', 'Tagged Layer3 cond1', None
        if re.search(r"\w*\(N[^)]*\)在\(V[^)]*\)(此|是)", text):
            return 'Layer3', 'Tagged Layer3 cond2', None

        # ===== Layer4 =====
        if re.search(r'\w+\(T8\)\s+在\([^)]*\)(?:\s+\w+\([^)]*\))*\s+\w+\(V[^)]*\)', text):
            return 'Layer4', 'Tagged 所+在', None
        return 'Layer4', 'Tagged default', None

class EnhancedProcessingSystem:
    def __init__(self):
        self.store = dict()  # Unique key -> (layer, entry)
        self.results_by_layer = defaultdict(list)
        self.overwritten = []
        self.infile_dedup_count = 0
        self.scanned_rows = 0

    @staticmethod
    def _make_key(record, condition):
        # >>> Important: Deduplicat keys "excluding cond_sub" # >>> CHANGED: Emphasis but logic remains the same
        ctx = (record['context'] or "").strip()
        zai = (record['zai'] or "").strip()
        nxt = (record['next'] or "").strip()
        cond = (condition or "").strip()
        if Config.DEDUP_KEY_MODE == 'file_id_text_cond':
            file_ = (record['file'] or "").strip()
            id_   = str(record['id']).strip()
            return ('file_id_text_cond', file_, id_, ctx, zai, nxt, cond)
        return ('text_cond', ctx, zai, nxt, cond)

    def _upsert(self, layer, record, condition, cond_sub=None):
        # >>> CHANGED: _upsert Added cond_sub storage
        key = self._make_key(record, condition)
        entry = (record['file'], record['id'], record['context'],
                 record['zai'], record['next'], condition, cond_sub)  # >>> CHANGED
        if key in self.store:
            prev_layer, prev_entry = self.store[key]
            self.overwritten.append((*prev_entry, prev_layer, 'OVERWRITTEN_KEEP_LAST'))
        self.store[key] = (layer, entry)

    def process_files(self, base_dir: Path):
        for century in CENTURIES:
            dir_path = base_dir / century
            if not dir_path.exists():
                print(f"Directory does not exist: {dir_path}")
                continue
            for file in dir_path.glob('*.xlsx'):
                df = pd.read_excel(file, header=None, engine='openpyxl')
                self.scanned_rows += len(df)

                # Deduplication within a file (context, zai, next) Preserve last occurrence
                before = len(df)
                df = df.drop_duplicates(
                    subset=[Config.CONTEXT_COL-1, Config.TARGET_COL-1, Config.NEXT_COL-1],
                    keep='last'
                )
                removed = before - len(df)
                if removed > 0:
                    self.infile_dedup_count += removed
                    print(f"[dedup-file] {file.name}: {removed} dup rows removed inside file (keep=last)")

                for _, row in df.iterrows():
                    record = {
                        'file': Path(file.name).stem,
                        'id': row.iloc[Config.ID_COL-1],
                        'context': str(row.iloc[Config.CONTEXT_COL-1]),
                        'zai': str(row.iloc[Config.TARGET_COL-1]),
                        'next': str(row.iloc[Config.NEXT_COL-1])
                    }
                    layer, cond, cond_sub = TaggedProcessor.classify(record)  # >>> CHANGED
                    self._upsert(layer, record, cond, cond_sub)              # >>> CHANGED

    def _collect_all_kept(self) -> pd.DataFrame:
        rows = []
        for layer, entry in self.store.values():
            rows.append((*entry, layer))
        cols = ['file','ID','context','zai','next','condition','cond_sub','layer']  # >>> CHANGED
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def _filter_by_initial(self, df: pd.DataFrame) -> pd.DataFrame:
        init_raw = pd.read_excel(INIT_XLSX_EMOC, header=None, dtype=str, engine='openpyxl', usecols=[2,3,4])
        init_raw.columns = ['context','zai','next']
        init_raw = init_raw.fillna("").astype(str)
        init_keys = set(map(tuple, init_raw[['context','zai','next']].itertuples(index=False, name=None)))

        mask = df.apply(lambda r: (r['context'], r['zai'], r['next']) in init_keys, axis=1)
        df2 = df.loc[mask].copy()

        if not Config.PRESERVE_INITIAL_MULTIPLICITY:
            df2 = df2.drop_duplicates(subset=['context','zai','next'], keep='last')
        return df2

    def export_results(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        all_kept = self._collect_all_kept()
        all_kept = self._filter_by_initial(all_kept)

        # >>> Important: The final deduplication also does not include cond_sub
        # >>> CHANGED: This is only for emphasis; the logic remains the same.
        subset = (['file','ID','context','zai','next','condition']
                  if Config.DEDUP_KEY_MODE == 'file_id_text_cond'
                  else ['context','zai','next','condition'])
        all_kept = all_kept.drop_duplicates(subset=subset, keep='last')

        total_written = 0
        for layer in ['Layer1','Layer2','Layer3','Layer4']:
            dfl = all_kept[all_kept['layer']==layer].copy()
            dfl = dfl[['file','ID','context','zai','next','condition','cond_sub']]  # >>> CHANGED
            total_written += len(dfl)
            dfl.to_csv(output_dir / f"EMoC_zai_{layer.lower()}.txt", sep='\t', index=False, encoding='utf-8')

        if self.overwritten or self.infile_dedup_count > 0:
            dup_cols = ['file','ID','context','zai','next','condition','cond_sub','pred_layer','reason']
            pd.DataFrame(self.overwritten, columns=dup_cols).to_csv(
                output_dir / "EMoC_duplicates_log.tsv", sep='\t', index=False, encoding='utf-8'
            )
            print(f"[log] EMoC_duplicates_log.tsv written with {len(self.overwritten)} overwritten rows "
                  f"+ {self.infile_dedup_count} removed inside files")

        self.results_by_layer.clear()
        for layer in ['Layer1','Layer2','Layer3','Layer4']:
            self.results_by_layer[layer] = all_kept[all_kept['layer']==layer][
                ['file','ID','context','zai','next','condition','cond_sub']  # >>> CHANGED
            ].values.tolist()

        return total_written

    def print_summary(self):
        print("\n=== EMoC Summary (keep-last, after initial-filter) ===")
        total_kept = 0
        for layer in ['Layer1','Layer2','Layer3','Layer4']:
            count = len(self.results_by_layer.get(layer, []))
            print(f"{layer}: {count} rows")
            total_kept += count
        print(f"Total kept (exported): {total_kept}")
        print(f"Scanned raw rows                 : {self.scanned_rows}")
        print(f"Removed inside files (keep=last) : {self.infile_dedup_count}")
        print(f"Overwritten by key (keep-last)   : {len(self.overwritten)}")

        all_rows = []
        for layer in ['Layer1','Layer2','Layer3','Layer4']:
            for r in self.results_by_layer.get(layer, []):
                all_rows.append((*r, layer))
        if all_rows:
            df = pd.DataFrame(all_rows, columns=['file','ID','context','zai','next','condition','cond_sub','layer'])  # >>> CHANGED
            n_ctn   = df.drop_duplicates(subset=['context','zai','next']).shape[0]
            n_ctnc  = df.drop_duplicates(subset=['context','zai','next','condition']).shape[0]
            print("\n--- Audit ---")
            print(f"Unique (context,zai,next)        : {n_ctn}")
            print(f"Unique (context,zai,next,cond)   : {n_ctnc}")
        else:
            print("\n--- Audit ---\n(no rows kept)")

    def export_count_report(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        kept_rows = sum(len(v) for v in self.results_by_layer.values())
        overview = {
            "scanned_raw_rows": self.scanned_rows,
            "removed_inside_files_keep_last": self.infile_dedup_count,
            "overwritten_keep_last": len(self.overwritten),
            "kept_rows_after_initial_filter": kept_rows,
        }

        all_rows = []
        for layer in ['Layer1','Layer2','Layer3','Layer4']:
            for r in self.results_by_layer.get(layer, []):
                all_rows.append((*r, layer))
        if all_rows:
            df = pd.DataFrame(all_rows, columns=['file','ID','context','zai','next','condition','cond_sub','layer'])  # >>> CHANGED
            overview["unique_ctn"]  = df.drop_duplicates(subset=['context','zai','next']).shape[0]
            overview["unique_ctnc"] = df.drop_duplicates(subset=['context','zai','next','condition']).shape[0]

        pd.DataFrame([overview]).to_csv(OUT_DIR / "EMoC_counts_overview.tsv", sep="\t", index=False)

        per_layer = []
        for layer in ['Layer1','Layer2','Layer3','Layer4']:
            data = self.results_by_layer.get(layer, [])
            if not data:
                per_layer.append({"layer": layer, "kept": 0, "unique_ctn": 0, "unique_ctnc": 0})
            else:
                dfl = pd.DataFrame(data, columns=['file','ID','context','zai','next','condition','cond_sub'])  # >>> CHANGED
                per_layer.append({
                    "layer": layer,
                    "kept": len(dfl),
                    "unique_ctn":  dfl.drop_duplicates(subset=['context','zai','next']).shape[0],
                    "unique_ctnc": dfl.drop_duplicates(subset=['context','zai','next','condition']).shape[0],
                })
        pd.DataFrame(per_layer).to_csv(OUT_DIR / "EMoC_counts_per_layer.tsv", sep="\t", index=False)

def main():
    system = EnhancedProcessingSystem()
    system.process_files(TAGGED_DIR)
    written = system.export_results(OUT_DIR)
    system.print_summary()
    system.export_count_report(OUT_DIR)
    print(f"\n[done] Exported to: {OUT_DIR} (total rows written across layers: {written})")
    print(f"[dedup mode] {Config.DEDUP_KEY_MODE}  "
          f"({'跨文件/ID强力去重' if Config.DEDUP_KEY_MODE=='text_cond' else '包含file/ID的保守去重'})")

if __name__ == '__main__':
    main()

INIT_XLSX = "/data/dataset/initial collection/TaggedZAI_EMoC.xlsx"
OUT_DIR   = Path("/data/dataset/layered data")

# 1) Read the initial (0=sequence number (can be ignored), 1=file, 2=context, 3=zai, 4=next)
init_raw = pd.read_excel(INIT_XLSX, header=None, dtype=str, engine="openpyxl")
sub = init_raw.iloc[:, [1,2,3,4]].fillna("")
sub.columns = ["file","context","zai","next"]
init_unique = sub.drop_duplicates(subset=["context","zai","next"], keep="last")

# 2) Read the final file (EMoC_zai_layer*.txt) and use (context,zai,next) as the unique key.
frames = []
for layer in ["layer1","layer2","layer3","layer4"]:
    p = OUT_DIR / f"EMoC_zai_{layer}.txt"
    if not p.exists():
        continue
    df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    frames.append(df[["context","zai","next","file"]])

if not frames:
    raise SystemExit("[ERR] No EMoC_zai_layer*.txt found in OUT_DIR")

kept = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["context","zai","next"], keep="last")

#3) Initially existed but ultimately not retained.
missing = init_unique.merge(
    kept[["context","zai","next"]],
    on=["context","zai","next"],
    how="left",
    indicator=True
)
missing = missing[missing["_merge"]=="left_only"].drop(columns=["_merge"]).copy()

print(f"[DIFF-EMoC] initial unique = {len(init_unique)}")
print(f"[DIFF-EMoC] kept (layered) unique = {kept[['context','zai','next']].drop_duplicates().shape[0]}")
print(f"[DIFF-EMoC] missing (initial but not kept) = {len(missing)}\n")

for r in missing[["file","context","zai","next"]].itertuples(index=False):
    print("\t".join(map(str, r)))

out_path = OUT_DIR / "EMoC_missing_from_layered.tsv"
missing[["file","context","zai","next"]].to_csv(out_path, sep="\t", index=False, encoding="utf-8")
print(f"\n[Saved] {out_path}")

# ============================
# Configuration
# ============================
import re
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/data/dataset/layered data")
MISSING_PATH = OUT_DIR / "EMoC_missing_from_layered.tsv"

# Deduplication strategy: True = by (context, zai, next, condition); False = by (context, zai, next)
DEDUP_BY_FOUR_KEYS = True

# ============================
# Classifier (same as the main program, returns cond_sub)
# ============================
class TaggedProcessor:
    TOKEN   = r'[^\s()]+'
    TOK     = rf'{TOKEN}\([^)]*\)'
    VERB    = rf'{TOKEN}\(V[^)]*\)'
    TRIGGER = re.compile(r'[正方值適]\([^)]*\)')
    LIGHT   = r'(?:[只便又還皆都且將]\([^)]*\)\s*)?'
    ZAI_ANY = r'在\([^)]*\)'
    ZAI_V   = r'在\(V[^)]*\)'
    NEGCHR  = r'[不弗勿毋未非莫沒無]'

    PAT_ZHENGZAI_V   = re.compile(rf'正在\([^)]*\)\s*{VERB}')
    PAT_ZAI_V_ZHE    = re.compile(rf'{ZAI_ANY}\s*{VERB}(?:\s*{TOK}){{0,3}}(?:著|呢)')
    PAT_TRIG_NEG     = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{NEGCHR}{TOK}(?:\s*{TOK}){{0,3}}')
    PAT_TRIG_V       = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_V}\s*{VERB}')
    PAT_TRIG_V_LOOSE = re.compile(rf'{TRIGGER.pattern}\s*{LIGHT}{ZAI_ANY}\s*{TOK}\s*{VERB}')

    @staticmethod
    def classify(record):
        ctx, zai, nxt = record['context'], record['zai'], record['next']
        text = f"{ctx}{zai}{nxt}"

        if TaggedProcessor.PAT_ZHENGZAI_V.search(text):
            return 'Layer1', 'Tagged 正在+V', 'L1_zhengzai_V'
        if TaggedProcessor.PAT_ZAI_V_ZHE.search(text):
            return 'Layer1', 'Tagged 在+动词+著/呢', 'L1_zai_V_zhe'
        if TaggedProcessor.PAT_TRIG_NEG.search(text):
            return 'Layer1', 'Tagged 触发+在+否定', 'L1_trigZai_neg'
        if TaggedProcessor.PAT_TRIG_V.search(text):
            return 'Layer1', 'Tagged 触发+在(动词)+动词', 'L1_trigZai_V'
        if TaggedProcessor.PAT_TRIG_V_LOOSE.search(text):
            return 'Layer1', 'Tagged 触发+在+1词+动词', 'L1_trigZai_V_loose'
        if re.search(r'在\(T\)', zai):
            return 'Layer1', 'Tagged 在(T)', 'L1_other'

        if re.search(r'在\(P\)', zai):
            return 'Layer2', 'Tagged Layer2 cond3', None
        if re.search(r"\w+\(V[^)]*\)在(?!\w+\(V\))\s*\w+\(N[cdg][^)]*\)", text):
            return 'Layer2', 'Tagged Layer2 cond1', None
        if re.search(r"\w+\(V[^)]*\)在\w+\(Na\)\s+\w+\(Na\)", text):
            return 'Layer2', 'Tagged Layer2 cond2', None

        if re.search(r'\w+\(VCL\)(?:\s+\w+\(N[cdg][^)]*\)|\s+\w+\(Na\)\s+\w+\(Ng\))', text):
            return 'Layer3', 'Tagged Layer3 cond3 VCL+N结构', None
        if re.search(r"\w*\(N[^)]*\)在\(V[^)]*\)\w+\(N[^)]*\)", text):
            return 'Layer3', 'Tagged Layer3 cond1', None
        if re.search(r"\w*\(N[^)]*\)在\(V[^)]*\)(此|是)", text):
            return 'Layer3', 'Tagged Layer3 cond2', None

        if re.search(r'\w+\(T8\)\s+在\([^)]*\)(?:\s+\w+\([^)]*\))*\s+\w+\(V[^)]*\)', text):
            return 'Layer4', 'Tagged 所+在', None
        return 'Layer4', 'Tagged default', None

# ============================
# Read missing list
# ============================
if not MISSING_PATH.exists():
    raise SystemExit(f"[ERR] Missing file not found: {MISSING_PATH}")

missing = pd.read_csv(MISSING_PATH, sep="\t", dtype=str).fillna("")
need_cols = ["file", "context", "zai", "next"]
for c in need_cols:
    if c not in missing.columns:
        raise SystemExit(f"[ERR] Column '{c}' is required in {MISSING_PATH}")
missing = missing[need_cols].copy()

print(f"[load] missing rows: {len(missing)}")

# ============================
# 判层 & 分桶（含 cond_sub）
# ============================
rows_by_layer = {"Layer1": [], "Layer2": [], "Layer3": [], "Layer4": []}

for _, r in missing.iterrows():
    record = {
        "file": str(r["file"]).strip(),
        "context": str(r["context"]),
        "zai": str(r["zai"]),
        "next": str(r["next"]),
    }
    layer, cond, cond_sub = TaggedProcessor.classify(record)  # >>> CHANGED
    rows_by_layer[layer].append({
        "file": record["file"],
        "ID": "",  # ID, leave blank.
        "context": record["context"],
        "zai": record["zai"],
        "next": record["next"],
        "condition": cond,
        "cond_sub": cond_sub or ""                        # >>> CHANGED
    })

for k in rows_by_layer:
    print(f"[bucket] {k} to add: {len(rows_by_layer[k])}")

# ============================
# Add and write back four EMoC layer files (including cond_sub; deduplication excludes it)
# ============================
added_total = 0
report = []

# >>> Still: Do not include cond_sub in the deduplication key.
dedup_subset = ["context", "zai", "next", "condition"] if DEDUP_BY_FOUR_KEYS else ["context", "zai", "next"]

for layer in ["Layer1", "Layer2", "Layer3", "Layer4"]:
    add_df = pd.DataFrame(rows_by_layer[layer],
                          columns=["file","ID","context","zai","next","condition","cond_sub"]).fillna("")  # >>> CHANGED
    out_path = OUT_DIR / f"EMoC_zai_{layer.lower()}.txt"

    if out_path.exists():
        cur = pd.read_csv(out_path, sep="\t", dtype=str).fillna("")
        # Standardize column order (add columns if missing from old files)
        for col in ["file","ID","context","zai","next","condition","cond_sub"]:  # >>> CHANGED
            if col not in cur.columns:
                cur[col] = ""
        cur = cur[["file","ID","context","zai","next","condition","cond_sub"]]   # >>> CHANGED
    else:
        cur = pd.DataFrame(columns=["file","ID","context","zai","next","condition","cond_sub"])  # >>> CHANGED

    before = len(cur)
    merged = pd.concat([cur, add_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=dedup_subset, keep="last")  # still does not include cond_sub
    after = len(merged)

    added_total += max(0, after - before)
    report.append((layer, max(0, after - before), after))

    merged.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

print("\n=== Backfill from EMoC_missing — summary ===")
for layer, added, final_rows in report:
    print(f"{layer}: +{added} appended, now {final_rows} rows")
print(f"TOTAL appended: {added_total}")
print(f"[dedup] subset = {dedup_subset}")
print("[done] backfill complete.")