"""Step 1, attempt #2: compare every banked data/*.csv cell with the table the
rebuilt notebooks/repro.ipynb regenerated (rerun/step1/attempt2/out/tables/).

Same rule as attempt #1: a cell matches when |regenerated - stored| <= half a
unit in the stored value's last decimal place. Non-numeric cells must be equal
as strings. Also checks row order and row count per file.
s7_clap_axis.csv is a Tier-2 input (per-clip CLAP sims need audio); as in
attempt #1 it is checked by its internal identity axis = sim_hi - sim_lo.
Run from the repo root.
"""
import sys
from decimal import Decimal
import numpy as np, pandas as pd

NEW = sys.argv[1] if len(sys.argv) > 1 else "rerun/step1/attempt2/out/tables"
OUTCSV = sys.argv[2] if len(sys.argv) > 2 else "rerun/step1/attempt2/cell_comparison.csv"

def decimals(s):
    try: d = Decimal(str(s).strip())
    except Exception: return None
    return max(0, -d.as_tuple().exponent)

rows = []
def cmp(file, key, field, stored, new):
    if new is None or (isinstance(new, float) and np.isnan(new) and str(stored) != "nan"):
        rows.append((file, key, field, stored, "", "MISSING", "MISSING")); return
    k = decimals(stored)
    if k is None or decimals(new) is None:
        st = "MATCH" if str(stored) == str(new) else "MISMATCH"
        rows.append((file, key, field, stored, new, st, st)); return
    s, n = float(stored), float(new); d = abs(n - s); half = 0.5 * 10 ** (-k)
    # strict rule: regenerated value rounds to the stored digits (no slack)
    st = "MATCH" if d <= half * (1 + 1e-9) else "MISMATCH"
    if st == "MATCH" and d > 1e-12 and abs(d - half) <= 1e-6 * half: st = "MATCH (rounding tie)"
    if st == "MISMATCH" and d <= 4 * np.spacing(abs(s)): st = "MISMATCH (<=4 ULP)"
    a1 = "MATCH" if d <= half + 1e-9 else "MISMATCH"   # attempt-1 rule (tol = half + 1e-9)
    rows.append((file, key, field, stored, repr(n) if isinstance(n, float) else n, st, a1)); return

S = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
order = {}

# long-format CI banks
for f in ["s6_bootstrap_cis.csv", "s7_bootstrap_cis.csv"]:
    old, new = S(f"data/{f}"), pd.read_csv(f"{NEW}/{f}", keep_default_na=False)
    order[f] = (list(zip(old.analysis, old.model, old.scope)) == list(zip(new.analysis, new.model, new.scope)),
                len(old), len(new))
    nk = new.set_index(["analysis", "model", "scope"])
    for _, r in old.iterrows():
        k = (r.analysis, r.model, r.scope)
        for fld in ["point", "lo", "hi"]:
            cmp(f, "|".join(k), fld, r[fld], nk.loc[k, fld] if k in nk.index else None)
        cmp(f, "|".join(k), "note", r.note, nk.loc[k, "note"] if k in nk.index else None)

# keyed wide tables
def wide(f, keys, fields=None, cast=None):
    old = S(f"data/{f}"); new = pd.read_csv(f"{NEW}/{f}", keep_default_na=False, dtype={k: str for k in keys})
    order[f] = (old[keys].values.tolist() == new[keys].astype(str).values.tolist(), len(old), len(new))
    nk = new.set_index(keys)
    flds = fields or [c for c in old.columns if c not in keys]
    for _, r in old.iterrows():
        k = tuple(r[c] for c in keys) if len(keys) > 1 else r[keys[0]]
        for fld in flds:
            v = nk.loc[k, fld] if (k in nk.index and fld in nk.columns) else None
            cmp(f, "|".join(k) if isinstance(k, tuple) else k, fld, r[fld], v)

wide("s7_leaderboard.csv", ["model"])
wide("s6_level_means.csv", ["model", "phrasing", "level"])
wide("s6_edge_cells.csv", ["model", "genre_slug"])
wide("s7_axis_level_means.csv", ["model", "phrasing", "level"])
wide("s7_skew_L1L9.csv", ["model", "genre_slug", "level", "seed"])
wide("s6_feature_decomp.csv", ["model"])
wide("s6_raw_ridge_level_means.csv", ["model"])

# s7_clap_axis.csv: input identity check (not regenerable at Tier 1)
ax = S("data/s7_clap_axis.csv")
for _, r in ax.iterrows():
    cmp("s7_clap_axis.csv", r.file, "axis", r.axis, float(r.sim_hi) - float(r.sim_lo))

res = pd.DataFrame(rows, columns=["file", "key", "field", "stored", "regenerated", "status", "status_attempt1_rule"])
# the 'note' column is text and was not counted in attempt #1's 2,707; report separately
cells = res[res.field != "note"]; notes = res[res.field == "note"]
cells.to_csv(OUTCSV, index=False)
st = cells.status
summ = cells.assign(match=st.str.startswith("MATCH"), mismatch=st.str.startswith("MISMATCH"),
                    ulp=st == "MISMATCH (<=4 ULP)", missing=st == "MISSING") \
            .groupby("file")[["match", "mismatch", "ulp", "missing"]].sum()
summ["cells"] = summ[["match", "mismatch", "missing"]].sum(axis=1)
print(summ.to_string()); print("TOTAL", int(summ.cells.sum()), "match", int(summ.match.sum()),
      "mismatch", int(summ.mismatch.sum()), "of which <=4 ULP", int(summ.ulp.sum()), "missing", int(summ.missing.sum()))
print("attempt-1 rule: match", int(cells.status_attempt1_rule.eq("MATCH").sum()), "/", len(cells))
print("ties:", int((st == "MATCH (rounding tie)").sum()))
print("note column (text):", (notes.status == "MATCH").sum(), "/", len(notes), "equal")
print("row order/count identical:", {k: v for k, v in order.items()})
mm = cells[st.str.startswith("MISMATCH") | (st == "MISSING")]
if len(mm): print(mm.to_string(index=False))
