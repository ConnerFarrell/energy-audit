"""Step 2 (Tier-2 rescoring) comparison. Run from the repo root:
    python rerun/step2/compare_step2.py rerun/step2/kaggle_out

1. Per clip: p0a, p0a_n, lufs must be bit-identical to data/scored_full_v2.csv
   (float equality after CSV round-trip; both files are written by pandas).
2. CLAP (clap, clap_n): the new run is seeded, so it is the new baseline.
   Reports, per model: mean old/new, mean shift, max |clip change|, max
   |cell-mean change| (model x phrasing x level, the README's drift metric),
   and Spearman rho between old and new clip values.
3. Banks re-made in the same run (s6_*, s7_*): every cell that does not depend
   on CLAP is compared to data/ with Step 1's strict rule; CLAP-dependent cells
   are listed as the new values.
Writes rerun/step2/clip_comparison.csv, clap_shift_by_model.csv,
clap_cell_means.csv, bank_comparison.csv.
"""
import sys, os
from decimal import Decimal
import numpy as np, pandas as pd
from scipy.stats import spearmanr

NEW = sys.argv[1] if len(sys.argv) > 1 else "rerun/step2/kaggle_out"
OUT = "rerun/step2"
old = pd.read_csv("data/scored_full_v2.csv")
new = pd.read_csv(f"{NEW}/scored_full_v2.csv")
assert len(new) == 1500 and new.file.is_unique, f"new run: {len(new)} rows"
assert set(old.file) == set(new.file), "clip sets differ"
m = old.merge(new, on="file", suffixes=("_old", "_new"))
for c in ["model", "prompt_id", "seed", "genre_slug", "level", "phrasing"]:
    assert (m[f"{c}_old"] == m[f"{c}_new"]).all(), f"metadata column {c} differs"
m = m.rename(columns={"model_old": "model", "phrasing_old": "phrasing", "level_old": "level"})

print("=== 1. Exact columns (per clip, bit-identical) ===")
exact = {}
for c in ["p0a", "p0a_n", "lufs"]:
    eq = (m[f"{c}_old"] == m[f"{c}_new"]) | (m[f"{c}_old"].isna() & m[f"{c}_new"].isna())
    exact[c] = int(eq.sum())
    d = (m[f"{c}_old"] - m[f"{c}_new"]).abs()
    print(f"  {c:6s} identical {eq.sum()}/1500   max|diff| {d.max():.3g}")
    if not eq.all():
        print(m.loc[~eq, ["file", f"{c}_old", f"{c}_new"]].head(10).to_string(index=False))
m[["file", "model"] + [f"{c}_{s}" for c in ["p0a", "p0a_n", "lufs", "clap", "clap_n"] for s in ("old", "new")]] \
    .to_csv(f"{OUT}/clip_comparison.csv", index=False)

print("\n=== 2. CLAP: new seeded baseline vs old bank ===")
rows = []
for col in ["clap", "clap_n"]:
    for mdl, g in m.groupby("model"):
        d = g[f"{col}_new"] - g[f"{col}_old"]
        cm = g.groupby(["phrasing", "level"])[[f"{col}_old", f"{col}_new"]].mean()
        rows.append(dict(column=col, model=mdl, mean_old=g[f"{col}_old"].mean(), mean_new=g[f"{col}_new"].mean(),
                         mean_shift=d.mean(), max_abs_clip_change=d.abs().max(),
                         max_abs_cell_mean_change=(cm[f"{col}_new"] - cm[f"{col}_old"]).abs().max(),
                         spearman_old_new=spearmanr(g[f"{col}_old"], g[f"{col}_new"])[0]))
shift = pd.DataFrame(rows); shift.to_csv(f"{OUT}/clap_shift_by_model.csv", index=False)
print(shift.round(4).to_string(index=False))
cells = m.groupby(["model", "phrasing", "level"])[["clap_old", "clap_new", "clap_n_old", "clap_n_new"]].mean()
cells.to_csv(f"{OUT}/clap_cell_means.csv")
print(f"  overall max |cell-mean change|: clap {(cells.clap_new - cells.clap_old).abs().max():.4f}, "
      f"clap_n {(cells.clap_n_new - cells.clap_n_old).abs().max():.4f}  (README states <= 0.033)")

print("\n=== 3. Banks from the same run vs data/ (strict rule) ===")
def decimals(s):
    try: return max(0, -Decimal(str(s).strip()).as_tuple().exponent)
    except Exception: return None
CLAP_DEP = lambda f, key, fld: (("clap" in str(key)) or (f == "s6_level_means.csv" and fld in ("clap", "clap_n")))
brow = []
def cmpcell(f, key, fld, s, n):
    if CLAP_DEP(f, key, fld):
        brow.append((f, key, fld, s, n, "CLAP (new baseline)")); return
    k = decimals(s)
    if k is None or decimals(n) is None:
        brow.append((f, key, fld, s, n, "MATCH" if str(s) == str(n) else "MISMATCH")); return
    ok = abs(float(n) - float(s)) <= 0.5 * 10 ** (-k) * (1 + 1e-9)
    brow.append((f, key, fld, s, n, "MATCH" if ok else "MISMATCH"))
S = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
for f in ["s6_bootstrap_cis.csv", "s7_bootstrap_cis.csv"]:
    if not os.path.exists(f"{NEW}/{f}"): print("  missing", f); continue
    o, n = S(f"data/{f}"), S(f"{NEW}/{f}").set_index(["analysis", "model", "scope"])
    for _, r in o.iterrows():
        k = (r.analysis, r.model, r.scope)
        for fld in ["point", "lo", "hi"]:
            cmpcell(f, "|".join(k), fld, r[fld], n.loc[k, fld] if k in n.index else "MISSING")
for f, keys in [("s7_leaderboard.csv", ["model"]), ("s6_level_means.csv", ["model", "phrasing", "level"]),
                ("s6_edge_cells.csv", ["model", "genre_slug"]), ("s7_skew_L1L9.csv", ["model", "genre_slug", "level", "seed"]),
                ("s6_feature_decomp.csv", ["model"]), ("s6_raw_ridge_level_means.csv", ["model"])]:
    if not os.path.exists(f"{NEW}/{f}"): print("  missing", f); continue
    o, n = S(f"data/{f}"), S(f"{NEW}/{f}").set_index(keys)
    for _, r in o.iterrows():
        k = tuple(r[c] for c in keys) if len(keys) > 1 else r[keys[0]]
        for fld in [c for c in o.columns if c not in keys]:
            v = n.loc[k, fld] if (k in n.index and fld in n.columns) else "MISSING"
            cmpcell(f, "|".join(k) if isinstance(k, tuple) else k, fld, r[fld], v)
B = pd.DataFrame(brow, columns=["file", "key", "field", "stored", "new", "status"])
B.to_csv(f"{OUT}/bank_comparison.csv", index=False)
print(B.groupby(["file", "status"]).size().unstack(fill_value=0).to_string())
mm = B[B.status == "MISMATCH"]
if len(mm): print(mm.head(30).to_string(index=False))
print("\nSUMMARY exact:", exact, "| bank non-CLAP mismatches:", len(mm))
