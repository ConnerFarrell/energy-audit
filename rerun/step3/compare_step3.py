"""Step 3 comparisons. Run from the repo root:  python rerun/step3/compare_step3.py [OUT]

OUT (default rerun/step3/score_out) is the Kaggle output of energy-audit-step3-score, without audio:
the main-run tables at the top level and the 6-model re-execution in OUT/six/.

1. Pass rule on the 1,500 banked clips (fresh pinned-image run):
   Arousal Index (p0a, p0a_n) and LUFS exact per clip against data/scored_full_v2.csv,
   and every banked cell within 1e-12 relative of data/. CLAP-dependent cells are reported
   separately: the bank holds the old unseeded CLAP by decision (Step 2 §10.3). They are
   compared against Step 2's seeded run (rerun/step2/kaggle_out) instead.
2. Comparability: every five-model row the 6-model re-execution (OUT/six) produces must equal
   the main run's row exactly. Pooled rows (scope 'ALL') are expected to change and are listed.
Outputs go to rerun/step3/.
"""
import os, sys
import numpy as np, pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "rerun/step3/score_out"
SIX = f"{OUT}/six"
DST = "rerun/step3"
S2 = "rerun/step2/kaggle_out"
BANKED = ["mgm", "mgs", "sao", "sa3s", "ace"]

def within(s, n):
    """1e-12 relative rule; text cells must be equal."""
    if s == n: return True
    try: s, n = float(s), float(n)
    except ValueError: return False
    if np.isnan(s) and np.isnan(n): return True
    return abs(n - s) <= 1e-12 * abs(s) if s != 0 else abs(n) <= 1e-12

S = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
CLAP_DEP = lambda f, key, fld: ("clap" in str(key)) or (f == "s6_level_means.csv" and fld in ("clap", "clap_n"))

TABLES = [("s6_bootstrap_cis.csv", ["analysis", "model", "scope"]),
          ("s7_bootstrap_cis.csv", ["analysis", "model", "scope"]),
          ("s7_leaderboard.csv", ["model"]),
          ("s6_level_means.csv", ["model", "phrasing", "level"]),
          ("s6_edge_cells.csv", ["model", "genre_slug"]),
          ("s7_skew_L1L9.csv", ["model", "genre_slug", "level", "seed"]),
          ("s6_feature_decomp.csv", ["model"]),
          ("s6_raw_ridge_level_means.csv", ["model"]),
          ("features_verbal.csv", ["file"])]

def cell_rows(ref, new, f, keys, label):
    """Compare every cell of ref (all rows) against new; returns long rows."""
    out = []
    if not os.path.exists(new):
        return [(label, f, "", "", "", "", "MISSING FILE")]
    o, n = S(ref), S(new)
    n = n.drop_duplicates(keys).set_index(keys)
    for _, r in o.iterrows():
        k = tuple(r[c] for c in keys)
        kk = k if len(keys) > 1 else k[0]
        for fld in [c for c in o.columns if c not in keys]:
            v = n.loc[kk, fld] if (kk in n.index and fld in n.columns) else "MISSING"
            out.append((label, f, "|".join(k), fld, r[fld], v, "MATCH" if within(r[fld], v) else "MISMATCH"))
    return out

# ---------------- 1. pass rule on the 1,500 banked clips ----------------
print("=== 1. Pass rule: 1,500 banked clips, fresh pinned-image run ===")
old = pd.read_csv("data/scored_full_v2.csv"); new = pd.read_csv(f"{OUT}/scored_full_v2.csv")
m = old.merge(new, on="file", suffixes=("_old", "_new"))
assert len(m) == 1500, len(m)
exact = {c: int((m[f"{c}_old"] == m[f"{c}_new"]).sum()) for c in ["p0a", "p0a_n", "lufs"]}
print("  per clip, exact:", exact, "of 1500")

rows = []
for f, keys in TABLES:
    for r in cell_rows(f"data/{f}", f"{OUT}/{f}", f, keys, "main_vs_bank"):
        lab, ff, key, fld, s, n, st = r
        if st != "MISSING FILE" and n == "MISSING" and "_axis" in key:
            st = "NOT PRODUCED (clap-axis.ipynb)"
        elif st != "MISSING FILE" and CLAP_DEP(ff, key, fld):
            st = "CLAP: " + st
        rows.append((lab, ff, key, fld, s, n, st))
B = pd.DataFrame(rows, columns=["comparison", "file", "key", "field", "stored", "new", "status"])
nonclap = B[~B.status.str.startswith("CLAP") & ~B.status.str.startswith("NOT PRODUCED")]
print(f"  not produced by the scoring notebook (CLAP axis, clap-axis.ipynb): {int(B.status.str.startswith('NOT PRODUCED').sum())}")
print(f"  banked non-CLAP cells within 1e-12: {int((nonclap.status == 'MATCH').sum())}/{len(nonclap)}")
print(f"  CLAP-dependent cells vs old (unseeded) bank: {int((B.status == 'CLAP: MATCH').sum())}/"
      f"{int(B.status.str.startswith('CLAP').sum())} within 1e-12 (bank holds the old CLAP by decision)")
print(nonclap.groupby(["file", "status"]).size().unstack(fill_value=0).to_string())
if (nonclap.status != "MATCH").any():
    print(nonclap[nonclap.status != "MATCH"].head(30).to_string(index=False))

# CLAP cells against Step 2's seeded run (the new CLAP baseline)
crow = []
for f, keys in TABLES:
    if os.path.exists(f"{S2}/{f}"):
        for r in cell_rows(f"{S2}/{f}", f"{OUT}/{f}", f, keys, "main_vs_step2_seeded"):
            if CLAP_DEP(r[1], r[2], r[3]): crow.append(r)
C = pd.DataFrame(crow, columns=B.columns)
print(f"  CLAP-dependent cells vs Step 2 seeded run: {int((C.status == 'MATCH').sum())}/{len(C)} within 1e-12")
clip_c = pd.read_csv(f"{S2}/scored_full_v2.csv").merge(new, on="file", suffixes=("_s2", "_new"))
for c in ["clap", "clap_n"]:
    d = (clip_c[f"{c}_s2"] - clip_c[f"{c}_new"]).abs()
    print(f"  per clip {c} vs Step 2 seeded: {int((d == 0).sum())}/1500 identical, max |diff| {d.max():.3g}")
pd.concat([B, C]).to_csv(f"{DST}/bank_comparison.csv", index=False)

# ---------------- 2. comparability: six/ five-model rows == main run ----------------
print("\n=== 2. Comparability: 6-model re-execution vs main run (five banked models) ===")
srow = []
for f, keys in TABLES:
    if not os.path.exists(f"{SIX}/{f}"):
        srow.append(("six_vs_main", f, "", "", "", "", "MISSING FILE")); continue
    o = S(f"{OUT}/{f}"); n = S(f"{SIX}/{f}")
    if "model" in keys:
        o = o[o.model.isin(BANKED)]
    elif f == "features_verbal.csv":
        o = o[~o.file.str.contains("mrt2__")]
    tmp = f"/tmp/_cmp_{f}"; o.to_csv(tmp, index=False)
    srow += cell_rows(tmp, f"{SIX}/{f}", f, keys, "six_vs_main")
X = pd.DataFrame(srow, columns=B.columns)
X["strict_equal"] = (X.stored == X.new) & (X.status != "MISSING FILE")
X.to_csv(f"{DST}/six_vs_main.csv", index=False)
print(X.groupby(["file", "strict_equal"]).size().unstack(fill_value=0).to_string())
pooled = {}
for f in ["s6_bootstrap_cis.csv", "s7_bootstrap_cis.csv"]:
    if os.path.exists(f"{SIX}/{f}"):
        a = S(f"{SIX}/{f}")
        pooled[f] = sorted(a[~a.model.isin(BANKED + ["mrt2"])][["analysis", "model", "scope"]].astype(str).agg("|".join, axis=1))
print("  rows not tied to one model (pooled; expected to change with mrt2 added):", {k: len(v) for k, v in pooled.items()})

# ---------------- 3. mrt2 rows ----------------
print("\n=== 3. mrt2 rows ===")
for f in ["s7_leaderboard.csv", "s6_feature_decomp.csv", "s6_raw_ridge_level_means.csv"]:
    if os.path.exists(f"{SIX}/{f}"):
        print(f"--- {f}"); print(pd.read_csv(f"{SIX}/{f}").to_string(index=False))
