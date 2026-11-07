"""Step 1 (Tier-1) full-precision comparison.

Loads the definitions cell (cell 1) of notebooks/repro.ipynb UNMODIFIED and
calls its own functions (ci, boot_rho, boot_stat, contrast logic) the same way
the notebook cells do, so every value is produced by the repo's code. Each
regenerated value is compared to the banked data/*.csv value at the stored
precision: a cell "matches" when |new - stored| <= half a unit in the stored
value's last decimal place (i.e. new rounds to the stored digits).

Reads data/ only. Writes only under rerun/step1/.
Run from the repo root: python rerun/step1/compare.py
"""
import json, os, sys
from decimal import Decimal
import numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = "rerun/step1/regenerated"
os.makedirs(OUT, exist_ok=True)

nb = json.load(open("notebooks/repro.ipynb"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
NS = {}
exec(src(code_cells[0]), NS)            # cell 1: imports, constants, ci/boot_* defs, df load
# cell 7's band definitions + contrast(), executed without its print loop
cell7 = src(code_cells[6]).split("for label, genres")[0]
exec(cell7, NS)
df, SV, MODELS, ci, boot_rho, boot_stat, contrast, BANDS = (
    NS[k] for k in ["df", "SV", "MODELS", "ci", "boot_rho", "boot_stat", "contrast", "BANDS"])

def decimals(s):
    s = str(s).strip()
    try: d = Decimal(s)
    except Exception: return None
    return max(0, -d.as_tuple().exponent)

rows = []   # file, key, field, stored, new, status, source
def cmp(file, key, field, stored, new, source):
    if new is None:
        rows.append((file, key, field, stored, "", "NOT REGENERATED", source)); return
    if isinstance(stored, str) and decimals(stored) is None:
        ok = str(stored) == str(new)
        rows.append((file, key, field, stored, new, "MATCH" if ok else "MISMATCH", source)); return
    k = decimals(stored); s = float(stored); n = float(new)
    tol = 0.5 * 10 ** (-k) + 1e-9
    d = abs(n - s)
    st = "MATCH" if d <= tol else "MISMATCH"
    half = 0.5 * 10 ** (-k)
    if st == "MATCH" and d > 1e-12 and abs(d - half) <= 1e-6 * half: st = "MATCH (rounding tie)"
    rows.append((file, key, field, stored, repr(n), st, source))

def raw(path):  # keep stored strings exactly as written
    return pd.read_csv(path, dtype=str, keep_default_na=False)

g = lambda m, ph: df[(df.model == m) & (df.phrasing == ph)]
regen = {}

# ---------------- s6_bootstrap_cis.csv / s7_bootstrap_cis.csv ----------------
def compute(analysis, model, scope):
    """Return (point, lo, hi, source) using the notebook's functions, or None."""
    if analysis == "rho_level_p0a":
        x = g(model, scope); p = spearmanr(x.level, x.p0a)[0]
        lo, hi = ci(boot_rho(x.level.values, x.p0a.values))
        return p, lo, hi, "nb cell 2 (verbal) / same call (numeric)"
    if analysis.startswith("delta_L9-L1_"):
        x = g(model, scope); a1 = x[x.level == 1].p0a.values; a9 = x[x.level == 9].p0a.values
        if analysis.endswith("median"):
            p = np.median(a9) - np.median(a1); fn = lambda xs: np.median(xs[1]) - np.median(xs[0])
        else:
            p = a9.mean() - a1.mean(); fn = lambda xs: xs[1].mean() - xs[0].mean()
        lo, hi = ci(boot_stat([a1, a9], fn))
        return p, lo, hi, "nb cell 2 (verbal) / same call (numeric)"
    if analysis == "phrasing_drho":
        gv, gn = g(model, "verbal"), g(model, "numeric")
        p = spearmanr(gv.level, gv.p0a)[0] - spearmanr(gn.level, gn.p0a)[0]
        lo, hi = ci(boot_rho(gv.level.values, gv.p0a.values) - boot_rho(gn.level.values, gn.p0a.values))
        return p, lo, hi, "nb cell 2"
    if analysis.startswith("droop_L9-L7_") and analysis[12:] in ("p0a", "clap"):
        col = analysis[12:]; s = SV[SV.model == model]
        g7, g9 = s[s.level == 7], s[s.level == 9]
        p = g9[col].mean() - g7[col].mean()
        lo, hi = ci(boot_stat([g7[col].values, g9[col].values], lambda xs: xs[1].mean() - xs[0].mean()))
        return p, lo, hi, "nb cell 6"
    if analysis == "K4_content_share":
        s = SV[SV.model == model]
        A1 = s[s.level == 1][["p0a", "p0a_n"]].values; A9 = s[s.level == 9][["p0a", "p0a_n"]].values
        rd = A9[:, 0].mean() - A1[:, 0].mean()
        p = (A9[:, 1].mean() - A1[:, 1].mean()) / rd
        with np.errstate(divide="ignore", invalid="ignore"):
            lo, hi = ci(boot_stat([A1, A9], lambda xs:
                (xs[1][:, 1].mean() - xs[0][:, 1].mean()) / (xs[1][:, 0].mean() - xs[0][:, 0].mean())))
        return p, lo, hi, "nb cell 3"
    if analysis in ("edge_gain_in_minus_out", "edge_gain_norock"):
        genres = list(BANDS) if analysis == "edge_gain_in_minus_out" else [x for x in BANDS if x != "rock"]
        sub = SV if model == "ALL" else SV[SV.model == model]
        p, lo, hi = contrast(sub, genres)
        return p, lo, hi, "nb cell 7 (ALL) / same contrast() per model"
    if analysis in ("rho_level_axis", "droop_L9-L7_axis"):
        ax = pd.read_csv("data/s7_clap_axis.csv")[["file", "axis"]]
        x = df.merge(ax, on="file", how="left")
        assert x.axis.notna().all()
        x = x[(x.model == model) & (x.phrasing == scope)]
        if analysis == "rho_level_axis":
            p = spearmanr(x.level, x.axis)[0]; lo, hi = ci(boot_rho(x.level.values, x.axis.values))
        else:
            a7, a9 = x[x.level == 7].axis.values, x[x.level == 9].axis.values
            p = a9.mean() - a7.mean()
            lo, hi = ci(boot_stat([a7, a9], lambda xs: xs[1].mean() - xs[0].mean()))
        return p, lo, hi, "nb functions on data/s7_clap_axis.csv (nb cell 11 only prints the bank)"
    return None   # rawridge / raw-scale / phrasing_ddelta / family_drho: no code in repo

for fname in ["s6_bootstrap_cis.csv", "s7_bootstrap_cis.csv"]:
    bank = raw(f"data/{fname}"); out = []
    for _, r in bank.iterrows():
        key = f"{r.analysis}|{r.model}|{r.scope}"
        res = compute(r.analysis, r.model, r.scope)
        if res is None:
            for f in ["point", "lo", "hi"]:
                cmp(fname, key, f, r[f], None, "no generating code in repo")
            continue
        p, lo, hi, source = res
        out.append(dict(analysis=r.analysis, model=r.model, scope=r.scope, point=p, lo=lo, hi=hi))
        for f, v in zip(["point", "lo", "hi"], [p, lo, hi]):
            cmp(fname, key, f, r[f], v, source)
    regen[fname] = pd.DataFrame(out)

# ---------------- s7_leaderboard.csv ----------------
bank = raw("data/s7_leaderboard.csv"); out = []
for _, r in bank.iterrows():
    m = r.model; v = g(m, "verbal"); n = g(m, "numeric")
    rho = spearmanr(v.level, v.p0a)[0]; rlo, rhi = ci(boot_rho(v.level.values, v.p0a.values))
    a1 = v[v.level == 1].p0a.values; a9 = v[v.level == 9].p0a.values
    md = np.median(a9) - np.median(a1)
    mlo, mhi = ci(boot_stat([a1, a9], lambda xs: np.median(xs[1]) - np.median(xs[0])))
    cs, clo, chi, _ = compute("K4_content_share", m, "verbal")
    rd_lo, rd_hi = ci(boot_stat([v[v.level == 1][["p0a"]].values, v[v.level == 9][["p0a"]].values],
                                 lambda xs: xs[1][:, 0].mean() - xs[0][:, 0].mean()))
    cs_val = "undefined" if rd_lo <= 0 <= rd_hi else cs
    vals = dict(house_level=n.p0a.mean(), rho_verbal=rho, rho_lo=rlo, rho_hi=rhi, med_delta=md,
                md_lo=mlo, md_hi=mhi, content_share=cs_val, cs_lo=clo, cs_hi=chi)
    out.append(dict(model=m, **vals))
    for f, val in vals.items():
        cmp("s7_leaderboard.csv", m, f, r[f], val,
            "house_level = numeric-phrasing mean p0a (inferred from paper §4.1 text)" if f == "house_level"
            else "nb cells 2/3 functions")
regen["s7_leaderboard.csv"] = pd.DataFrame(out)

# ---------------- s6_level_means.csv ----------------
bank = raw("data/s6_level_means.csv")
lm = df.groupby(["model", "phrasing", "level"])[["p0a", "clap", "lufs", "p0a_n", "clap_n"]].mean()
regen["s6_level_means.csv"] = lm.reset_index()
for _, r in bank.iterrows():
    k = (r.model, r.phrasing, int(r.level))
    for f in ["p0a", "clap", "lufs", "p0a_n", "clap_n"]:
        cmp("s6_level_means.csv", "|".join(map(str, k)), f, r[f], lm.loc[k, f], "groupby mean of scored_full_v2.csv")

# ---------------- s6_edge_cells.csv ----------------
bank = raw("data/s6_edge_cells.csv")
edge = SV.groupby(["model", "genre_slug", "level"]).p0a.mean().unstack()   # nb cell 7 (before its .round(2))
regen["s6_edge_cells.csv"] = edge.reset_index()
for _, r in bank.iterrows():
    for L in ["1", "3", "5", "7", "9"]:
        stored = r[L].rstrip("*")
        cmp("s6_edge_cells.csv", f"{r.model}|{r.genre_slug}", f"L{L}", stored,
            edge.loc[(r.model, r.genre_slug), int(L)], "nb cell 7")

# ---------------- s7_axis_level_means.csv ----------------
bank = raw("data/s7_axis_level_means.csv")
ax = pd.read_csv("data/s7_clap_axis.csv")[["file", "axis"]]
am = df.merge(ax, on="file").groupby(["model", "phrasing", "level"]).axis.mean()
regen["s7_axis_level_means.csv"] = am.reset_index()
for _, r in bank.iterrows():
    k = (r.model, r.phrasing, int(r.level))
    cmp("s7_axis_level_means.csv", "|".join(map(str, k)), "axis", r.axis, am.loc[k],
        "groupby mean of s7_clap_axis.csv (nb cell 11 prints the bank)")

# ---------------- s7_clap_axis.csv (internal identity axis = sim_hi - sim_lo) ----------------
bank = raw("data/s7_clap_axis.csv")
for _, r in bank.iterrows():
    cmp("s7_clap_axis.csv", r.file, "axis", r.axis, float(r.sim_hi) - float(r.sim_lo),
        "identity: sim_hi - sim_lo (per-clip inputs are Tier-2 outputs)")

# ---------------- s7_skew_L1L9.csv ----------------
bank = raw("data/s7_skew_L1L9.csv")
sk = SV[SV.level.isin([1, 9])].set_index(["model", "genre_slug", "level", "seed"]).p0a
regen["s7_skew_L1L9.csv"] = sk.reset_index()
for _, r in bank.iterrows():
    k = (r.model, r.genre_slug, int(r.level), int(r.seed))
    cmp("s7_skew_L1L9.csv", "|".join(map(str, k)), "p0a", r.p0a, sk.loc[k], "subset of scored_full_v2.csv")

# ---------------- s6_feature_decomp.csv / s6_raw_ridge_level_means.csv ----------------
for fname in ["s6_feature_decomp.csv", "s6_raw_ridge_level_means.csv"]:
    bank = raw(f"data/{fname}")
    for _, r in bank.iterrows():
        for f in bank.columns[1:]:
            cmp(fname, r.model, f, r[f], None,
                "no generating code in repo (needs pre-calibration ridge scores / decomposition)")

# ---------------- write ----------------
res = pd.DataFrame(rows, columns=["file", "key", "field", "stored", "regenerated", "status", "source"])
res.to_csv("rerun/step1/cell_comparison.csv", index=False)
for k, v in regen.items():
    v.to_csv(f"{OUT}/{k}", index=False, float_format="%.17g")
summ = res.assign(m=res.status.str.startswith("MATCH"), mm=res.status == "MISMATCH",
                  nr=res.status == "NOT REGENERATED").groupby("file")[["m", "mm", "nr"]].sum()
summ["cells"] = summ.sum(axis=1)
print(summ.to_string())
print("\nMISMATCHES:")
print(res[res.status == "MISMATCH"].to_string(index=False))
print("\nTIES:", (res.status == "MATCH (rounding tie)").sum())
