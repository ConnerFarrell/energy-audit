"""Cross-check numbers reported in paper/main.tex (= PDF text) and README.md
against the banked CSVs / regenerated values. A reported number matches when
the source value rounds to it at the reported number of decimals (half-up and
half-even both accepted, flagged if they disagree). Run from repo root."""
import pandas as pd, numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from scipy.stats import spearmanr

b6 = pd.read_csv("data/s6_bootstrap_cis.csv"); b7 = pd.read_csv("data/s7_bootstrap_cis.csv")
B = pd.concat([b6, b7]).set_index(["analysis", "model", "scope"])
dec = pd.read_csv("data/s6_feature_decomp.csv").set_index("model")
lm = pd.read_csv("data/s6_level_means.csv").set_index(["model", "phrasing", "level"])
lead = pd.read_csv("data/s7_leaderboard.csv").set_index("model")
edge = pd.read_csv("data/s6_edge_cells.csv", dtype=str).set_index(["model", "genre_slug"])
df = pd.read_csv("data/scored_full_v2.csv")
ax = pd.read_csv("data/s7_clap_axis.csv")

def bank(a, m, s, f): return float(B.loc[(a, m, s), f])
def rnd(x, k, mode): return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-k), rounding=mode))

rows = []
def chk(where, claim, reported, source_val, source):
    if source_val is None:
        rows.append((where, claim, reported, "", "NO SOURCE IN REPO", source)); return
    s = reported.replace("−", "-").replace("+", "").rstrip("%")
    pct = reported.endswith("%")
    v = float(source_val) * (100 if pct else 1)
    k = max(0, -Decimal(s).as_tuple().exponent)
    up, ev = rnd(v, k, ROUND_HALF_UP), rnd(v, k, ROUND_HALF_EVEN)
    ok = float(s) in (up, ev)
    note = "" if up == ev else f" (half-up {up} / half-even {ev})"
    rows.append((where, claim, reported, f"{v:.6g}", ("MATCH" if ok else "MISMATCH") + note, source))

# ---- Table 1 (tex l.181-185; PDF Table 1) ----
T1 = {"mgs": [".04", "-.13", ".20", ".39", ".25", ".51", ".35", ".20", ".50", "2.05", "0.80", "2.40", "0.93", "0.40", "1.45"],
      "mgm": [".04", "-.13", ".20", ".54", ".41", ".64", ".50", ".33", ".67", "1.30", "0.50", "2.10", "1.28", "0.91", "1.63"],
      "sao": [".01", "-.17", ".16", ".50", ".37", ".62", ".50", ".37", ".63", "1.30", "0.55", "2.00", "1.13", "0.78", "1.50"],
      "sa3s": [".04", "-.14", ".20", ".38", ".22", ".53", ".35", ".22", ".48", "1.60", "0.55", "2.20", "0.83", "0.31", "1.33"],
      "ace": [".07", "-.09", ".23", ".33", ".17", ".48", ".26", ".06", ".45", "0.10", "-0.05", "0.70", "0.18", "-0.12", "0.47"]}
cols = [("rho_level_p0a", "numeric", "rho numeric"), ("rho_level_p0a", "verbal", "rho verbal"),
        ("phrasing_drho", "verbal-numeric", "delta rho"), ("delta_L9-L1_median", "verbal", "median delta"),
        ("delta_L9-L1_mean", "verbal", "mean delta")]
for m, vals in T1.items():
    for i, (a, sc, lab) in enumerate(cols):
        for j, f in enumerate(["point", "lo", "hi"]):
            chk("tex Table 1", f"{m} {lab} {f}", vals[3 * i + j], bank(a, m, sc, f), f"s6_bootstrap_cis {a}/{sc}")

# ---- §4.1 house levels (l.207) ----
for m, v in {"sao": "6.04", "mgm": "5.83", "mgs": "5.81", "sa3s": "5.80", "ace": "4.35"}.items():
    chk("tex §4.1 l.207", f"{m} house level", v, df[(df.model == m) & (df.phrasing == "numeric")].p0a.mean(), "scored_full_v2 numeric mean")

# ---- Table 2 mechanism (l.265-269) ----
T2 = {"mgm": (".76", ".63", ".91", "1.86", "-2%", "62%", "40%"), "sa3s": (".82", ".64", "1.15", "0.73", "18%", "39%", "43%"),
      "mgs": (".76", ".47", ".99", "0.96", "21%", "27%", "53%"), "sao": (".59", ".50", ".69", "1.39", "38%", "38%", "24%")}
for m, v in T2.items():
    for f, r in zip(["point", "lo", "hi"], v[:3]):
        chk("tex Table 2", f"{m} content share {f}", r, bank("K4_content_share", m, "verbal", f), "s6_bootstrap_cis K4")
    chk("tex Table 2", f"{m} d_raw", v[3], dec.loc[m, "d_raw_total"], "s6_feature_decomp")
    for c, r in zip(["loudness_share", "noisiness_share", "tonality_share"], v[4:]):
        chk("tex Table 2", f"{m} {c}", r, dec.loc[m, c], "s6_feature_decomp")
chk("tex Table 2", "ace d_raw", "0.21", dec.loc["ace", "d_raw_total"], "s6_feature_decomp")

# ---- §4.2 LUFS deltas (l.244-246) ----
for m, r in [("sao", "6.3"), ("ace", "-0.4")]:
    chk("tex §4.2 l.244-246", f"{m} verbal LUFS L9-L1 (dB)", r,
        lm.loc[(m, "verbal", 9), "lufs"] - lm.loc[(m, "verbal", 1), "lufs"], "s6_level_means lufs")

# ---- §4.3 bands (l.280-285, 293) ----
chk("tex §4.3 l.280", "pooled in-band gain", "0.223", None, "only the difference is banked/regenerated")
chk("tex §4.3 l.280", "pooled out-of-band gain", "0.033", None, "only the difference is banked/regenerated")
for f, r in zip(["point", "lo", "hi"], ["0.190", "0.096", "0.279"]):
    chk("tex §4.3 l.281", f"pooled difference {f}", r, bank("edge_gain_in_minus_out", "ALL", "verbal", f), "s6_bootstrap_cis")
for m, r in [("ace", "0.297"), ("sa3s", "0.291"), ("mgm", "0.206"), ("mgs", "0.132"), ("sao", "0.026")]:
    chk("tex §4.3 l.282", f"{m} in-minus-out", r, bank("edge_gain_in_minus_out", m, "verbal", "point"), "s6_bootstrap_cis")
for f, r in zip(["point", "lo", "hi"], ["0.217", "0.111", "0.321"]):
    chk("tex §4.3 l.284", f"rock-excluded difference {f}", r, bank("edge_gain_norock", "ALL", "verbal", f), "s7_bootstrap_cis")
chk("tex §4.3 l.293", "mgm rock L1", "6.37", float(edge.loc[("mgm", "rock"), "1"].rstrip("*")), "s6_edge_cells")
chk("tex §4.3 l.293", "sa3s rock L1", "6.48", float(edge.loc[("sa3s", "rock"), "1"].rstrip("*")), "s6_edge_cells")

# ---- §4.4 droop (l.303-316) ----
D = [("ace", "droop_L9-L7_p0a", ("-0.55", "-0.82", "-0.25")), ("ace", "droop_L9-L7_clap", ("-0.086", "-0.138", "-0.031")),
     ("ace", "droop_L9-L7_rawridge", ("-0.37", "-0.56", "-0.18")), ("mgs", "droop_L9-L7_p0a", ("-0.42", "-0.86", "0.03")),
     ("mgs", "droop_L9-L7_rawridge", ("-0.73", "-1.37", "-0.15")), ("mgs", "droop_L9-L7_clap", ("-0.085", "-0.170", "0.004"))]
for m, a, v in D:
    for f, r in zip(["point", "lo", "hi"], v):
        chk("tex §4.4 l.303-309", f"{m} {a} {f}", r, bank(a, m, "verbal", f), "s6_bootstrap_cis")
for f, r in zip(["point", "lo", "hi"], ["1.86", "1.47", "2.24"]):
    chk("tex §4.4 l.313", f"mgm delta_raw_mean {f}", r, bank("delta_raw_mean", "mgm", "verbal", f), "s7_bootstrap_cis")
chk("tex §4.4 l.315", "raw rho ace", ".34", bank("rho_level_raw", "ace", "verbal", "point"), "s7_bootstrap_cis")
chk("tex §4.4 l.315", "raw rho mgm", ".59", bank("rho_level_raw", "mgm", "verbal", "point"), "s7_bootstrap_cis")
for f, r in zip(["point", "lo", "hi"], ["0.21", "-0.03", "0.43"]):
    chk("tex §4.4 l.316", f"ace delta_raw_mean {f}", r, bank("delta_raw_mean", "ace", "verbal", f), "s7_bootstrap_cis")

# ---- §4.5 scale/era (tex ~l.324-333) ----
for key, v in [("scale_mgm-mgs", ("0.15", "-0.00", "0.32")), ("era_sa3s-sao", ("-0.12", "-0.27", "0.02"))]:
    for f, r in zip(["point", "lo", "hi"], v):
        chk("tex §4.5", f"family_drho {key} {f}", r, bank("family_drho", key, "verbal", f), "s7_bootstrap_cis")
sv = df[df.phrasing == "verbal"]
for g_, r in [("edm", ".08"), ("rock", ".71")]:
    x = sv[(sv.model == "ace") & (sv.genre_slug == g_)]
    chk("tex §4.5", f"ace per-genre rho {g_} (computed directly, not banked)", r, spearmanr(x.level, x.p0a)[0],
        "spearman on scored_full_v2 (my computation)")
for m, r in [("mgm", "-.21"), ("sa3s", "-.27"), ("sao", "-.20")]:
    chk("tex §4.5", f"{m} rock-minus-others rho", r, None, "definition not in repo; not banked")
chk("tex §4.5", "mgm/mgs drho .54 vs .39", ".54", bank("rho_level_p0a", "mgm", "verbal", "point"), "s6_bootstrap_cis")

# ---- CLAP axis (README finding 2; tex l.~348) ----
for m, r in [("ace", ".27"), ("mgm", ".60")]:
    chk("README #2 / tex", f"axis verbal rho {m}", r, bank("rho_level_axis", m, "verbal", "point"), "s7_bootstrap_cis")
x = df.merge(ax, on="file")
chk("README #2", "clip-level agreement AI vs axis rho=.70 (all 1500 clips, my computation)", ".70",
    spearmanr(x.p0a, x.axis)[0], "spearman on scored_full_v2 x s7_clap_axis (definition not stated)")
xv = x[x.phrasing == "verbal"]
chk("README #2", "same, verbal clips only (alternative reading)", ".70", spearmanr(xv.p0a, xv.axis)[0], "same")

# ---- README findings + leaderboard ----
chk("README #2", "median range low (ace)", "0.1", lead.loc["ace", "med_delta"], "s7_leaderboard")
chk("README #2", "median range high (mgs)", "2.05", lead.loc["mgs", "med_delta"], "s7_leaderboard")
chk("README #3", "content share low (sao) 59%", "59%", bank("K4_content_share", "sao", "verbal", "point"), "s6_bootstrap_cis")
chk("README #3", "content share high (sa3s) 82%", "82%", bank("K4_content_share", "sa3s", "verbal", "point"), "s6_bootstrap_cis")
chk("README #3", "sao loudness share 38%", "38%", dec.loc["sao", "loudness_share"], "s6_feature_decomp")
RL = {"mgm": (".54", ".41", ".64", "1.30", "0.50", "2.10", ".76", ".63", ".91"),
      "sao": (".50", ".37", ".62", "1.30", "0.55", "2.00", ".59", ".50", ".69"),
      "mgs": (".39", ".25", ".51", "2.05", "0.80", "2.40", ".76", ".47", ".99"),
      "sa3s": (".38", ".22", ".53", "1.60", "0.55", "2.20", ".82", ".64", "1.15"),
      "ace": (".33", ".17", ".48", "0.10", "-0.05", "0.70")}
for m, v in RL.items():
    srcs = [("rho_level_p0a", "verbal")] * 3 + [("delta_L9-L1_median", "verbal")] * 3 + [("K4_content_share", "verbal")] * 3
    for i, r in enumerate(v):
        a, sc = srcs[i]; f = ["point", "lo", "hi"][i % 3]
        chk("README leaderboard", f"{m} {a} {f}", r, bank(a, m, sc, f), "s6_bootstrap_cis")

# ---- App. A edge table, App. B features (l.396-407) ----
for g_, v in [("rock", ["6.37", "6.38", "6.48", "6.50", "6.48"]), ("folk", ["4.43", "4.03", "5.82", "4.90", "6.25"])]:
    for L, r in zip(["1", "3", "5", "7", "9"], v):
        chk("tex App. A", f"mgm/{g_} L{L}", r, float(edge.loc[("mgm", g_), L].rstrip("*")), "s6_edge_cells")
chk("tex App. B", "mgm ZCR delta", "0.126", dec.loc["mgm", "dfeat_zcr_mean"], "s6_feature_decomp")
ton = [c for c in dec.columns if c.startswith("dfeat_") and c.endswith("_mean") and abs(dec.loc["mgm", c] + 14.83) < 0.01]
chk("tex App. B", "mgm tonality delta", "-14.8", dec.loc["mgm", ton[0]] if ton else None, f"s6_feature_decomp {ton}")
chk("tex App. B", "mgm RMS delta", "-0.009", dec.loc["mgm", "dfeat_loudness_mean"], "s6_feature_decomp")
chk("tex App. B", "sao RMS delta", "0.106", dec.loc["sao", "dfeat_loudness_mean"], "s6_feature_decomp")

# ---- instrument / Tier-2 numbers with no Tier-1 source ----
for w, c, r in [("tex §3 l.136", "Arousal Index DEAM R^2", "0.29"), ("tex §3 l.137", "PMEmo transfer rho", "0.77"),
                ("tex §3 l.150", "control clips score range", "3.7-4.1 vs 6.0-6.5"),
                ("tex l.350 / README Tier 2", "CLAP run-to-run drift", "<=0.033")]:
    rows.append((w, c, r, "", "NO SOURCE IN REPO (instrument paper / Tier 2)", "out of Tier-1 scope"))

res = pd.DataFrame(rows, columns=["where", "claim", "reported", "source_value", "status", "source"])
res.to_csv("rerun/step1/paper_crosscheck.csv", index=False)
print(res.status.str.split(" \\(").str[0].value_counts().to_string())
print(res[~res.status.str.startswith("MATCH")].to_string(index=False))
print(res[res.status.str.contains("half-up")].to_string(index=False))
