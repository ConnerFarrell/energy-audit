"""Genre-adherence statistics, exactly as fixed in rerun/step3/GENRE_PLAN.md (commit 7b45f37).
Run from the repo root:  python rerun/step3/genre_analysis.py [genre_clap_per_clip.csv]

Writes to rerun/step3/genre/:
  adherence_by_cell.csv   per model x phrasing x level: top-1 %, mean prompted-genre prob, EDM share, 95% CIs
  drop_L1_L9_verbal.csv   per model: drop (L1 - L9, verbal) in both metrics, 95% CI
  mrt2_vs_others.csv      drop_mrt2 - drop_model, 95% CI, per metric (the decision rule)
  quiet_clips.csv         every clip below -50 dBFS RMS, with p0a, p0a_n, LUFS
"""
import os, sys
import numpy as np, pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "rerun/step3/genre/genre_clap_per_clip.csv"
OUT = "rerun/step3/genre"; os.makedirs(OUT, exist_ok=True)
B, SEED = 2000, 0
SLUGS = ["rock", "edm", "hiphop", "folk", "orchestral"]
MODELS = ["ace", "mgm", "mgs", "mrt2", "sa3s", "sao"]

G = pd.read_csv(SRC)
assert len(G) == 1800 and G.file.is_unique and set(G.model) == set(MODELS)
G["hit"] = (G.top_genre == G.genre_slug).astype(float)
G["p_prompted"] = [r[f"p_{r['genre_slug']}"] for r in G.to_dict("records")]
G["is_edm"] = (G.top_genre == "edm").astype(float)

def ci(boots):
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def boot_mean(a):
    rng = np.random.default_rng(SEED); a = np.asarray(a, float)
    return a[rng.integers(0, len(a), (B, len(a)))].mean(axis=1)

def boot_drop(a1, a9):
    """L1 - L9 with L1 and L9 clips resampled independently (one rng, fixed draw order)."""
    rng = np.random.default_rng(SEED); a1, a9 = np.asarray(a1, float), np.asarray(a9, float)
    i1 = rng.integers(0, len(a1), (B, len(a1))); i9 = rng.integers(0, len(a9), (B, len(a9)))
    return a1[i1].mean(axis=1) - a9[i9].mean(axis=1)

# per cell
rows = []
for (m, ph, L), g in G.groupby(["model", "phrasing", "level"]):
    r = dict(model=m, phrasing=ph, level=L, n=len(g))
    for col, name in [("hit", "top1"), ("p_prompted", "p_prompted"), ("is_edm", "edm_share")]:
        lo, hi = ci(boot_mean(g[col]))
        r[name], r[f"{name}_lo"], r[f"{name}_hi"] = g[col].mean(), lo, hi
    rows.append(r)
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/adherence_by_cell.csv", index=False)

# drops, verbal
V = G[G.phrasing == "verbal"]
drops, dboot = [], {}
for m in MODELS:
    a1, a9 = V[(V.model == m) & (V.level == 1)], V[(V.model == m) & (V.level == 9)]
    r = dict(model=m)
    for col in ["hit", "p_prompted"]:
        bt = boot_drop(a1[col], a9[col]); dboot[(m, col)] = bt
        lo, hi = ci(bt)
        r[f"drop_{col}"], r[f"drop_{col}_lo"], r[f"drop_{col}_hi"] = a1[col].mean() - a9[col].mean(), lo, hi
    drops.append(r)
D = pd.DataFrame(drops); D.to_csv(f"{OUT}/drop_L1_L9_verbal.csv", index=False)

# decision rule: drop_mrt2 - drop_model, independent resamples per model
cmp_rows = []
for m in [x for x in MODELS if x != "mrt2"]:
    for col in ["hit", "p_prompted"]:
        a1m, a9m = V[(V.model == "mrt2") & (V.level == 1)][col], V[(V.model == "mrt2") & (V.level == 9)][col]
        a1o, a9o = V[(V.model == m) & (V.level == 1)][col], V[(V.model == m) & (V.level == 9)][col]
        rng = np.random.default_rng(SEED)
        draws = [rng.integers(0, len(a), (B, len(a))) for a in (a1m, a9m, a1o, a9o)]
        arrs = [np.asarray(a, float) for a in (a1m, a9m, a1o, a9o)]
        bt = (arrs[0][draws[0]].mean(1) - arrs[1][draws[1]].mean(1)) - (arrs[2][draws[2]].mean(1) - arrs[3][draws[3]].mean(1))
        pt = (arrs[0].mean() - arrs[1].mean()) - (arrs[2].mean() - arrs[3].mean())
        lo, hi = ci(bt)
        cmp_rows.append(dict(metric="top1" if col == "hit" else "p_prompted", other=m, diff=pt, lo=lo, hi=hi,
                             mrt2_drops_more=bool(lo > 0)))
X = pd.DataFrame(cmp_rows); X.to_csv(f"{OUT}/mrt2_vs_others.csv", index=False)
prim = X[X.metric == "top1"]
verdict = "YES (all five CIs above 0)" if prim.mrt2_drops_more.all() else \
          ("NO for all five" if not prim.mrt2_drops_more.any() else "MIXED: " + ", ".join(f"{r.other} {'yes' if r.mrt2_drops_more else 'no'}" for r in prim.itertuples()))

# quiet clips
sc = pd.concat([pd.read_csv("data/scored_full_v2.csv")[["file", "p0a", "p0a_n", "lufs"]],
                pd.read_csv("rerun/step3/score_out/scores_mrt2.csv")[["file", "p0a", "p0a_n", "lufs"]]])
Q = G[G.rms_dbfs < -50][["file", "model", "prompt_id", "seed", "rms_dbfs"]].merge(sc, on="file", how="left")
assert Q.p0a.notna().all()
Q.sort_values("rms_dbfs").to_csv(f"{OUT}/quiet_clips.csv", index=False)

pd.set_option("display.width", 220)
print("top-1 adherence (verbal) by level:\n", C[C.phrasing == "verbal"].pivot(index="model", columns="level", values="top1").round(2))
print("EDM share (verbal) by level:\n", C[C.phrasing == "verbal"].pivot(index="model", columns="level", values="edm_share").round(2))
print("\ndrops (L1 - L9, verbal):\n", D.round(3).to_string(index=False))
print("\nmrt2 drop minus each model's drop:\n", X.round(3).to_string(index=False))
print("\nDECISION (primary, top-1):", verdict)
print(f"\nclips below -50 dBFS RMS: {len(Q)}\n", Q.sort_values('rms_dbfs').round(2).to_string(index=False))
print("RMS dBFS range per model:", G.groupby("model").rms_dbfs.agg(["min", "median"]).round(1).to_dict("index"))
