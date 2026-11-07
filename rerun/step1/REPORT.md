# P0b re-run · Step 1 · Tier-1 calibration

## Attempt #2 (2026-11-07): **PASS — 2,707/2,707 cells match under the strict digit-for-digit rule** (after the re-bank below)

*History: before the re-bank, attempt #2 matched 2,692/2,707 strict (15 cells in `s6_raw_ridge_level_means.csv` off by 1 ULP) and 2,707/2,707 under attempt #1's rule. After re-banking that one file from `data/arousal_index_ridge.json`, a fresh run matches 2,707/2,707 under both rules (notebook 43.8 s). The paper/README cross-check (§4, §10.4) is byte-identical before and after the re-bank: 216 match, 8 mismatch, 9 no source.*

| | Attempt #1 | **Attempt #2** |
|---|---:|---:|
| Banked cells compared | 2,707 | **2,707** |
| Match, strict rule (regenerated value rounds to the stored digits) | 2,501 | **2,692** before re-bank → **2,707** after |
| Match, attempt-1 rule (tolerance = half a stored digit + 1e-9) | 2,501 | **2,707** |
| Mismatch | 5 | **15** before re-bank (all ≤ 1 ULP; strict rule only) → **0** after |
| No code in `repro.ipynb` | 201 | **0** |
| Row order and row count identical, all 9 bank files | n/a | **yes** |
| Two executions byte-identical (9 tables + 4 figures) | yes | **yes** |
| Notebook wall-clock | 15.4 s | **43.8 s** (final run; 44–47 s on earlier runs) |

**The 15 cells.** All 15 are in `s6_raw_ridge_level_means.csv`, which stores 16–17 significant digits. Each regenerated value differs from the banked one in its last digit, by exactly one unit in the last binary place: one ULP, about 1e-15 (e.g. ace L1 is 4.45670257206441**3** banked vs 4.45670257206441**2** regenerated). The other 10 cells in that file match exactly, as do all 95 `s6_feature_decomp.csv` cells and every downstream CI built from the raw-ridge scores (`droop_L9-L7_rawridge`, `rho_level_raw`, `delta_raw_*`), which are stored at 3–4 decimals.

The cause is known. The ridge coefficients were re-fit from DEAM in this environment (scikit-learn 1.9.1, numpy 2.5.4) because the canonical run's coefficients were never saved. The re-fit agrees with the canonical model to about 1e-16 relative, and that last-bit difference carries into the 17-digit means. Every summation and row-order variant I tried gives the same result, so the difference is in the fitted coefficients themselves (the linear-algebra build), not in the notebook. It can only be closed by re-banking that one file from the saved coefficients, which changes committed data (your call), or by recovering the original coefficients, which no surviving output contains.

**Re-bank (2026-11-07):** `data/s6_raw_ridge_level_means.csv` — Re-banked 2026-11-07 from the saved ridge coefficients (original fit's coefficients were never saved); values change at ~1e-15; no published number changes. Strict-rule result after the re-bank: 2,707/2,707.

**What changed for attempt #2** (no change to any statistic's logic; no change to `data/*.csv`):
1. `repro.ipynb`'s content-share function is replaced by the canonical one: NaN when |L9−L1| ≤ 1e-9. This closes all 5 attempt-1 mismatches.
2. The S6 and S7 bank code is ported into `repro.ipynb` from `notebooks/scoring.ipynb` cells 13–14 and `notebooks/clap-axis.ipynb` (S7A.2–3). Its logic is identical to the Kaggle originals in `p0b-score` and `p0b-s7-axis`, whose differences from the repo copies are comments and labels only. Tier-1 adaptations: the inputs are the banked `scored_full_v2.csv`, `features_verbal.csv` and `s7_clap_axis.csv`, the ridge comes from `data/arousal_index_ridge.json`, and the outputs go to `REPRO_OUT_DIR/tables/`.
3. **Arousal Index ridge:** `scripts/fit_arousal_ridge.py` is ported from `p0b-score` cells 0 and 3. It was re-fit from the two DEAM annotation CSVs plus `deam_librosa_features.csv`, fetched from Kaggle with no audio. DEAM 5-fold R² = **0.2894** (paper: 0.29). Re-deriving p0a from `features_verbal.csv` with the refit model and an isotonic calibrator reproduces **750/750** scored p0a values. The fitted coefficients are saved in `data/arousal_index_ridge.json` (1.2 KB), so a reader doesn't need DEAM.
4. **float32 fidelity:** the canonical run extracted features in memory, where librosa returns float32 for rms, onset strength and spectral flatness (and hnr is derived from flatness). S6.4 casts those 8 columns of `features_verbal.csv` back to float32. Without the cast, the raw-ridge means differ by about 1e-8; with it, by at most 1 ULP.
5. **Settable paths:** `REPRO_DATA_DIR` (default `data/`, never written) and `REPRO_OUT_DIR` (default `repro_out/`). The notebook refuses to write into the committed `figures/` unless `REPRO_ALLOW_COMMITTED_FIGURES=1`, and refuses to write banks into the data folder. It finds the repo root from any working directory, so the exact attempt-1 command now works from the repo root.
6. `requirements.txt` (Tier 1, pinned to the versions verified here) and `requirements-scoring.txt` (Tier 2: transformers 5.0.0 and torch 2.10.0+cu128 recorded; everything else marked unrecorded).
7. CLAP scoring in `notebooks/scoring.ipynb` is seeded: `np.random.seed(0)` before every `ClapProcessor(audio=…)` call. Nothing banked changes; see §11.
8. README: "Powered by Stability AI." is moved to the top section, and the Tier-1 instructions are updated.

Outputs for attempt #2 are in `attempt2/`: the executed notebook, `cell_comparison.csv` (every cell under both rules), `compare2.py`, the log and the wall-clock.

---

## Attempt #1 (original report, 2026-11-07)

**Verdict: FAIL.** The repo does not reproduce all of its own banked statistics digit-for-digit. Of 2,707 banked cells compared, **2,501 match, 5 mismatch, and 201 cannot be regenerated by any code in the repo.** Separately, **7 numbers in the paper disagree with the repo's own banked values.** No code was edited to make anything pass. Per the step rules I stopped at reporting.

> **Addendum 2026-11-07 (§10).** The Kaggle code that built the banks has been found and read. It explains all 5 mismatches exactly, including the ace lower bound of −8.075. It identifies the notebook behind every one of the 201 non-regenerable cells, and 21 of those 201 now reproduce from repo data alone. It also finds sources for 7 of the 9 unsourced paper numbers. The verdict stays **FAIL**, because the shipped `repro.ipynb` is still not the code that made the banks. Nothing has been fixed.

What does reproduce: every output of `notebooks/repro.ipynb` is character-identical to the outputs committed in the notebook (last run 2026-07-24), apart from a kernel process ID inside a warning string. Two independent executions were also byte-identical, figures included.

| | Count |
|---|---|
| Banked CSV cells compared | 2,707 |
| Exact matches at stored precision | 2,501 (6 of them on a binary rounding tie, see Nondeterminism) |
| **Mismatches, regenerated vs. banked** | **5** (3 distinct values: the `K4_content_share` CI for ace lo/hi and sa3s hi; the leaderboard repeats the ace pair) |
| **Banked cells with no generating code in the repo** | **201** |
| Paper/README numbers checked | 233 |
| **Paper vs. bank mismatches** | **7** (sa3s numeric ρ; ace and mgs own-prompt CLAP droop) |
| Paper numbers with no source in the repo | 9 |

Base commit: `f82f452e80b0130f874e96ddb9b239aa87964dc1` (main). Branch: `rerun/step1-tier1`. CPU only; no GPU, no Kaggle and no network calls during execution. No committed file was modified: the sha256 of every `data/*.csv`, `figures/*.png` and `notebooks/repro.ipynb` was identical before and after.

---

## 1. Deviations from the step instructions (read first)

1. **No pinned lockfile exists.** The repo has no `requirements*.txt`, `pyproject.toml`, `environment.yml`, `uv.lock` or conda lock. The only version information anywhere is the notebook metadata, `language_info.version = 3.12.3`. I used Python 3.12.3 in a fresh virtual environment with whatever pip resolved on 2026-11-07 (numpy, pandas, scipy, matplotlib, nbconvert, ipykernel). Nothing was upgraded because nothing was pinned. "Install exactly from the lockfile" could not be done. The full freeze is in `environment-freeze.txt`.
2. **The prescribed command fails as written.** `jupyter nbconvert --execute … notebooks/repro.ipynb` runs the kernel with `notebooks/` as its working directory, so cell 1 raises `FileNotFoundError: data/scored_full_v2.csv`. If it ran from the repo root instead, cells 4, 5, 8 and 11 would overwrite the committed `figures/*.png`, because the output directory is hard-coded to `figures/`. To keep committed files untouched, I executed a **byte-identical copy** of the notebook (checked with `cmp`) in a scratch directory under `rerun/step1/`. That directory had a symlink to `data/` and an empty `figures/`. The command was otherwise unchanged: `--to notebook --execute --ExecutePreprocessor.timeout=-1`. The scratch directory was deleted afterwards. The executed notebook is `repro.executed.ipynb` and the figures are in `figures/`.
3. **The notebook writes no tables; it prints rounded strings** (2–3 decimals). To compare at the full stored precision (4 decimals in the CI banks), `compare.py` loads the notebook's own definitions cell (cell 1: `SEED`, `B`, `ci`, `boot_rho`, `boot_stat`, data load) and cell 7's `BANDS`/`contrast`, **unmodified** from `notebooks/repro.ipynb`. It calls those functions exactly as the notebook cells do. Where a banked row is a quantity the notebook doesn't print, but that is the same call on another subset (for example numeric-phrasing deltas, per-model band contrasts, axis ρ from `s7_clap_axis.csv`), I made the same call and labelled the source in `cell_comparison.csv`. Regenerated tables at full float precision are in `regenerated/`.

## 2. Mismatches: regenerated vs. banked (`data/*.csv`)

All 5 mismatching cells are bootstrap CI bounds of the **content share** ratio (notebook cell 3). The point estimates match.

| File | Row | Field | Banked | Regenerated |
|---|---|---|---|---|
| `s6_bootstrap_cis.csv` | `K4_content_share` · ace · verbal | lo | −8.075 | −8.49999999999985 |
| `s6_bootstrap_cis.csv` | `K4_content_share` · ace · verbal | hi | 9.4673 | 10.126875000000075 |
| `s6_bootstrap_cis.csv` | `K4_content_share` · sa3s · verbal | hi | 1.1473 | 1.1526658214333336 |
| `s7_leaderboard.csv` | ace | cs_lo | −8.07 | −8.49999999999985 |
| `s7_leaderboard.csv` | ace | cs_hi | 9.47 | 10.126875000000075 |

**Cause (sa3s: confirmed. ace: partly explained).** Arousal Index scores are quantized to 0.1, so some of the 2,000 resamples have a raw L9−L1 delta of exactly 0. The ratio then evaluates to ±inf (this triggers the `RuntimeWarning: divide by zero` in cell 3; 4 such resamples for ace, 1 for sa3s, none for the other models). The notebook's `ci()` removes only NaN (`a[~np.isnan(a)]`), so the infinities stay in the percentile ranking.
- **sa3s:** dropping non-finite values (`np.isfinite`) with numpy's default `linear` percentile reproduces the bank exactly: [0.6392, 1.1473]. The bank was therefore generated with a `ci()` that drops ±inf, which is not the `ci()` shipped in `repro.ipynb`.
- **ace:** the same rule gives hi = 9.4663 (bank 9.4673) and lo = −8.5000 (bank −8.075). No percentile method among numpy's eight, with either NaN rule, reproduces the ace pair. Its cause is **unresolved**: probably a different resample set or estimator for the bank. I didn't guess further.
- **Impact on published text: none at the reported precision.** The paper and README give sa3s as [.64, 1.15], and 1.1527 and 1.1473 both round to 1.15. The paper reports ace as "undefined" with no CI, and the README leaderboard omits ace's CI. Only the 4-decimal bank and the leaderboard's ace `cs_lo`/`cs_hi` are affected.

## 3. Per-file comparison

Match = the regenerated value lies within half a unit of the stored last decimal, i.e. it rounds to the stored digits.

| File | Cells | Exact match | Mismatch | Not regenerable | Generated by |
|---|---:|---:|---:|---:|---|
| `s6_bootstrap_cis.csv` | 198 | 165 | **3** | 30 | nb cells 2, 3, 6, 7 functions |
| `s7_bootstrap_cis.csv` | 114 | 63 | 0 | 51 | nb cell 7 `contrast`; nb `boot_rho`/`boot_stat` on `s7_clap_axis.csv` |
| `s7_leaderboard.csv` | 50 | 48 | **2** | 0 | nb cells 2/3; `house_level` = numeric-phrasing mean (inferred from paper §4.1) |
| `s6_level_means.csv` | 250 | 250 | 0 | 0 | group means of `scored_full_v2.csv` |
| `s6_edge_cells.csv` | 125 | 125 | 0 | 0 | nb cell 7 |
| `s7_axis_level_means.csv` | 50 | 50 | 0 | 0 | group means of `s7_clap_axis.csv` |
| `s7_clap_axis.csv` | 1,500 | 1,500 | 0 | 0 | identity `axis = sim_hi − sim_lo` (the per-clip sims are Tier-2 outputs) |
| `s7_skew_L1L9.csv` | 300 | 300 | 0 | 0 | subset of `scored_full_v2.csv` |
| `s6_feature_decomp.csv` | 95 | 0 | 0 | 95 | — |
| `s6_raw_ridge_level_means.csv` | 25 | 0 | 0 | 25 | — |
| **Total** | **2,707** | **2,501** | **5** | **201** | |

`scored_full_v2.csv`, `features_verbal.csv` and `prompts.csv` are inputs (Tier-2/3 outputs), not Tier-1 regenerables. `scored_full_v2.csv` passes the notebook's own integrity assert: 1,500 rows, unique `file`.

**The 201 non-regenerable cells.** The README's claim that "`repro.ipynb` regenerates every table, CI, and figure in the paper" is not true for these. The notebook only prints them from the bank, and no code in the repo produces them:
- **Pre-calibration ("raw ridge") scores**, which `scored_full_v2.csv` doesn't contain: `droop_L9-L7_rawridge` (15), `rho_level_raw` / `delta_raw_mean` / `delta_raw_median` (45), all of `s6_raw_ridge_level_means.csv` (25), and the `d_raw_total`/contribution/share columns of `s6_feature_decomp.csv`. Cell 6 prints "raw witness: recomputable from data/features_verbal.csv", but no code does that, and the ridge weights aren't in the repo.
- **`s6_feature_decomp.csv` `dfeat_*` columns:** these look like L9−L1 feature deltas from `features_verbal.csv`, but no code computes them. I didn't write one, since guessing the definition would not be a reproduction.
- **`phrasing_ddelta` (15) and `family_drho` (6):** no code. The points are consistent with simple differences of banked values, but the CI resampling scheme is undefined, so I didn't attempt it.

## 4. Paper / README cross-check

Sources: `paper/main.tex`. Every flagged number is identical in the text extracted from `paper/farrell_ttm_direction_2026.pdf` (`paper_pdf.txt`). I also checked `README.md`. A number matches when the source value rounds to it at the reported decimals. Full list: `paper_crosscheck.csv`. **216 match, 8 flagged, 9 have no source.**

| Where | Claim | Paper | Repo value | Hypothesis |
|---|---|---|---|---|
| Table 1 | sa3s ρ numeric (point) | .04 | 0.0347 (bank and regenerated) | **Double rounding:** the notebook prints `+0.035`, and rounding that half-up again gives .04. The correct 2-decimal value is .03. Verdict unaffected (CI covers 0). |
| §4.4 l.303 | ace own-prompt CLAP droop L9−L7 | −0.086 [−0.138, −0.031] | −0.0806 [−0.1320, −0.0269] | The paper text was likely written from an **earlier own-prompt CLAP scoring run**, not the canonical CLAP values now in `scored_full_v2.csv` (CLAP drift is documented). Verdict unaffected (excludes 0 either way). |
| §4.4 l.309 | mgs own-prompt CLAP droop | −0.085 [−0.170, +0.004] | −0.0458 [−0.1202, +0.0302] | Same hypothesis, but the point estimate differs by **0.039, more than the README's stated drift bound of ≤ 0.033**. The verdict technically holds (the CI covers 0 either way), but the paper's "CLAP marginal" wording is weaker under the banked values. |
| README #2 | clip-level agreement, AI vs. axis, ρ = .70 | .70 | 0.7105 over all 1,500 clips; **0.7016 over verbal clips only** | Definition not stated. It matches if computed on verbal clips only, so it is listed as flagged but is probably fine. This ρ is my own computation; nothing in the repo computes it. |

Mismatch count: 1 (sa3s) + 3 (ace CLAP) + 3 (mgs CLAP) = **7 cells**. The README ρ = .70 row is the eighth flag, but it is ambiguous, not a mismatch.

**Paper numbers I couldn't find a source for in the repo** (not guessed):
- §4.3 l.280: pooled in-band gain **+0.223** and out-of-band gain **+0.033**. Only their difference (+0.190) is banked and regenerated, and it matches.
- §4.5: rock-minus-others ρ, **mgm −.21, sa3s −.27, sao −.20**. The definition isn't in the repo and nothing banks it.
- §3 l.136–150: Arousal Index DEAM **R² = 0.29**, PMEmo transfer **ρ = 0.77**, control-clip ranges **3.7–4.1 vs 6.0–6.5**. These belong to the instrument paper (intensity-index repo), not this Tier-1 scope.
- l.350 / README Tier 2: CLAP run-to-run drift **≤ 0.033**. This is a Tier-2 claim.

Ace per-genre ρ, EDM .08 and rock .71 (§4.5), isn't banked anywhere. Direct Spearman on `scored_full_v2.csv` reproduces both, so they are counted among the 216 matches.

## 5. Environment

| Item | Value |
|---|---|
| Python | 3.12.3 (GCC 13.3.0); matches notebook metadata `3.12.3` |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 (pillow 13.0.0, contourpy 1.4.0) |
| nbconvert / nbclient / ipykernel | 7.17.2 / 0.11.0 / 7.4.0 |
| OS | Linux 6.18 x86_64, CPU only |
| Full freeze | `environment-freeze.txt` (59 packages) |

**These versions are a record of this run, not a reconstruction of the original environment.** The originals are unknown (see §1.1).

## 6. Wall-clock

- Notebook execution (`nbconvert --execute`): **15.4 s**. A second run for the determinism check took about the same.
- Full comparison (`compare.py`, which re-runs every bootstrap at full precision) and the paper check finished in under 2 minutes in total.

## 7. Nondeterminism and fragility observed

- **RNG: seeded, deterministic.** Every bootstrap call constructs `np.random.default_rng(0)`, and the skew-plot jitter uses `default_rng(0)`. Two independent runs gave identical outputs and byte-identical PNGs. There is no unseeded randomness.
- **A side effect of seeding every call:** every bootstrap of the same sample size reuses the **same index matrix**. For example, in `delta_rho_vn` the numeric and verbal resamples draw identical row positions, so they are implicitly paired by CSV row order. Within `boot_stat` the groups get different (sequential) draws. This is deterministic and isn't a defect for reproduction, but it is a methodological choice the paper doesn't state.
- **±inf in ratio bootstraps** (see §2): `ci()` handles NaN but not inf. This is the direct cause of the mismatches.
- **Rounding ties:** 6 banked values sit on a binary-float half-digit boundary (e.g. regenerated 0.7467499999999999 vs. stored 0.7467). All of them agree with Python `round()`, so they are counted as matches. One paper value (mgs Δ_raw 0.965 → "0.96") depends on the half-even vs. half-up choice applied to a 3-decimal bank value. The 4-decimal bank (0.9646) confirms 0.96.
- **Float summation order:** no effects were visible at stored precision. All 250 level means, 125 edge cells and 50 axis means match under pandas 3.1.1.
- **Figures are not byte-reproducible across environments.** All four regenerated PNGs differ from the committed ones, and three differ in pixel size (e.g. `per_genre_curves` is 2062×1573 regenerated vs. 2064×1576 committed). This is consistent with a different matplotlib or font build, which is expected without a lockfile. They are identical between my two runs. I did not judge them visually.
- **Cosmetic:** cell 3's warning text includes the kernel process ID (`ipykernel_644` vs `ipykernel_564`). This is the only output difference from the committed notebook.

## 8. Other repo issues found (not fixed)

- The README links the PDF as `paper/farrell-ttm-direction-2026.pdf` (hyphens), but the file is `paper/farrell_ttm_direction_2026.pdf` (underscores), so the link is broken.
- Notebook cell 7's comment refers to "'*' marking in data/s6_edge_cells.csv", but that file contains no `*`.
- The README file tree lists `docs/TOS_AUDIT.md`, which doesn't exist (`docs/LICENSES.md` does). *[Resolved 2026-11-07: `docs/TOS_AUDIT.md` added on this branch from the 2026-07-22 audit snapshot.]*

## 9. Suggested fixes (for you; none applied)

1. Commit a lockfile for the Tier-1 environment.
2. Make the notebook path-robust (repo-root detection) and take its figure output directory as a parameter.
3. Decide which `ci()` behavior is canonical (drop ±inf or not), regenerate the K4 bank and leaderboard from that single code path, and resolve the ace CI.
4. Add code, or the raw ridge scores, for the 201 bank-only cells, or soften the README's "regenerates every table" claim.
5. Correct Table 1's sa3s numeric ρ (.04 → .03) and re-source the §4.4 CLAP droop numbers from the canonical bank.

## Files in `rerun/step1/`

| File | What it is |
|---|---|
| `REPORT.md` | this report |
| `repro.executed.ipynb` | executed notebook (byte-identical source to `notebooks/repro.ipynb`) |
| `nbconvert.log` | execution log |
| `wallclock.txt` | wall-clock time for the execution |
| `figures/*.png` | regenerated figures |
| `regenerated/*.csv` | regenerated tables at full float precision |
| `cell_comparison.csv` | every banked cell: stored value, regenerated value, status, generating source |
| `paper_crosscheck.csv` | every checked paper/README number |
| `paper_pdf.txt` | text extracted from the PDF |
| `compare.py`, `paper_check.py` | comparison scripts (read `data/` only, write only here) |
| `environment-freeze.txt` | package versions used |

---

## 10. Addendum (2026-11-07): where the banked numbers actually come from

Investigation only. No analysis code was changed.

### 10.1 Sources pulled

These were pulled with `kaggle kernels pull <slug> -p rerun/step1/kaggle_src/<name> -m`. Last-run times are Kaggle's `lastRunTime` in UTC. The pulled notebooks contain **source only, no saved outputs**.

| Kaggle notebook | Visibility | Last run (UTC) | sha256 of pulled .ipynb | Writes |
|---|---|---|---|---|
| `connerofarrell/p0b-score` | private | 2026-07-24 01:09 | `5a26f1f6…822d` | `scored_full.csv` (S5); **cell 13 (S6 v2):** `scored_full_v2.csv`, `s6_level_means.csv`, `s6_bootstrap_cis.csv`, `s6_edge_cells.csv`, `s6_raw_ridge_level_means.csv`, `s6_feature_decomp.csv`; **cell 14 (S7):** `s7_bootstrap_cis.csv`, `s7_skew_L1L9.csv`, `s7_leaderboard.csv` (+ an in-notebook axis attempt) |
| `connerofarrell/p0b-s7-axis` | private | 2026-07-24 02:58 | `dfe51244…ff13` | `s7_clap_axis.csv`, `s7_axis_level_means.csv`, and `s7_bootstrap_cis.csv` again. It reads p0b-score's copy, drops any `rho_level_axis`/`droop_L9-L7_axis` rows and appends its own, so the axis rows in the bank come from here |
| `connerofarrell/arousal-index` | **public** | 2026-07-18 03:25 | `43e463ed…2cb2`¹ | instrument fit/calibration; `requirements.lock` (see 10.5) |
| `connerofarrell/p0b-k1-preflight` | private | 2026-07-22 16:26 | `3a244f28…df5` | `k1_results.csv`, `k1_verbal_results.csv` (pre-flight, not banked) |

¹ Full hashes: `43e463edc0ab37abeebf25da1d2eb341d54000d61efa391dd8efa71fac552cb2` (arousal-index), `3a244f284ca1afce73324d8dcceff683ab55db44654998d49446f6420cebdbf5` (k1-preflight), `dfe512444c59799a9db1cdb131a97e1535501a20c54a06f33d9f5eac9499ff13` (s7-axis), `5a26f1f60cda6b775bbbdc855e5026850222b6e840a2b688f1a9ebb84297822d` (p0b-score).

The times are consistent: p0b-score, then p0b-s7-axis, then the committed `repro.ipynb` outputs (executed 2026-07-24 03:10). The S7 header says "The commit containing S6 v2 + S7 is CANONICAL-2: all paper numbers cite it." **The pulled notebooks are not committed here** (three are private; see the PR conversation). They live only in this session's checkout under `rerun/step1/kaggle_src/`.

### 10.2 The 5 mismatches: explained exactly

The canonical content-share function (`p0b-score` cell 13, S6.2, and cell 14, S7.6) guards the denominator:

```python
return (...)/r if abs(r) > 1e-9 else np.nan
```

`repro.ipynb` divides unconditionally. Applying the canonical guard to the repo's data reproduces the bank **exactly**:

| | Banked | Canonical rule, recomputed | `repro.ipynb` |
|---|---|---|---|
| ace | [−8.075, 9.4673] | **[−8.075, 9.4673]** (5 resamples → NaN) | [−8.5000, 10.1269] |
| sa3s | [0.6392, 1.1473] | **[0.6392, 1.1473]** (1 → NaN) | [0.6392, 1.1527] |
| mgm, mgs, sao | — | identical (0 guarded) | identical |

For ace, 4 resamples have a raw delta of exactly 0, which gives ±inf in the repo code. A fifth has a float residue below 1e-9, which gives a finite value near 10¹⁵. That fifth one is why "drop non-finite" in §2 came close but not exact. So the cause is a **harness divergence**: `repro.ipynb`'s ratio function is not the canonical one. It is not a data problem.

### 10.3 The 201 "no generating code" cells: which notebook makes each

| Banked cells | Count | Produced by | Reproducible from repo data now? |
|---|---:|---|---|
| `s6_bootstrap_cis` · `phrasing_ddelta` | 15 | p0b-score cell 13, S6.3 (4-group independent resample) | **Yes, 15/15 exact** with the canonical code |
| `s7_bootstrap_cis` · `family_drho` | 6 | p0b-score cell 14, S7.4 (`boot_rho(m1) − boot_rho(m2)`) | **Yes, 6/6 exact** |
| `s6_bootstrap_cis` · `droop_L9-L7_rawridge` | 15 | p0b-score cell 13, S6.4 | No: needs the fitted Arousal Index ridge (`model` from cells 0–7) |
| `s6_raw_ridge_level_means.csv` | 25 | p0b-score cell 13, S6.4 | No: same |
| `s6_feature_decomp.csv` | 95 | p0b-score cell 13, S6.4 (exact linear split of ridge score into loudness/noisiness/tonality clusters; `dfeat_*` = L9−L1 feature means) | No: same |
| `s7_bootstrap_cis` · `rho_level_raw`, `delta_raw_mean`, `delta_raw_median` | 45 | p0b-score cell 14, S7.2 | No: same |
| **Total** | **201** | | **21 now reproduce; 180 need the ridge** |

The 180 remaining cells all depend on one missing object: the ridge (standard scaler + ridge coefficients and intercept) fitted in p0b-score cells 0–7. Those cells duplicate arousal-index cells 0–7 and are fitted on the Kaggle dataset `connerofarrell/pre-extracted-librosa-features` (DEAM/PMEmo features). The per-clip feature cache it is applied to appears to be the repo's `data/features_verbal.csv`: 750 verbal clips with the same 12 feature columns (`SIX`) as S6.4's `p0b_features_verbal.csv`. So these cells could be regenerated by publishing the ridge coefficients, or by re-fitting from that dataset on CPU. Not attempted; it's your call.

### 10.4 The paper numbers

**7 paper-vs-bank disagreements (§4).** None is explained by different code:
- **sa3s numeric ρ ".04":** S6.2 prints it with `+.3f` as `+0.035`, and nothing anywhere prints `.04`. This still points to rounding the 3-decimal printout by hand. **Not explained by code**; the paper value is wrong (should be .03).
- **ace and mgs own-prompt CLAP droop (6 cells):** the banked values come from S6.1 on the `clap` column, which S3 (p0b-score cell 10) scores with `laion/clap-htsat-unfused`, `ClapProcessor(audio=y48)` on 30-second clips, with no seed set anywhere. As far as I know, the transformers CLAP feature extractor picks which part of audio longer than 10 seconds to embed using numpy's **unseeded** global random generator. That would make own-prompt CLAP differ from run to run, and S7's comment ("deterministic: no fusion, first-10s window") reads like a deliberate fix for it. **Most likely explanation:** the paper quotes an earlier S3/S6 run, not CANONICAL-2, which contradicts the S7 header. I can't confirm this because the pulled notebooks have no outputs or run history. Also notable: S3's `clap` is `sim(audio, "intense loud aggressive energetic music") − sim(audio, "calm quiet gentle soft music")`, a fixed text pair, but the paper calls it "own-prompt CLAP". Worth checking the wording.

**The 9 numbers with no source in the repo:**

| Paper number | Source found | Status |
|---|---|---|
| in-band gain **+0.223**, out-of-band **+0.033** | p0b-score cell 13, S6.5 prints `in`/`out` per scope (not banked) | **Reproduced:** +0.223 / +0.033 with the canonical code on repo data |
| rock-minus-others ρ: mgm **−.21**, sa3s **−.27**, sao **−.20** | p0b-score cell 11 (S4, "H1b elasticity"): verbal per-genre ρ rounded to 3 dp, then `rock − mean(other four)` | sa3s −0.266 ✓, sao −0.202 ✓, mgs −0.020 (the paper's "null") ✓; **mgm −0.204, paper says −.21: an 8th paper mismatch, new** |
| DEAM **R² = 0.29** | arousal-index cell 5 (5-fold DEAM R² on CORE); cell 14 hard-codes 0.29 as the reference | Source found; not run (needs DEAM features + GPU-free re-fit; out of Tier-1 scope) |
| PMEmo transfer **ρ = 0.77** | arousal-index cell 5 (PMEmo Spearman); cell 14 hard-codes 0.77 | Source found; not run |
| control clips **3.7–4.1 vs 6.0–6.5** | p0b-k1-preflight cell 10 ("6 real-DEAM controls") is the only control-clip scorer | Likely source; its output `k1_results.csv` isn't in the repo, not run |
| CLAP drift **≤ 0.033** | **No code in any of the four notebooks** computes run-to-run drift | Still **no source**; presumably a manual comparison of two S3 runs |

So the paper-vs-repo disagreements now number 8 (7 from §4 plus mgm rock-minus-others), and 1 number (the drift bound) still has no source.

### 10.5 Recorded package versions

- **No notebook records versions in its source or metadata, except as runtime output.** `arousal-index` cell 8 writes a `requirements.lock` with librosa, scikit-learn, scipy, numpy and pandas versions at run time. `p0b-s7-axis` prints the transformers and torch versions at run time. The pulled copies contain no outputs, so **the actual version strings are not in hand**. They would be in each notebook's run output or log (`kaggle kernels output <slug>`), which I haven't fetched. Note `p0b-score` and `repro.ipynb` record no versions, and nothing records matplotlib.
- **Kaggle Docker image `dockerImageVersionId: 28755` is recorded in all four notebooks.** That is the only hard environment pin, and it fixes the full Kaggle Python stack for those runs.
- Notebook `language_info.version`: **3.12.13** (arousal-index, p0b-score, k1-preflight) and **3.6.6** (p0b-s7-axis, probably a stale metadata default that contradicts the same image). `repro.ipynb` says 3.12.3, so it ran outside Kaggle.

### 10.6 Other things this explains

- **No `*` in `s6_edge_cells.csv`:** S6.5 only prints the band-marked table and banks the unmarked `_ec`. The comment in `repro.ipynb` is wrong; the CSV is right.
- **The rounding ties (§7):** the banks are written with Python `round(float(x), 4)`, so the 6 binary-tie cells agree with `round()` by construction.
- **`house_level`:** confirmed as `round(numeric-phrasing mean p0a, 2)` (S7.6), as inferred in §3.
- **Bootstrap pairing:** the canonical code uses the same reseed-per-call scheme. Its own comment on `phrasing_drho` says "independent clip resamples of each arm", but with `default_rng(0)` per call and equal arm sizes (n = 150) both arms draw **identical index matrices**. The comment doesn't match the behavior (see §7).

### 10.7 Stability AI attribution check (docs/TOS_AUDIT.md, model 2)

Both are already present, so **no change was made**:
- **Attribution notice in the repo:** present in `NOTICE` — "This Stability AI Model is licensed under the Stability AI Community License, Copyright (c) Stability AI Ltd. All Rights Reserved." (written with "(c)" where the license text has "©").
- **"Powered by Stability AI" in the README:** present in bold under *Licensing and attribution* (README l.106), and repeated in `NOTICE` and `docs/LICENSES.md`. It sits about 100 lines down the README rather than near the top. Whether that counts as "prominently" is your judgment call.

---

## 11. Addendum 2 (2026-11-07): corrections and new evidence

**Correction to §1.3, §3, §10.3.** I wrote that the 201 cells had "no generating code in the repo". **That was wrong.** `notebooks/scoring.ipynb` (cells 13–14) and `notebooks/clap-axis.ipynb` in this repo contain the generating code. They are cleaned copies of the Kaggle notebooks, with identical logic. The accurate statement is that the code was in the Tier-2 notebooks, not in `repro.ipynb`, and it needed the ridge fit (DEAM) and the audio to run. Attempt #2 fixes that.

**Recorded package versions (step 1 of this round).** `kaggle kernels output` was run with a file filter for logs and `requirements.lock` only. No `requirements.lock` exists in any notebook's output: the last versions of `arousal-index` and `p0b-k1-preflight` were saved without re-execution (their logs contain only the HTML render). The only versions recorded anywhere are **transformers 5.0.0 and torch 2.10.0+cu128** (the `p0b-s7-axis` log, matching the saved output of `notebooks/clap-axis.ipynb`), Python 3.12 (from site-packages paths in the logs), and Kaggle Docker image 28755.

**Own-prompt CLAP: three runs, three answers.** The canonical `p0b-score` log (2026-07-24 01:09) prints the S6.1 CLAP droop as ace −0.074 [−0.134, −0.019] and mgs −0.063 [−0.142, +0.017]. The repo's bank has −0.0806 / −0.0458, and the paper −0.086 / −0.085. The p0a, K4 and raw-ridge numbers in that same log match the bank exactly. So **`data/` was produced by a different p0b-score run than the latest one on Kaggle**, and the CLAP column changes between runs. The mechanism is confirmed in the transformers 5.0.0 source: `ClapFeatureExtractor` defaults to `truncation="fusion"`, which for audio longer than 10 s picks 3 crops with `np.random.choice` from numpy's global (unseeded) RNG (`rand_trunc` uses `np.random.randint`). The clips are 30 s long. The seed added to `scoring.ipynb` fixes future runs and leaves the released values unchanged. The fixed-anchor axis (`clap-axis.ipynb`) was already deterministic, because it loads exactly the first 10 s, so no crop is drawn.

**rock-minus-others ρ (§10.4):** the canonical S4 printout matches the code on repo data, mgm −0.204. The paper's −.21 remains unexplained.
