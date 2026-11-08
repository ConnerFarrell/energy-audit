# P0b re-run · Step 3 · Comparison memo: Magenta RealTime 2 (mrt2) against the paper's five models

*2026-11-08. Metrics only. Nothing here describes how any clip sounds, and no LLM judged audio. Source run:
private Kaggle notebook `energy-audit-step3-score` v1 (Tesla T4, pinned image `sha256:37c64f7d…7d461`),
which ran the Step 2 scoring notebook unchanged, plus three added cells. Every number below comes from that
run's tables, which are committed in `score_out/`; the comparisons are in `compare_step3.py` and
`compare_step3.log`.*

## 1. Did the existing 1,500 clips reproduce exactly? Yes.

| check (fresh run on the pinned image, against `data/`) | result |
|---|---|
| Arousal Index per clip (`p0a`) | **1,500/1,500 identical** |
| Arousal Index on loudness-normalized copies (`p0a_n`) | **1,500/1,500 identical** |
| LUFS per clip | **1,500/1,500 identical** |
| Banked non-CLAP cells (every numeric cell in `s6_*`, `s7_*`, `features_verbal`) | **10,081/10,081 within 1e-12 relative** |
| Same-session repeat (sao, 300 clips) | Arousal Index, CLAP 300/300 identical; LUFS 292/300 identical, the rest ≤ 1.9e-16 relative (inside the accepted 1e-12) |

The pass rule holds. Not part of it, by earlier decision:
- **CLAP:** the CLAP-dependent cells match the old, unseeded bank in only 8/120 cells, as expected (Step 2,
  decision 3). They match Step 2's seeded run in 120/120 cells, and per clip in 1,500/1,500 (difference 0).
- **CLAP axis:** the 60 fixed-anchor CLAP-axis cells come from `clap-axis.ipynb`, which this run didn't
  execute.

## 2. Leaderboard: mrt2 beside the paper's five (verbal instructions)

| model | house level | ρ(level, score) [95% CI] | median Δ(L9−L1) [CI] | content share [CI] |
|---|---|---|---|---|
| **mrt2** (google/magenta-realtime-2, `mrt2_base`) | 5.25 | **.890** [.854, .915] | **2.80** [2.80, 2.80] | **.89** [.83, .94] |
| mgm (musicgen-medium) | 5.83 | .538 [.414, .644] | 1.30 [0.50, 2.10] | .76 [.63, .91] |
| sao (stable-audio-open-1.0) | 6.04 | .504 [.367, .619] | 1.30 [0.55, 2.00] | .59 [.50, .69] |
| mgs (musicgen-small) | 5.81 | .388 [.247, .512] | 2.05 [0.80, 2.40] | .76 [.47, .99] |
| sa3s (stable-audio-3-small-music) | 5.80 | .384 [.223, .525] | 1.60 [0.55, 2.20] | .82 [.64, 1.15] |
| ace (ACE-Step 1.5 turbo) | 4.35 | .328 [.167, .476] | 0.10 [−0.05, 0.70] | undefined |

The five banked rows are digit-identical to `data/s7_leaderboard.csv`. Same definitions as the paper:
2,000-resample clip-level bootstrap, percentile CIs, every bootstrap re-seeded at 0. House level is the mean
Arousal Index under the numeric phrasing.

## 3. What's different about mrt2

- **Strongest compliance, by a wide margin.** mrt2's verbal ρ is .890 [.854, .915]. Its CI lies entirely
  above every other model's CI (the highest of the five upper bounds is mgm's .644). Its mean Arousal Index
  moves 2.68 [2.50, 2.79] from "very calm" to "extremely intense", against 0.18–1.28 for the five.
- **The widest range on the scale.** mrt2 reaches the lowest calm-cell mean of the six (L1 3.79; next lowest
  ace 4.16) and the highest intense-cell mean (L9 6.48; next highest mgm 6.44).
- **Numeric phrasing still does nothing**, as for the five: ρ −.002 [−.182, .170], cell means 5.14–5.35. The
  verbal-minus-numeric gap is the largest of the six (Δρ .892 [.718, 1.073]; next mgm .500).
- **Mostly content, and also the largest loudness swing.** 89% [83%, 94%] of its movement survives −14 LUFS
  normalization. At the same time its verbal L9−L1 loudness change is the largest of the six: +11.2 LU
  (−28.6 → −17.4 LUFS), against +6.3 for sao and −0.4 to +2.3 for the others. The feature decomposition puts
  its loudness share at 0.195, with noisiness 0.310 and tonality 0.496.
- **No top-of-scale drop**, unlike ace and mgs. The L9−L7 change is +0.20 [+0.07, +0.36] on the Arousal
  Index, +0.40 [+0.15, +0.63] on the raw ridge, and +0.100 [+0.047, +0.152] on seeded CLAP. All three CIs
  exclude zero, upward. The genre-plausibility gap is positive and in the same range as the five: +0.174
  [+0.069, +0.298] (pooled six-model +0.188). Its calm-rock cell (rock L1, verbal) is 3.70, against 5.83–6.37
  for sao and mgm.

## 4. Caveats

1. **One near-silent clip; nothing excluded.** `mrt2__edm_L1_verb__s0` ("instrumental electronic dance track,
   very calm") measures −62.4 LUFS and needs **+48.4 dB** to reach −14 LUFS. It is the only clip over the
   +40 dB line. The next quietest needs +32.3 dB, and the quietest of the paper's 1,500 clips needed +25.0 dB.
   It stays in both lenses, per the rule.
2. **Excluded clips: none.** No clip measured −inf LUFS, so the normalized-lens exclusion removed 0 clips
   (0 in every model × phrasing × level cell; `mrt2_exclusions_by_cell.csv`). No scorer errors.
3. **CLAP numbers come from the seeded run, not the old bank.** All CLAP values in this memo, for all six
   models, are seeded scoring (identical to Step 2's seeded baseline). The paper's printed CLAP numbers
   (§4.4) come from the old unseeded bank, and which run the paper cites is still a pending RESEARCH
   decision.
4. **No fixed-anchor CLAP axis for mrt2.** That replication (`clap-axis.ipynb`) was not run in this step.
5. **The median Δ CI is degenerate:** [2.80, 2.80]. All 2,000 resamples gave the same median, because the
   calibrated scale is coarse. Mean Δ 2.68 [2.50, 2.79] and raw-ridge median Δ 2.89 [2.45, 3.20] carry the
   uncertainty instead.
6. **Generation differences.** mrt2 output is 48 kHz stereo, exactly 30.00 s. Its seed sets both the
   sampler key and the text-mapper noise, because the library exposes no seed and the harness sets both. The
   scorer front end is the same for every model: mono, 22.05 kHz, first 30 s.

## 5. What failed or looks off

1. **The near-silent clip scores high on the Arousal Index.** `mrt2__edm_L1_verb__s0` is a "very calm" clip
   at −62.4 LUFS, yet it scores **6.0** raw and 6.0 normalized. That's above its own cell's mean (edm L1
   verbal, 4.05) and above every other mrt2 L1 verbal genre cell (3.70–3.82). The scale was calibrated on
   DEAM, where no clip is this quiet. This is a measurement concern about the instrument at extreme low
   level, not a statement about the clip. It works *against* mrt2's compliance numbers, which are strong
   regardless.
2. **The 6-model re-run is not bit-identical on two feature-derived tables.** Every leaderboard,
   bootstrap-CI, level-mean, edge-cell and skew cell for the five banked models came out identical in the
   6-model re-execution. But:
   - `s6_feature_decomp` differs by up to 5.3e-6 relative, and `s6_raw_ridge_level_means` by up to 2.4e-9.
   - Cause: in the main run the verbal features are extracted fresh and carry float32 precision (the banked
     file shows 8 significant digits). In the re-run, the same 750 rows come back from the CSV cache as
     float64. Clip-level features differ by at most 2.5e-16 relative.
   - Effect: this is a precision effect of the cache path, not of adding mrt2. mrt2's own decomposition and
     raw-ridge values carry the same ~1e-6 relative imprecision, invisible at the 3-decimal precision
     reported.
3. **Generation needed a second run.** Version 1 halted at 66/300 on my original −60 dBFS silence gate (the
   clip in caveat 1). Under decision A it was kept. The resumed corpus is bit-identical to an unbroken run:
   the regenerated clip matches the rejected one exactly. Details: `RUN_MANIFEST.md`.
4. **Pooled rows changed, as expected.** Two pooled rows changed because mrt2 is now included: the
   genre-plausibility gap went from +0.190 [0.096, 0.279] to +0.188 [0.098, 0.274], and the same gap with rock
   excluded went from +0.217 to +0.206. The two family contrasts (mgm−mgs, sa3s−sao) are unchanged. The
   paper's pooled numbers are the five-model values in `data/`.

## Files

`score_out/` holds the run's tables: the main run at the top level, the 6-model re-execution in `six/`, and
the mrt2 scores and flags in `scores_mrt2.csv`, `scores_mrt2_raw.csv`, `mrt2_clip_flags.csv` and
`mrt2_exclusions_by_*.csv`. The other files:

- `compare_step3.py`, `compare_step3.log`: the comparisons.
- `bank_comparison.csv`: every banked cell, main run vs `data/`.
- `six_vs_main.csv`: every five-model cell, 6-model run vs main run.
- `kaggle_score/energy-audit-step3-score.ipynb`: the notebook as pushed.

No audio is committed. The Kaggle log is not committed either, because it contains the account name.
