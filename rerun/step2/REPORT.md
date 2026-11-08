# P0b re-run · Step 2 · Tier-2 rescoring — **FAIL on the original determinism gate; ACCEPTED 2026-11-08 under the revised standard (§10)**

**Verdict: FAIL**, by the gate as specified ("run one model twice to confirm identical scores"). Rescoring all 300 sao clips a second time in the same Kaggle session gave bit-identical Arousal Index values, but **LUFS differed on 67/300 clips by exactly 1 ULP** (one unit in the last binary place; max 3.6e-15 dB), and CLAP on 173/300 (`clap`) and 163/300 (`clap_n`) clips, by at most about 1e-16 absolute. Nothing was changed to make it pass.

Everything that compares this run with the banked corpus passes:

| Check | Result |
|---|---|
| Audio audit, every clip, before any score is computed | **PASS** 1,500/1,500 |
| Arousal Index `p0a` vs `data/scored_full_v2.csv`, bit-identical | **PASS** 1,500/1,500 |
| Arousal Index on normalized copies `p0a_n`, bit-identical | **PASS** 1,500/1,500 |
| LUFS vs bank, bit-identical (stored at 11–15 decimals) | **PASS** 1,500/1,500 |
| Per-clip features `features_verbal.csv` (750 × 12), bit-identical | **PASS** |
| Non-CLAP banked statistics re-made by this run (strict rule) | **PASS** 997/997 vs the original values (details in §4) |
| Same-session repeat of one model (sao), bit-identical | **FAIL**: p0a 300/300, p0a_n 300/300, LUFS 233/300, clap 127/300, clap_n 137/300 |
| CLAP verdicts (droop CI sign) under the new seeded baseline | **unchanged** 5/5 |
| HF token needed | **No**: no 401 or gated error; nothing in the scoring path is gated |

Run: Kaggle notebook `energy-audit-rescore` v1 (private), Tesla T4, pinned to the canonical run's Docker image digest. Pushed 2026-11-07 23:49 UTC, complete by 00:23 UTC. Kernel run time 1,827 s (30.5 min); scoring rate about 103 clips/min.

---

## 1. What ran

`notebooks/scoring.ipynb` from branch `rerun/step1-tier1` (CLAP seeded: `np.random.seed(0)` before every `ClapProcessor(audio=…)` call), unchanged, plus three inserted cells: an environment record, the audio audit (after S1), and the determinism rescore (after S3). The pushed notebook is `kaggle_push/energy-audit-rescore.ipynb`.

Inputs are the same sources as the canonical `p0b-score` run: the three generation notebooks' audio tars (A: mgs + mgm, C: sa3s, D: ace), the loose sao dataset, DEAM, PMEmo, and the pre-extracted DEAM/PMEmo features. Internet was on, for `pip install pyloudnorm` and the public `laion/clap-htsat-unfused` checkpoint.

## 2. Environment (recorded inside the run, `kaggle_out/env.json`)

| | |
|---|---|
| Docker image | pinned to `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…7d461` (the canonical run's image) |
| Python | 3.12.13 |
| torch / CUDA / cuDNN | 2.10.0+cu128 / 12.8 / 9.10.2 (recorded as 91002) |
| transformers / tokenizers / huggingface_hub | 5.0.0 / 0.22.2 / 1.11.0 |
| numpy / pandas / scipy / scikit-learn | 2.0.2 / 2.3.3 / 1.16.3 / 1.6.1 |
| librosa / soundfile / soxr / numba | 0.11.0 / 0.13.1 / 1.0.0 / 0.60.0 |
| pyloudnorm, audioread | installed and working; neither exposes `__version__`, so no version string was recorded |
| GPU | Tesla T4 |

transformers 5.0.0 and torch 2.10.0+cu128 match the versions recorded for the canonical run, which is consistent with the image pin having taken effect. **These are the first recorded versions of the Tier-2 numeric stack** (numpy, pandas, scipy, scikit-learn, librosa); `requirements-scoring.txt` on branch `rerun/step1-tier1` currently marks them unrecorded.

## 3. Audio audit (`kaggle_out/audio_audit.csv`, every clip)

| model | clips | sample rate | channels | duration | format | min peak |
|---|---:|---:|---:|---:|---|---:|
| ace | 300 | 48,000 Hz | 2 | 30.00 s | PCM_16 | 0.891 |
| mgm | 300 | 32,000 Hz | 1 | 29.94 s | PCM_16 | 0.074 |
| mgs | 300 | 32,000 Hz | 1 | 29.94 s | PCM_16 | 0.129 |
| sa3s | 300 | 44,100 Hz | 2 | 30.00 s | PCM_16 | 0.363 |
| sao | 300 | 44,100 Hz | 2 | 30.00 s | PCM_16 | 0.304 |

Assertions applied:
- 1,500 unique files, 300 per model.
- One sample rate and one channel count per model.
- The documented formats: MusicGen 32 kHz mono; Stable Audio Open 44.1 kHz stereo.
- Every duration between 20 and 48 s. These bounds are my choice: requested about 30 s, and Stable Audio Open's maximum is 47 s.
- Finite samples, no silent clip.

All passed. Every model is uniform, with no spread in duration at 2-decimal precision. ace's high minimum peak (0.89) is consistent with its documented −1 dB peak normalization.

## 4. Per-clip comparison with the bank

**Arousal Index and LUFS: exact.** `p0a`, `p0a_n` and `lufs` are bit-identical to `data/scored_full_v2.csv` on all 1,500 clips, with maximum difference 0. Full per-clip table: `clip_comparison.csv`.

**Banked statistics re-made by this run** (S6/S7 inside `scoring.ipynb`) vs `data/`, strict rule:

| file | match | CLAP-dependent (new values) | not produced by this notebook | mismatch |
|---|---:|---:|---:|---:|
| s6_bootstrap_cis | 183 | 15 | 0 | 0 |
| s6_level_means | 150 | 100 | 0 | 0 |
| s6_edge_cells | 125 | 0 | 0 | 0 |
| s6_feature_decomp | 95 | 0 | 0 | 0 |
| s6_raw_ridge_level_means | 8 vs current `data/` (**25/25 vs the original bank**) | 0 | 0 | see below |
| s7_bootstrap_cis | 69 | 0 | 45 (the 15 CLAP-axis rows: `clap-axis.ipynb` makes them; `scoring.ipynb`'s in-notebook axis attempt crashes and is skipped, exactly as in the canonical run) | 0 |
| s7_leaderboard | 50 | 0 | 0 | 0 |
| s7_skew_L1L9 | 300 | 0 | 0 | 0 |

**Important for Step 1: the re-bank of `data/s6_raw_ridge_level_means.csv` should be reverted.** On the pinned original image, this run reproduces the **original** 17-digit raw-ridge means exactly, 25/25. That's the file as it was before commit `0ea95aa`. Against the re-banked file it differs at about 1e-15 on 17 cells. So the original values were canonical and reproducible on their own image, and the 1-ULP gap in Step 1 came from the container's numeric stack (numpy 2.5.4 / scikit-learn 1.9.1 vs the image's 2.0.2 / 1.6.1), not from the bank. I recommended the re-bank, and it was the wrong call. A clean fix: restore the original file, and regenerate `data/arousal_index_ridge.json` on the pinned image, which needs one small CPU run of `scripts/fit_arousal_ridge.py` there, so Tier 1 matches strictly without touching canonical data. **Not done here**; it's your decision.

## 5. Determinism (same-session repeat, sao, `kaggle_out/determinism_sao.csv`)

| column | identical | differing clips | max \|run1 − run2\| | max size in ULPs |
|---|---:|---:|---:|---:|
| p0a | 300/300 | 0 | 0 | 0 |
| p0a_n | 300/300 | 0 | 0 | 0 |
| lufs | 233/300 | 67 | 3.6e-15 dB | 1 |
| clap | 127/300 | 173 | 9.7e-17 | 247 |
| clap_n | 137/300 | 163 | 8.3e-17 | 12 |

How to read it:
- **The first pass reproduces across sessions; the second pass inside a session doesn't, quite.** The first-pass LUFS matched the July bank bit-for-bit on all 1,500 clips, yet the in-session repeat moved 67 sao clips by 1 ULP. So LUFS is reproducible run-to-run (same image, fresh session), but not repeat-to-repeat within one process. LUFS is pure CPU (soundfile decode, then pyloudnorm's filters and means), so this isn't the GPU. **Hypothesis, not verified:** numpy's SIMD reductions can round differently depending on an array's memory alignment, which changes between the first and second allocation in a long-running process.
- **CLAP differences are GPU arithmetic, not crops.** About 1e-16 absolute is floating-point noise in cuBLAS/cuDNN kernels. The large ULP count for `clap` (247) is because some `clap` values are near zero, where one ULP is tiny. The seed did its job: the run-to-run CLAP drift that was 0.01–0.04 at cell-mean level (§6) is now at the 1e-16 level within a session. Making CLAP bit-exact would need `torch.use_deterministic_algorithms(True)` and deterministic cuBLAS settings. Not attempted.
- **Arousal Index is fully deterministic** (rounded to 0.1, CPU).

## 6. CLAP: the new seeded baseline vs the old bank

The old bank's CLAP came from an unseeded run, so these shifts measure that run's random crops, not a change in the models. **This run is the new CLAP baseline.**

| model | column | mean, old bank | mean, new | mean shift | max \|clip change\| | max \|cell-mean change\| | Spearman old↔new |
|---|---|---:|---:|---:|---:|---:|---:|
| ace | clap | 0.1271 | 0.1254 | −0.0018 | 0.356 | 0.0152 | 0.878 |
| mgm | clap | 0.2774 | 0.2889 | +0.0116 | 0.479 | **0.0368** | 0.879 |
| mgs | clap | 0.2155 | 0.2243 | +0.0089 | 0.553 | 0.0250 | 0.893 |
| sa3s | clap | 0.2733 | 0.2750 | +0.0018 | 0.149 | 0.0190 | 0.978 |
| sao | clap | 0.2792 | 0.2688 | −0.0104 | 0.215 | 0.0214 | 0.951 |
| ace | clap_n | 0.1686 | 0.1785 | +0.0099 | 0.562 | **0.0348** | 0.840 |
| mgm | clap_n | 0.3263 | 0.3355 | +0.0093 | 0.516 | **0.0421** | 0.852 |
| mgs | clap_n | 0.2798 | 0.2894 | +0.0096 | 0.489 | 0.0303 | 0.892 |
| sa3s | clap_n | 0.3048 | 0.3057 | +0.0010 | 0.144 | 0.0088 | 0.975 |
| sao | clap_n | 0.2997 | 0.2970 | −0.0027 | 0.167 | 0.0126 | 0.939 |

Cell = model × phrasing × level (30 clips), the README's drift metric. Full cell table: `clap_cell_means.csv`.

- **The README's "CLAP run-to-run drift ≤ 0.033 at the cell-mean level" is exceeded:** 0.0368 (mgm `clap`), and 0.0348 / 0.0421 (ace / mgm `clap_n`). This agrees with Step 1's finding that the paper's mgs droop differed from the bank by 0.039.
- **Single clips move a lot** (up to 0.55), and the old and new clip values agree only at ρ 0.84–0.98. Random 10-second crops of 30-second clips are a large source of variance per clip, so per-clip own-prompt CLAP from the old bank shouldn't be cited.
- **No CLAP verdict changes** (`clap_droop_old_vs_new.csv`):

| model | droop L9−L7, old [CI] | new [CI] | verdict |
|---|---|---|---|
| ace | −0.0806 [−0.1320, −0.0269] | −0.0706 [−0.1248, −0.0151] | drop, CI excludes 0 (both) |
| mgm | +0.0930 [+0.0150, +0.1627] | +0.0958 [+0.0231, +0.1641] | rise, CI excludes 0 (both) |
| mgs | −0.0458 [−0.1202, +0.0302] | −0.0341 [−0.1071, +0.0417] | CI includes 0 (both) |
| sa3s | −0.0013 [−0.0861, +0.0852] | −0.0015 [−0.0848, +0.0824] | CI includes 0 (both) |
| sao | +0.0670 [−0.0004, +0.1358] | +0.0662 [−0.0007, +0.1360] | CI includes 0 (both) |

The paper's §4.4 CLAP numbers (ace −0.086 [−0.138, −0.031]; mgs −0.085 [−0.170, +0.004]) now differ from three runs: the July bank, the July latest Kaggle run and this seeded run. The new seeded values are the ones to cite going forward.

## 7. Three clips to listen to

Picked uniformly at random from all 1,500 with `numpy.random.default_rng(20261107)`, **before** this run, so no score influenced the choice:

| clip | prompt | where the audio is |
|---|---|---|
| `sa3s__rock_L7_num__s1.wav` | "instrumental rock track, energy level 7 out of 11" | `p0b-generate-c` output, `audio_C.tar` |
| `ace__rock_L1_verb__s1.wav` | "instrumental rock track, very calm" | `p0b-generate-d` output, `audio_D.tar` |
| `mgm__edm_L1_num__s5.wav` | "instrumental electronic dance track, energy level 1 out of 11" | `p0b-generate-a` output, `audio_A.tar` |

Their scores are in `clip_comparison.csv`, if you want to listen blind first.

## 8. Files

All in `rerun/step2/`:

| path | what it is |
|---|---|
| `REPORT.md` | this report |
| `kaggle_push/energy-audit-rescore.ipynb` | the notebook as pushed |
| `kaggle_out/` | every CSV the run wrote, plus `env.json` |
| `compare_step2.py` | the comparison script |
| `clip_comparison.csv` | per-clip old vs new values |
| `clap_shift_by_model.csv` | CLAP shift per model |
| `clap_cell_means.csv` | CLAP cell means, old vs new |
| `clap_droop_old_vs_new.csv` | CLAP droop CIs and verdicts, old vs new |
| `bank_comparison.csv` | every banked cell from this run vs `data/` |
| `status.log` | Kaggle status checks |

Not committed:
- the Kaggle run log and `kernel-metadata.json` (they contain your Kaggle username in input paths and slugs)
- any audio

## 9. Not done / your decisions

1. **Determinism gate:** make CLAP and the in-session LUFS repeat bit-exact, or accept the last-binary-digit noise? Bit-exact would mean deterministic torch/cuBLAS flags, plus a CPU-side check of the LUFS alignment hypothesis.
2. **Revert the Step-1 re-bank** of `s6_raw_ridge_level_means.csv` and regenerate the ridge JSON on the pinned image (§4).
3. **Adopt this run's CLAP** (`kaggle_out/scored_full_v2.csv`, `clap`/`clap_n`) as the banked CLAP in `data/`? That changes committed data and the CLAP-dependent banks, so I haven't done it.
4. **README CLAP drift bound** (≤ 0.033) is contradicted by the data (0.042). Wording change pending your call.

---

## 10. Decisions applied (2026-11-08)

1. **Determinism standard, accepted as revised.** Fresh runs on the pinned image must match the bank exactly: **achieved** (p0a, p0a_n and LUFS 1,500/1,500 bit-identical; non-CLAP banks 997/997). Repeats *within one session* may differ by at most 1e-12 relative. The measured maximum was about 1e-15 for LUFS (1 ULP) and about 1e-16 absolute for CLAP, which is within the limit. This is now stated in the README under Tier 2.
2. **Re-bank reverted.** `data/s6_raw_ridge_level_means.csv` is byte-identical to the original again. `data/arousal_index_ridge.json` is refit on the pinned image (CPU run `energy-audit-ridge-refit`). **The Tier-1 strict check on that image is 2,688/2,707, not 2,707**: 8 raw-ridge cells off by ≤ 1 ULP and 11 feature-decomp cells by 1–225 ULP. The details and the hypothesis are in the Step 1 report's "Current status". The changelog line in the README is corrected.
3. **Seeded CLAP saved; banked CLAP untouched.** This run's `clap`/`clap_n` are in `data/clap_seeded_2026-11-07.csv` (1,500 rows; round-trip checked exact). The CLAP-dependent tables derived from it are in `rerun/step2/clap_seeded_derived/`. **The paper's CLAP numbers (§4.4 droop, and every CLAP-dependent figure) are pending a RESEARCH decision** on which run to cite: the July bank, or this seeded run.
4. **README drift claim replaced.** "CLAP drift ≤ 0.033" now reads as measured: unseeded cell-mean drift up to 0.042, scoring now seeded.
5. **Listening pack.** Private Kaggle notebook `energy-audit-listen-clips` (pinned image, CPU) extracts only the three clips and writes one spectrogram PNG per clip. Each PNG has three panels: the file as generated (mono sum, native rate), the scorer's input (mono, 22.05 kHz, first 30 s), and that input after −14 LUFS normalization.
   - **Verified sources:** `sa3s__rock_L7_num__s1.wav` from `audio_C.tar`, `ace__rock_L1_verb__s1.wav` from `audio_D.tar`, `mgm__edm_L1_num__s5.wav` from `audio_A.tar`.
   - **Download:** the three WAVs and PNGs are in that notebook's Output tab. They are **not committed** here: MusicGen-derived audio, and images made from it, stay out of this public repo under the CC-BY-NC terms in `docs/TOS_AUDIT.md`. The notebook source is `kaggle_listen/energy-audit-listen-clips.ipynb`.
