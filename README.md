# Do Text-to-Music Models Take Direction?
**An instrumented audit of energy-instruction compliance in five open text-to-music models (2023–2026).**

When you ask a text-to-music model for a *calm* folk track or an *extremely
intense* EDM track, does the audio actually move? We generated 1,500 clips
across a frozen matrix — 5 models × 5 genres × 5 requested energy levels ×
2 phrasings (numeric "energy level L out of 11" vs. a verbal ladder) × 6
seeds — and scored every clip with three instruments: an interpretable,
calibrated arousal model (the Arousal Index), integrated loudness (LUFS),
and CLAP text–audio similarity, each also applied to loudness-normalized
copies.

## Headline findings

1. **Numeric instructions do nothing.** Every model ignores "energy level L
   out of 11" (all monotonicity CIs cover zero); each sits at a fixed
   "house level."
2. **Verbal instructions work — narrowly.** All five models become reliably
   monotonic under the verbal ladder, but the median achieved movement from
   "very calm" to "extremely intense" spans **0.1–2.05 points of an
   11-point scale**. Replicated on an independent fixed-anchor CLAP axis
   (verbal ρ .27–.60, all CIs excluding zero; numeric flat; clip-level
   agreement with the Arousal Index ρ = .70).
3. **It's mostly the music, partly the volume knob.** 59–82% of measured
   compliance survives loudness normalization; a feature-level decomposition
   attributes the movement mainly to noisiness and tonality change. The most
   loudness-reliant model (Stable Audio Open, 38% loudness share, +6.3 dB
   from calm to intense) is exactly the one that loses the most under
   normalization — two independent methods agree.
4. **Compliance is gated by genre plausibility.** Achieved movement
   concentrates on requests inside each genre's plausible energy band
   (pooled in-band vs. out-of-band gain +0.223 vs. +0.033 per requested
   unit). Models cannot make *calm rock* (rock sits at ceiling even at
   level 1) and cannot make *extremely intense folk*.
5. **The top of the scale saturates.** One model's energy *decreases* from
   level 7 to level 9 on all three instruments (ACE-Step 1.5); a second
   (MusicGen-small) shows the same reversal on the pre-calibration scale.

## Leaderboard (verbal instruction compliance)

| model | checkpoint | ρ(level, score) [95% CI] | median Δ(L9−L1) [CI] | content share [CI] |
|---|---|---|---|---|
| mgm  | facebook/musicgen-medium | **.54** [.41, .64] | 1.30 [0.50, 2.10] | .76 [.63, .91] |
| sao  | stabilityai/stable-audio-open-1.0 | **.50** [.37, .62] | 1.30 [0.55, 2.00] | .59 [.50, .69] |
| mgs  | facebook/musicgen-small | **.39** [.25, .51] | **2.05** [0.80, 2.40] | .76 [.47, .99] |
| sa3s | stabilityai/stable-audio-3-small-music | **.38** [.22, .53] | 1.60 [0.55, 2.20] | .82 [.64, 1.15] |
| ace  | ACE-Step/Ace-Step1.5 (turbo) | .33 [.17, .48] | 0.10 [−0.05, 0.70] | undefined |

Content share = fraction of compliance surviving −14 LUFS normalization
(1.0 = pure musical change, 0 = pure volume knob). All CIs: 2,000-resample
clip-level bootstrap, percentile method, fixed seed — every number
regenerates digit-for-digit from the released CSVs.

## Reproduce

**Tier 1 — statistics (minutes, no GPU).** `repro.ipynb` regenerates every
table, CI, and figure in the paper from `data/*.csv`.

**Tier 2 — scoring (Kaggle, ~1 h GPU).** `notebooks/scoring.ipynb` re-runs
the full three-instrument scoring stack over the released audio corpus and
re-banks all CSVs. Arousal Index and LUFS reproduce bit-for-bit; CLAP shows
documented run-to-run drift ≤ 0.033 at the cell-mean level (verdicts stable;
the fixed-anchor CLAP axis in `data/s7_*.csv` is the deterministic
replacement).

**Tier 3 — generation.** Prompt set (`data/prompts.csv`), model revisions,
and generation harness notes in `docs/MODELS.md`.

## Repository layout

```
data/       scored_full_v2.csv (per-clip scores, raw + normalized lenses),
            features_verbal.csv (per-clip acoustic descriptors),
            s6_*.csv, s7_*.csv (banked analyses), prompts.csv
notebooks/  scoring.ipynb (scoring stack), repro.ipynb (statistics)
figures/    compliance_curves_ci.png, per_genre_curves.png, skew_L1L9.png
docs/       MODELS.md (roster, revisions), TOS_AUDIT.md (license audit)
paper/      main.tex, refs.bib (workshop submission source)
```

## Audio exhibits

Short audio exhibits for the failure gallery will be linked here upon
de-anonymization; they are not stored in this repository. MusicGen-derived audio is CC-BY-NC and
appears only in non-commercial research contexts.

## Licensing and attribution

- Analysis code in this repository: MIT.
- **Powered by Stability AI.** Portions of the audited corpus were generated
  with Stability AI models under the Stability AI Community License; see
  `NOTICE`. Stable Audio 3 Small Music additionally redistributes a T5Gemma
  text encoder under the Gemma Terms of Use.
- MusicGen checkpoints: code MIT, weights CC-BY-NC-4.0 (Meta); cite
  Copet et al., 2023 (arXiv:2306.05284). MusicGen-derived audio is used
  non-commercially only.
- ACE-Step 1.5: MIT; cite arXiv:2602.00744.
- Evaluation corpora: DEAM (Aljanaki et al., 2017), PMEmo (Zhang et al.,
  2018) — used for instrument training/controls only; no audio
  redistributed here.

## Citation

Paper under review (double-blind); citation entry will be added on
de-anonymization. Until then, cite this repository.
