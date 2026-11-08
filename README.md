# Do Text-to-Music Models Take Direction?
**An instrumented audit of energy-instruction compliance in five open text-to-music models (2023–2026).**

**Conner O. Farrell** · CCRMA, Stanford University · Preprint, 2026 (arXiv + DOI forthcoming) · the submitted PDF is in [`paper/`](paper/farrell-ttm-direction-2026.pdf) · the arousal instrument lives in its own repo: [intensity-index](https://github.com/ConnerFarrell/intensity-index)

**Powered by Stability AI.** Portions of the audited corpus were generated with Stability AI models under the Stability AI Community License; the attribution notice is in [`NOTICE`](NOTICE).

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

**Tier 1 — statistics (minutes, no GPU, no audio, no DEAM).**
`notebooks/repro.ipynb` regenerates every banked table and CI (`data/s6_*.csv`,
`data/s7_*.csv`) and every figure from `data/`:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir repro_out notebooks/repro.ipynb
```

It can be run from anywhere inside the repo. Inputs come from `REPRO_DATA_DIR`
(default `data/`, never written). Outputs go to `REPRO_OUT_DIR` (default
`repro_out/`): regenerated banks in `tables/` under the same file names as
`data/`, figures in `figures/`. The committed `figures/` folder is never
overwritten by default. The pre-calibration ("raw ridge") statistics use the
fitted Arousal Index ridge in `data/arousal_index_ridge.json`, re-fit from DEAM
by `scripts/fit_arousal_ridge.py`.

**Tier 2 — scoring (Kaggle, ~30 min on a T4).** `notebooks/scoring.ipynb` re-runs
the full three-instrument scoring stack over the generated corpus and
re-banks all CSVs (corpus access and environment details: `docs/MODELS.md`;
versions: `requirements-scoring.txt`).

*Reproducibility guarantee.* A fresh run on the pinned Kaggle image
(the canonical run's Docker image digest) must match the bank exactly. This
was achieved on 2026-11-07: the Arousal Index, LUFS and every non-CLAP banked
statistic were bit-identical to `data/`. Repeats *within one session* may
differ by at most 1e-12 relative, from floating-point last-digit noise. The
measured maximum was about 1e-15 for LUFS (1 ULP) and about 1e-16 absolute
for CLAP. This is accepted, not a bug.

*CLAP.* Scoring was unseeded until 2026-11-07. The CLAP feature extractor
crops 30-second clips to 10 seconds at random, so CLAP (`clap`, `clap_n`) varied
between runs, with measured cell-mean drift (model × phrasing × level) of up
to **0.042** between the banked run and a re-run. Scoring is now seeded. The
seeded values are in `data/clap_seeded_2026-11-07.csv`; the banked `clap` and
`clap_n` columns are unchanged pending a decision on which run the paper
cites. No CLAP verdict changed. The fixed-anchor CLAP axis in `data/s7_*.csv`
was deterministic throughout.

**Tier 3 — generation.** Prompt set (`data/prompts.csv`), model revisions,
and generation harness notes in `docs/MODELS.md`.

## What's here

```
├── paper/
│   ├── farrell-ttm-direction-2026.pdf   # the paper (preprint)
│   ├── main.tex                         # paper source
│   └── refs.bib                         # verified bibliography
├── data/
│   ├── scored_full_v2.csv               # per-clip scores, raw + normalized lenses
│   ├── features_verbal.csv              # per-clip acoustic descriptors
│   ├── s6_*.csv, s7_*.csv               # banked analyses (CIs, decompositions, axes)
│   ├── arousal_index_ridge.json         # fitted Arousal Index ridge (Tier 1 needs no DEAM)
│   ├── clap_seeded_2026-11-07.csv       # seeded CLAP (not yet the banked CLAP)
│   └── prompts.csv                      # the frozen 50-prompt matrix
├── notebooks/
│   ├── scoring.ipynb                    # three-instrument scoring stack (Kaggle)
│   ├── clap-axis.ipynb                  # fixed-anchor CLAP axis (Kaggle)
│   └── repro.ipynb                      # regenerates every statistic in the paper
├── scripts/fit_arousal_ridge.py         # re-fits the ridge from DEAM
├── requirements.txt                     # Tier-1 environment (pinned)
├── requirements-scoring.txt             # Tier-2 environment (partly recorded)
├── figures/                             # compliance curves, per-genre curves, skew
├── docs/
│   ├── MODELS.md                        # model roster + exact checkpoint revisions
│   └── TOS_AUDIT.md                     # license audit for the model roster
├── LICENSE                              # MIT (analysis code)
└── NOTICE                               # Stability AI attribution
```

## Audio exhibits

A curated failure gallery — *listen* to a model refuse "calm rock," hear the
level-9 collapse — is in preparation and will be hosted on the project page
(link will be added here). Exhibits will respect each model's license
(MusicGen audio is CC-BY-NC and appears only in non-commercial research
contexts). Until then, every quantitative claim is independently checkable
from the released CSVs and notebooks above.

## Licensing and attribution

- Analysis code in this repository: MIT.
- Stability AI models (Stable Audio Open 1.0, Stable Audio 3 Small Music):
  Stability AI Community License; attribution notice in `NOTICE` ("Powered by
  Stability AI", see top of this README). Stable Audio 3 Small Music
  additionally redistributes a T5Gemma text encoder under the Gemma Terms of
  Use.
- MusicGen checkpoints: code MIT, weights CC-BY-NC-4.0 (Meta); cite
  Copet et al., 2023 (arXiv:2306.05284). MusicGen-derived audio is used
  non-commercially only.
- ACE-Step 1.5: MIT; cite arXiv:2602.00744.
- Evaluation corpora: DEAM (Aljanaki et al., 2017), PMEmo (Zhang et al.,
  2018) — used for instrument training/controls only; no audio
  redistributed here.

## Citation

```bibtex
@misc{farrell2026direction,
  author = {Farrell, Conner O.},
  title  = {Do Text-to-Music Models Take Direction? An Instrumented Audit
            of Energy-Instruction Compliance},
  year   = {2026},
  note   = {Preprint. Code and data:
            https://github.com/ConnerFarrell/energy-audit},
}
```
The arXiv ID and Zenodo DOI will be appended here once live. The companion
instrument paper: Farrell, *The Arousal Index* (submitted to ICASSP 2027) —
[intensity-index](https://github.com/ConnerFarrell/intensity-index).

## Changelog

- **2026-11-08** — Added `data/clap_seeded_2026-11-07.csv` (CLAP `clap`/`clap_n` from the first seeded Tier-2 run, pinned image). The banked `clap`/`clap_n` columns are unchanged. Replaced the README's "CLAP drift ≤ 0.033" claim with the measured value (cell-mean drift up to 0.042, unseeded) and stated the Tier-2 reproducibility guarantee.
- **2026-11-08** — `data/s6_raw_ridge_level_means.csv`: correction to the 2026-11-07 entry. The file was briefly re-banked from ridge coefficients fit outside the canonical environment (values changed at ~1e-15); that re-bank is reverted, and the file is back to its original values, which the pinned Kaggle image reproduces exactly (25/25). `data/arousal_index_ridge.json` is now fit on that pinned image (`scripts/fit_arousal_ridge.py`). No published number changed at any point.
