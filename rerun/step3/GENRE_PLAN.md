# P0b re-run · Step 3 · Genre-adherence check (fixed before looking, 2026-11-08)

Motivation: Conner's listening notes on 7 mrt2 clips (MEMO.md). The mrt2 L9 clips seemed to collapse toward
loud EDM, whatever genre was requested. This file fixes the question, the metric and the decision rule
**before** any genre number is computed. It is committed before the measurement runs.

## Question

Does mrt2's genre adherence drop from L1 to L9 (verbal phrasing) more than the other five models'?

## Metric

All 1,800 already-generated clips (five banked models + mrt2). No new generation.

- **Classifier:** CLAP zero-shot genre classification with the same checkpoint and audio path as the scorer's
  CLAP. That is `laion/clap-htsat-unfused` and `ClapProcessor`. The audio is the librosa mono load at the native
  rate, resampled to 48 kHz, with no loudness normalization. It is seeded with `np.random.seed(0)` immediately
  before every processor call, as in `score_all`.
- **Labels:** the five genres of `data/prompts.csv`, as "<genre> music": "rock music", "electronic dance
  music", "hip-hop music", "acoustic folk music", "orchestral music".
- **Per clip:** softmax over the five audio–text logits (`logits_per_audio`), giving one probability per genre.
  Top genre = argmax.
- **Per model × phrasing × level:**
  (a) top-1 adherence = % of clips whose top genre is the prompted genre;
  (b) mean prompted-genre probability.
  95% CIs come from a 2,000-resample clip-level bootstrap, percentile method, `default_rng(0)`, as in the paper.
- **Drop** = value at L1 minus value at L9, verbal (positive means adherence falls as intensity rises). The
  bootstrap resamples L1 and L9 clips independently.
- **Descriptive only, no decision attached:** the share of clips classified as "electronic dance music",
  per model × level.

## Decision rule (primary metric: top-1 adherence)

mrt2's adherence "drops more than the other five" if, **for each** of the five models, the 95% CI of
(drop_mrt2 − drop_model) lies entirely above 0. The difference is bootstrapped from independent resamples of
each model's L1 and L9 clips. If that holds for some models but not all, the answer is reported model by model.
The same comparison on mean prompted-genre probability is reported as secondary.

## Limits, stated in advance

- CLAP is a measurement instrument, not ground truth for genre. Its own label bias (for example toward
  "electronic dance music" for loud, dense audio) would affect every model the same way, which is why the
  question is a between-model comparison of drops.
- No LLM judges audio. CLAP is a contrastive audio–text embedding model run as a fixed measurement.

## Also requested

Every clip in the 1,800-clip corpus below −50 dBFS RMS (whole file, all channels, the harness's audit
formula), listed with its Arousal Index (`p0a`, `p0a_n`) and LUFS.
