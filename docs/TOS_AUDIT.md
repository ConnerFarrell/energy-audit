# TOS_AUDIT — model licensing audit for the energy-audit study

*Snapshot of the licensing review done on 2026-07-22, before any clip was generated. Licenses change; re-check the live pages linked below before reusing any of this. This is a researcher's reading of public license text, not legal advice.*

## What this checks

For each candidate model, whether its license permits:

- (a) generating audio for research,
- (b) publishing evaluation scores and figures,
- (c) publishing short audio exhibits,

and what obligations come with it.

**Legend:** OK = clearly permitted · OK* = permitted with obligations · EXCLUDED = not used, reason given.

## Models used in the study

### 1. facebook/musicgen-small · facebook/musicgen-medium
- **Page:** https://huggingface.co/facebook/musicgen-small (medium: https://huggingface.co/facebook/musicgen-medium)
- **License:** code MIT; **weights CC-BY-NC-4.0** (both checkpoints, as stated on the model card).
- **(a) Generate for research:** OK. The card's stated primary use is research into the limitations of generative models.
- **(b) Publish scores/figures:** OK. These are measurements of outputs, published for research.
- **(c) Publish audio exhibits:** OK*. The outputs come from a non-commercial-licensed model, so MusicGen audio exhibits appear only on the free public demo page and in the paper's supplement, never in commercial material.
- **Obligations:** cite Copet et al. 2023 (arXiv:2306.05284); keep any redistribution of the audio non-commercial.
- **Notes:** 32 kHz mono output; trained on instrumental music with vocals removed, which is one reason every prompt uses the "instrumental" prefix.

### 2. stabilityai/stable-audio-open-1.0
- **Page:** https://huggingface.co/stabilityai/stable-audio-open-1.0
- **License:** Stability AI Community License (gated repo; accept the terms and use a Hugging Face token). Full text reviewed: Stability AI Community License Agreement, last updated 2024-07-05.
- **(a) Generate for research:** OK. §II grants research and non-commercial use.
- **Output ownership:** §IV(c)(iii) states the user owns outputs, to the extent permitted by law.
- **(b) + (c) Publish scores/figures and audio exhibits:** OK*.
- **Obligations (§IV(a)):** keep the attribution notice in the repo — *"This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved"* — and display **"Powered by Stability AI"** prominently on the related website, demo page, and README.
- **Notes:** 44.1 kHz stereo, up to 47 s. The model card notes it is stronger at sound effects and field recordings than at music, a caveat the paper reports.

### 3. stabilityai/stable-audio-3-small-music
- **Page:** https://huggingface.co/stabilityai/stable-audio-3-small-music
- **License:** Stability AI Community License (same terms and obligations as above) **plus the Gemma Terms of Use** for the bundled T5Gemma text encoder, including the Gemma §3.2 use restrictions. Gated repo.
- **Verdict:** OK*. Same attribution as Stable Audio Open 1.0; the Gemma terms acceptance is noted in the repo's license notes.
- **Notes:** the 2026 music-tuned successor to Stable Audio Open; together the two form the study's same-vendor 2024 → 2026 comparison. Requires the stable-audio-3 / stable-audio-tools libraries.

### 4. ACE-Step/Ace-Step1.5 (turbo DiT variant)
- **Page:** https://huggingface.co/ACE-Step/Ace-Step1.5
- **License:** MIT. The model card explicitly permits commercial use of generated music.
- **Verdict:** OK. The least restrictive license in the roster.
- **Obligations:** cite the technical report (arXiv:2602.00744).
- **Notes:** 2026 hybrid language-model planner + diffusion transformer, tuned with reinforcement learning for prompt adherence, which is the property this study measures. Custom pipeline (https://github.com/ace-step/ACE-Step-1.5); supports duration control, used to request ~30 s clips.

### 5. google/magenta-realtime-2 (`mrt2_base`)
- **Status:** added 2026-11-08 for the re-run's Step 3 (one new model). Reviewed before any clip was generated, per the standing policy below.
- **Page:** https://huggingface.co/google/magenta-realtime-2 (code: https://github.com/magenta/magenta-realtime; library: PyPI `magenta-rt` 2.0.3, 2026-07-30)
- **Re-checked:** 2026-11-08, live model card at HF revision `010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc` (last modified 2026-06-04). Not gated.
- **License:** code Apache 2.0; **weights CC-BY-4.0**. The card's Terms of Use add: *"Google claims no rights in outputs you generate using Magenta RealTime 2. You and your users are solely responsible for outputs and their subsequent uses,"* and ask users not to generate content that infringes the rights of others, including copyrighted content.
- **(a) Generate for research:** OK. Research is a listed use. The card states the model is trained mostly on instrumental stock music (~71k hours).
- **(b) Publish scores/figures:** OK. These are measurements of outputs, published for research.
- **(c) Publish audio exhibits:** OK*. Google claims no rights in outputs, and the weights license is not non-commercial. Exhibits follow the general terms only: no infringing content, and the user is responsible for outputs.
- **Obligations:** if the weights are redistributed (we don't), CC-BY-4.0 attribution: credit Google DeepMind, link the license, note any changes. For the paper: cite the Magenta RealTime / live music models paper (arXiv:2508.04651) and the SpectroStream codec (arXiv:2508.05207). Keep the Apache 2.0 notice if we vendor or patch library code.
- **Notes:** 2.4B-parameter decoder-only transformer generating SpectroStream tokens frame by frame (25 frames/s, 48 kHz stereo), conditioned on a MusicCoCa text-style embedding. It's a streaming "live music" architecture, a class not in the roster. Duration is set exactly by frame count (30 s = 750 frames). Generation here uses the library's offline JAX (CUDA) backend, not the real-time MLX path. The 2026-07-22 audit set it aside because "its real-time streaming stack doesn't suit batch evaluation"; that reason no longer holds, since `magenta-rt` now ships an offline JAX backend for "offline / batch generation and research." The library's JAX `generate()` exposes no sampling seed (the sampler key is fixed at `PRNGKey(0)`), so the harness (`gen/mrt2.py`) sets the per-clip seed explicitly on the initial sampler state and on the MusicCoCa mapper. This is a harness detail, not a license matter.

## Considered, not used (candidates for future work)

- **google/magenta-realtime-2** — license clean (Apache 2.0 code, CC-BY-4.0 weights; Google claims no rights in outputs). Not used: its real-time streaming stack doesn't suit batch evaluation. A strong candidate for future work. *(2026-11-08: re-reviewed and added as entry 5 above; the library now has an offline JAX backend.)*
- **HeartMuLa/HeartMuLa-oss-3B** — license clean (Apache 2.0). Not used: young library stack, untested in this study's compute setup. Revisit for an extended version.
- **riffusion/riffusion-model-v1** — license fine (CreativeML OpenRAIL-M). Not used: a 2022 spectrogram-image approach, no longer representative.
- **ACE-Step/ACE-Step-v1-3.5B** — Apache 2.0. Superseded by Ace-Step 1.5 from the same team.

## Excluded: commercial services

**Suno-class and Udio-class services — EXCLUDED.** Their free tiers appear to restrict outputs to non-commercial use with the platform retaining output rights. That reading comes from secondary summaries reviewed 2026-07-22, not independently verified against the live terms of service. Both platforms were also in active litigation or settlement with major labels. Publishing third-party evaluations of their outputs was not clearly permitted under any tier reviewed, so the study uses open models only. Commercial systems are named in the paper as the natural extension.

## Standing policy

- Every headline figure is reproducible from open models only (satisfied: the roster is open-only).
- Audio exhibits from non-commercial-licensed models (MusicGen) are hosted only on the free public demo page and in the paper's supplement. MIT and Stability outputs follow their own terms above.
- Any model added later gets its own entry here **before** a single clip is generated.
