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

## Considered, not used (candidates for future work)

- **google/magenta-realtime-2** — license clean (Apache 2.0 code, CC-BY-4.0 weights; Google claims no rights in outputs). Not used: its real-time streaming stack doesn't suit batch evaluation. A strong candidate for future work.
- **HeartMuLa/HeartMuLa-oss-3B** — license clean (Apache 2.0). Not used: young library stack, untested in this study's compute setup. Revisit for an extended version.
- **riffusion/riffusion-model-v1** — license fine (CreativeML OpenRAIL-M). Not used: a 2022 spectrogram-image approach, no longer representative.
- **ACE-Step/ACE-Step-v1-3.5B** — Apache 2.0. Superseded by Ace-Step 1.5 from the same team.

## Excluded: commercial services

**Suno-class and Udio-class services — EXCLUDED.** Their free tiers appear to restrict outputs to non-commercial use with the platform retaining output rights. That reading comes from secondary summaries reviewed 2026-07-22, not independently verified against the live terms of service. Both platforms were also in active litigation or settlement with major labels. Publishing third-party evaluations of their outputs was not clearly permitted under any tier reviewed, so the study uses open models only. Commercial systems are named in the paper as the natural extension.

## Standing policy

- Every headline figure is reproducible from open models only (satisfied: the roster is open-only).
- Audio exhibits from non-commercial-licensed models (MusicGen) are hosted only on the free public demo page and in the paper's supplement. MIT and Stability outputs follow their own terms above.
- Any model added later gets its own entry here **before** a single clip is generated.
