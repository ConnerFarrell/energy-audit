# Licenses and obligations

Analysis code in this repository: MIT. The audited model checkpoints carry
their own licenses, which govern the generated audio referenced by this
work. License pages were reviewed 2026-07-22 and re-checked before release;
verify the live pages if you redistribute anything derived from these
models.

| model | weights license | audio exhibits policy here |
|---|---|---|
| facebook/musicgen-small / -medium | CC-BY-NC-4.0 (code MIT) | non-commercial research contexts only; linked, never embedded in commercial artifacts |
| stabilityai/stable-audio-open-1.0 | Stability AI Community License | permitted with attribution (see `NOTICE`); "Powered by Stability AI" |
| stabilityai/stable-audio-3-small-music | Stability AI Community License + Gemma Terms of Use (redistributed T5Gemma encoder) | same as above; Gemma ToU noted |
| ACE-Step/Ace-Step1.5 | MIT | unrestricted |

Evaluation corpora: DEAM (Aljanaki et al., 2017) and PMEmo (Zhang et al.,
2018) were used to train and control the arousal instrument; no corpus
audio is redistributed here.

Required citations: Copet et al. 2023 (arXiv:2306.05284) for MusicGen;
Evans et al. 2025 (ICASSP; arXiv:2407.14358) for Stable Audio Open;
arXiv:2602.00744 for ACE-Step 1.5.
