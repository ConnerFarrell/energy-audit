# Model roster

Five open text-to-music checkpoints crossing scale (within family), era
(within vendor), and architecture. All generation used the frozen prompt
matrix below; all scoring used one identical front end (mono sum,
22.05 kHz, first 30 s).

| tag | HF checkpoint | year | params | architecture | harness |
|---|---|---|---|---|---|
| mgs  | facebook/musicgen-small | 2023 | ~300M | AR LM over EnCodec tokens | transformers `pipeline("text-to-audio")` |
| mgm  | facebook/musicgen-medium | 2023 | 1.5B | AR LM over EnCodec tokens | same as mgs |
| sao  | stabilityai/stable-audio-open-1.0 | 2024 | ~1B | latent diffusion (DiT) | diffusers `StableAudioPipeline` |
| sa3s | stabilityai/stable-audio-3-small-music | 2026 | 0.6B | latent diffusion, music-tuned | stable-audio-tools |
| ace  | ACE-Step/Ace-Step1.5 (v15-turbo DiT) | 2026 | ~2B class | hybrid LM-planner + DiT, RL adherence-tuned | ACE-Step-1.5 pipeline (vllm; thinking=True, shift=3.0) |

## Pinned revisions

| tag | HF revision hash | pulled |
|---|---|---|
| mgs  | `4c8334b02c6ec4e8664a91979669a501ec497792` | 2026-07-22 |
| mgm  | `d3bd7b00761b78ad7a8a05145ee31e7832e9916c` | 2026-07-22 |
| sao  | `f21265c1e2710b3bd2386596943f0007f55f802e` | 2026-07-22 |
| sa3s | `0fef1392cd842149a2b6d445e181c97608faac06` | 2026-07-22 |
| ace  | `19671f406d603126926c1b7e2adc169acbcade22` | 2026-07-22 |

## Frozen prompt matrix

- numeric: `instrumental {genre} track, energy level {L} out of 11`
- verbal: `instrumental {genre} track, {descriptor}` with the fixed ladder
  *very calm / calm / moderate energy / energetic / extremely intense*
- genres: rock, electronic dance (EDM), hip-hop, acoustic folk, orchestral
- levels L ∈ {1, 3, 5, 7, 9}; six seeds per cell; ~30 s clips
- "instrumental" is prepended throughout to neutralize the vocal-capability
  confound across models
- full list: `data/prompts.csv`

## Notes

- ACE-Step 1.5 peak-normalizes its output to −1 dB by design; its loudness
  channel is therefore partially self-clamped (reported as model behavior).
- Stable Audio Open's model card notes it is stronger on sound effects and
  field recordings than music — one motivation for the 2024→2026
  within-vendor comparison.
