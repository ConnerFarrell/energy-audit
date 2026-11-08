# P0b re-run · Step 3 · Run manifest: mrt2 full generation

*Written at push time, 2026-11-08. Fields that only exist after the run are tagged [STATUS?]. They get filled
from the run's own `run_manifest.json` (Kaggle Output) once Conner reports the run finished. Nothing here is
inferred.*

| field | value |
|---|---|
| model | `google/magenta-realtime-2`, size `mrt2_base` (2.4B-parameter decoder-only transformer, SpectroStream codec, MusicCoCa text conditioning) |
| HF revision | `010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc` (pulled 2026-11-08) |
| library | `magenta-rt` 2.0.3, offline JAX backend; JAX 0.11.0 (CUDA 12 build), flax 0.12.8 |
| environment | `rerun/step3/requirements-gen-mrt2.txt` (exact versions resolved in smoke v1; identical in smoke v2) |
| harness | `gen/mrt2.py` @ `34f50fc` (sha256 `dc584778…f71138b1`), fetched and hash-checked inside the run |
| prompts | `data/prompts.csv` (50 prompts, "instrumental" prefix), sha256 `9bf21975…12f0e2a2e`; 6 seeds each = 300 clips |
| seeds | per clip 300000 + crc32(filename) mod 10^6, used for both the sampler key and the MusicCoCa mapper noise |
| sampling | temperature 1.3, top-k 40, CFG musiccoca 3.0 / notes 1.0 / drums 1.0, mapper on (library defaults) |
| clip format | 48 kHz, 2 channels, PCM_16, exactly 30.000 s (750 frames); audited on every clip by the harness |
| Kaggle notebook | private `energy-audit-mrt2-generate`, version 1 (source: `kaggle_generate/energy-audit-mrt2-generate.ipynb`) |
| GPU | Tesla T4 (machine shape `NvidiaTeslaT4`; Kaggle attaches 2× T4, JAX uses `cuda:0`); driver 580.178.04 in both smoke runs |
| image | pinned to digest `sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (the canonical run's image) |
| pushed | 2026-11-08 05:08 UTC; status RUNNING at 05:09 UTC |
| expected wall-clock | ~6.3 h (smoke: 73.8 s/clip × 300 + ~3 min setup), not a measurement |
| time stop | harness stops cleanly at 11.0 h; the finished clips are packed into the Output |
| finished | [STATUS?] |
| measured wall-clock | [STATUS?] (`run_manifest.json` → `notebook_wall_h`, `gen_s_total_h`) |
| completeness | [STATUS?] (`run_manifest.json` → `status`, `checks`) |
| GPU quota used | [STATUS?] (not exposed by the Kaggle API; read it from the Kaggle quota page) |
| corpus bank | [STATUS?] private Kaggle dataset version, created from this run's `audio_mrt2.tar` + `manifest_mrt2.csv` after it finishes |

## If the run stops at the 11 h mark

`run_manifest.json` will say `PARTIAL`. To resume, push version 2 of the same notebook with its own output
attached (`kernel_sources: ["<owner>/energy-audit-mrt2-generate"]`). Cell 3 restores every clip that has a
manifest row, and the harness generates only the rest. Same seeds give bit-identical audio across sessions
(smoke v1 vs v2: 6/6 identical), so a resumed corpus is the same as an unbroken one.

## Smoke evidence (Part B)

| | v1 | v2 |
|---|---|---|
| JAX CUDA 12 on GPU | yes | yes |
| audit (6 clips) | pass | pass |
| 6 seeds → distinct clips | 6/6 | 6/6 |
| same seed twice | identical | identical |
| identical to v1 | — | 6/6 |
| s/clip (excluding first) | 73.5 | 73.8 |

Records: `smoke_v1/`, `smoke_v2/`. Listening clips (s0, s1, s3 of `rock_L1_verb`) are in the private smoke
notebook's Output. Conner approved them on 2026-11-08 ("clips OK").
