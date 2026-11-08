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
| finished | 2026-11-08 06:34 UTC (notebook COMPLETE on Kaggle; corpus **PARTIAL**, see below) |
| measured wall-clock | 1.42 h notebook (1.34 h generating; 73.1 s/clip mean over 66 clips) |
| completeness | **66/300.** The harness stopped at clip 67 (`mrt2__edm_L1_verb__s0`) on its non-silence audit: RMS −67.8 dBFS, peak 0.031, below the −60 dBFS floor. Records: `generate_v1/` |
| GPU quota used | about 1.4 h by wall-clock; the exact figure is [STATUS?] (Kaggle quota page) |
| corpus bank | version 1: not banked, the corpus was incomplete (66/300). *Superseded: the complete corpus is banked after version 2, see below.* |

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

## Version 1 outcome (2026-11-08)

The run stopped at 66/300 clips after 1.42 h. This was not a crash: the harness's own non-silence check
(`audit()`, floor −60 dBFS RMS, chosen by me in Part B) rejected clip 67,
`mrt2__edm_L1_verb__s0` ("instrumental electronic dance track, very calm"):

* seconds 0–9: about −92 dBFS RMS (peak ≤ 0.0001), near-digital silence; 15.5% of samples are exact zeros
* seconds 9–30: −61 to −73 dBFS RMS per second, peaks 0.003–0.030; very quiet, not empty
* whole clip: −67.8 dBFS RMS, peak 0.031

The other 66 clips (all 60 rock clips plus the 6 `edm_L1_num` clips) range from −29.1 to −13.5 dBFS RMS and all pass.

Harness defect, separate from the gate: the rejected clip was still on disk as `….tmp.wav`, which matches
`*.wav`. So `audio_mrt2.tar` holds 67 WAVs against 66 manifest rows. The 66 manifest rows are the valid
corpus; the `.tmp.wav` is the rejected clip.

Resume is blocked on the gate decision. Not done: banking, scoring.

## Version 2: resume (2026-11-08)

* **Decision (Conner, 2026-11-08):** A. Quiet clips are model output. The harness halts only on format errors,
  non-finite samples or digital silence, and flags quiet clips (`quiet_flag`, RMS < −60 dBFS). The `.tmp.wav`
  bug is fixed: clips are written to `partial/` and moved into `audio/` only after the audit passes.
* **Harness:** `gen/mrt2.py` @ `91d5677` (sha256 `3bfaec69…cc0755df`). Only the audit, the manifest and the
  temp path changed; the generation path (sampling, seeding, conditioning) is identical to `34f50fc`.
* **Resume:** version 2 pushed 2026-11-08 06:44 UTC (RUNNING at 06:45), with version 1's output attached as
  input. It restores the 66 audited clips (tested locally against v1's real output: 66 restored, the
  `.tmp.wav` dropped, all 66 PCM hashes match) and generates the other 234. Clip 67 is regenerated from the
  same seed. Expected ~4.8 h of generation (234 × 73 s), not a measurement.
* **Finished:** 2026-11-08 11:36 UTC, status **COMPLETE**, harness exit code 0. Records: `generate_v2/`.
* **Wall-clock:** 4.86 h notebook for version 2 (234 clips, 73.6 s/clip). Whole corpus: 6.12 h of generation
  (73.5 s/clip mean over 300); 1.42 + 4.86 = 6.28 h of notebook time across both versions.
* **Completeness:** 300/300 clips on disk, in the manifest and in the tar; all 48 kHz stereo, exactly 30 s; 300
  distinct PCM hashes and 300 distinct seeds; `partial/` empty.
* **Quiet clips:** 1, `mrt2__edm_L1_verb__s0` (−67.8 dBFS RMS), kept and flagged.
* **Determinism, checked after download:** the regenerated clip 67 is bit-identical to the clip version 1
  rejected, the 66 restored clips' hashes are unchanged, and all 300 clips in the tar match the manifest.
* **Corpus bank:** private Kaggle dataset `energy-audit-mrt2-corpus`, version 1, created 2026-11-08 19:18 UTC
  (status `ready`). Contents: `audio_mrt2.tar` (sha256 `4f9cd40b…51a84d63`, 1,728,624,640 bytes; Kaggle shows
  it unpacked as `audio_mrt2/audio/*.wav`), `manifest_mrt2.csv` (sha256 `124d71cd…67f40f4`), `run_manifest.json`,
  `prompts.csv`.
* **GPU quota, both versions:** about 6.3 h by notebook wall-clock; the exact figure is [STATUS?] (Kaggle quota page).
* **Scoring rule** (−inf loudness excluded from the normalized lens only; +40 dB gain flagged as a caveat) and
  the check of July's quiet-clip handling: `SCORING_PLAN.md`.
