# P0b re-run · Step 3 · Scoring plan for mrt2 (fixed before scoring, 2026-11-08)

## 1. How the July pipeline handled very quiet clips (checked 2026-11-08)

Sources read: the four July generation notebooks (`p0b-generate-a`/`-b`/`-c`/`-d`, private Kaggle), the
canonical scoring notebook (`p0b-score`), `notebooks/scoring.ipynb` (repo), `data/scored_full_v2.csv`.

**Generation.** None of the four July harnesses had a silence or loudness gate. MusicGen (A) and Stable Audio
Open (B) clip samples to [-1, 1] before writing; Stable Audio 3 (C) clamps to [-1, 1]; ACE-Step (D) moves
the pipeline's own WAV into place unchanged (the pipeline peak-normalizes to −1 dB itself). Every generated clip was kept. The revised `gen/mrt2.py` matches this: no clip is
rejected for being quiet, and the int16 output can't exceed full scale.

**Scorer (`score_all`, used unchanged).** Integrated loudness comes from `pyloudnorm.Meter(sr)` on the
librosa mono load at the native rate. The normalized lens applies gain `10**((-14 - lufs)/20)` and clips to
[-1, 1], with no guard on `lufs`. If anything in `score_all` raises, the clip gets NaN in **all** columns
(`p0a`, `lufs`, `clap`, `p0a_n`, `clap_n`). Downstream statistics drop NaNs (`a[~np.isnan(a)]`).

**The banked July corpus never reached this regime.** Over 1,500 clips: quietest −39.0 LUFS (mgm), largest
gain needed to reach −14 LUFS 25.0 dB (mgm), zero clips needing more than +40 dB, zero NaNs in `p0a`, `lufs`
or `p0a_n`. Per model, max gain: ace 10.2, sa3s 13.9, sao 13.5, mgs 20.2, mgm 25.0 dB.

**Consequence:** the rule in §2 cannot change any of the paper's five rows. It only applies where mrt2
produces clips quieter than anything in July.

## 2. Rule for mrt2 (Conner, 2026-11-08)

1. The scorer runs **unchanged** (same notebook code, pinned image) on the 300 mrt2 clips.
2. **−inf loudness exclusion, normalized lens only.** Any clip whose scorer-measured `lufs` is −inf (below
   pyloudnorm's −70 LUFS absolute gate) gets `p0a_n` and `clap_n` set to NaN **after** scoring, so it is
   excluded from the −14 LUFS normalized lens only. Its raw-lens values (`p0a`, `lufs`, `clap`) are kept. The
   excluded count is reported per cell (model × phrasing × level, and per prompt).
3. **+40 dB caveat.** Any clip needing more than +40 dB gain to reach −14 LUFS (`lufs < −54`) is flagged and
   listed in the memo as a caveat. It is not excluded.
4. If `score_all` raises on a clip, the July behavior applies (NaN in every column), and it is reported.
5. The harness's `quiet_flag` (RMS < −60 dBFS) is reported alongside the scores but plays no part in any
   computation.

## 3. Pass rule (unchanged)

On a fresh pinned-image run, the scoring notebook must also reproduce the banked values for the existing
1,500 clips: Arousal Index (`p0a`, `p0a_n`) and LUFS exact; banked cells within 1e-12 relative. That makes the
mrt2 row comparable to the paper's five.
