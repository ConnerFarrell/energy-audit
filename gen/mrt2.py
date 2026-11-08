#!/usr/bin/env python3
"""Generate the frozen prompt matrix with Magenta RealTime 2 (google/magenta-realtime-2).

Tag: mrt2 (model size mrt2_base unless --size says otherwise).

The script uses the library's offline JAX backend (magenta-rt 2.0.3), and the
weights are pulled from Hugging Face at a pinned revision. Every clip is exactly
DURATION_S long: 25 frames/s, 1,920 samples/frame, 48 kHz stereo, PCM_16.

Seeding. magenta-rt's JAX `generate()` exposes no seed: its sampler RNG is
fixed at `jax.random.PRNGKey(0)` when the initial streaming state is built, and
every clip would come out identical. So this harness builds the initial state
itself, with `PRNGKey(seed)` for the sampler (`seeded_initial_state`), and
passes the same seed to the MusicCoCa text-to-style mapper noise
(`embed_style(..., use_mapper=True, seed=seed)`). Per-clip seeds follow the
July harness convention: 300000 + crc32(filename) mod 1e6.

Determinism. XLA_FLAGS gets --xla_gpu_deterministic_ops=true before jax is
imported, so the same seed is expected to give bit-identical audio. The smoke
test checks this.

Outputs: <out>/audio/mrt2__<prompt_id>__s<k>.wav, plus <out>/manifest_mrt2.csv
(one row per clip, appended as clips finish, so a run can stop and resume).

Usage (Kaggle):
  python gen/mrt2.py --prompts data/prompts.csv --out /kaggle/working \
      --magenta-home /tmp/magenta [--only rock_L1_verb] [--seeds 0,1,2] \
      [--deadline-h 11.0] [--repeat-first]
"""
from __future__ import annotations

import os

# Must be set before jax is imported anywhere.
_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_deterministic_ops" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_gpu_deterministic_ops=true").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import argparse
import csv
import hashlib
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd
import soundfile as sf

TAG = "mrt2"
HF_REPO = "google/magenta-realtime-2"
HF_REVISION = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"  # pulled 2026-11-08
LIB_VERSION = "2.0.3"

SAMPLE_RATE = 48_000
CHANNELS = 2
FRAME_RATE = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE  # 1,920
DURATION_S = 30
N_SEEDS = 6

# Library defaults for MagentaRT2System (jax/system.py), stated explicitly.
TEMPERATURE = 1.3
TOP_K = 40
CFG_SCALES = {"musiccoca": 3.0, "notes": 1.0, "drums": 1.0}
USE_MAPPER = True  # as in the library's own `mrt jax generate`

# Non-silence floor for the audit.
MIN_PEAK = 1e-3
MIN_RMS_DBFS = -60.0

MUSICCOCA_FILES = [
    "resources/musiccoca/audio_preprocessor.tflite",
    "resources/musiccoca/mapper.tflite",
    "resources/musiccoca/music_encoder.tflite",
    "resources/musiccoca/pretrained_vector_quantizer.tflite",
    "resources/musiccoca/spm.model",
    "resources/musiccoca/text_encoder.tflite",
]

MANIFEST_FIELDS = [
    "clip", "model", "size", "hf_revision", "prompt_id", "prompt_text",
    "seed_index", "seed", "mapper_seed", "frames", "gen_s",
    "sample_rate", "channels", "n_samples", "duration_s", "peak", "rms_dbfs",
    "pcm_sha256",
]


def clip_name(prompt_id: str, k: int) -> str:
    return f"{TAG}__{prompt_id}__s{k}.wav"


def clip_seed(name: str) -> int:
    """Per-clip seed, the same convention as the July generation harness."""
    return 300000 + zlib.crc32(name.encode()) % 10**6


def download_assets(magenta_home: pathlib.Path, size: str, revision: str) -> pathlib.Path:
    """Fetch the checkpoint and MusicCoCa resources at a pinned HF revision."""
    from huggingface_hub import hf_hub_download

    root = magenta_home / "magenta-rt-v2"
    for f in [f"checkpoints/{size}.safetensors", *MUSICCOCA_FILES]:
        hf_hub_download(repo_id=HF_REPO, filename=f, revision=revision, local_dir=str(root))
    return root


def build_system(magenta_home: pathlib.Path, size: str):
    from magenta_rt import paths
    paths.set_magenta_home(magenta_home / "magenta-rt-v2")
    from magenta_rt import MagentaRT2Jax
    import magenta_rt

    assert magenta_rt.__version__ == LIB_VERSION, magenta_rt.__version__
    return MagentaRT2Jax(size=size, temperature=TEMPERATURE, top_k=TOP_K,
                         cfg_scales=dict(CFG_SCALES))


def make_seeded_init(mrt):
    """Return init(seed) -> initial streaming state whose sampler RNG is PRNGKey(seed).

    Mirrors MagentaRT2System._compile()'s `_init_state`, but with the 'random'
    key passed in rather than fixed at PRNGKey(0). The state has the same
    structure, so the library's AOT-compiled streaming step is reused unchanged.
    """
    import jax
    import jax.numpy as jnp

    sampler = mrt._sampler
    spec = jax.ShapeDtypeStruct([mrt._num_channels], jnp.int32)

    @jax.jit
    def _init(params, key):
        rngs = {"params": jax.random.PRNGKey(42), "random": key}
        return sampler.apply(params, 1, spec, constants={}, training=False,
                             rngs=rngs, method=sampler.get_initial_state)

    def init(seed: int):
        return _init(mrt._params, jax.random.PRNGKey(seed))

    return init


def generate_pcm(mrt, init, text: str, seed: int, frames: int) -> np.ndarray:
    """One clip as int16 [n_samples, 2]: the library's own int16 output, unrequantized."""
    from magenta_rt.config import MUSICCOCA

    emb = mrt.embed_style(text, use_mapper=USE_MAPPER, seed=seed)
    wav, _ = mrt.generate(conditioning={MUSICCOCA.key: emb}, frames=frames, state=init(seed))
    x = np.asarray(wav.samples, dtype=np.float32)
    pcm = np.round(x * 32768.0).astype(np.int16)
    # Waveform holds int16/32768 exactly, so this round trip must be lossless.
    assert np.array_equal(pcm.astype(np.float32) / 32768.0, x), "int16 round trip not exact"
    return pcm


def audit(path: pathlib.Path, duration_s: float) -> dict:
    """Assert format, exact duration and non-silence. Returns the measured values."""
    info = sf.info(str(path))
    assert info.samplerate == SAMPLE_RATE, (path.name, info.samplerate)
    assert info.channels == CHANNELS, (path.name, info.channels)
    assert info.subtype == "PCM_16", (path.name, info.subtype)
    n_expect = int(round(duration_s * SAMPLE_RATE))
    assert info.frames == n_expect, (path.name, info.frames, n_expect)
    y, _ = sf.read(str(path), dtype="float64", always_2d=True)
    assert np.isfinite(y).all(), path.name
    peak = float(np.abs(y).max())
    rms = float(np.sqrt(np.mean(y ** 2)))
    rms_dbfs = 20 * np.log10(max(rms, 1e-12))
    assert peak >= MIN_PEAK and rms_dbfs > MIN_RMS_DBFS, (path.name, peak, rms_dbfs)
    return dict(sample_rate=info.samplerate, channels=info.channels, n_samples=info.frames,
                duration_s=info.frames / info.samplerate, peak=peak, rms_dbfs=rms_dbfs)


def pcm_sha256(path: pathlib.Path) -> str:
    y, _ = sf.read(str(path), dtype="int16", always_2d=True)
    return hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", default="data/prompts.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--magenta-home", required=True)
    ap.add_argument("--size", default="mrt2_base", choices=["mrt2_base", "mrt2_small"])
    ap.add_argument("--revision", default=HF_REVISION)
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--only", default="", help="comma-separated prompt_ids (default: all 50)")
    ap.add_argument("--seeds", default="", help="comma-separated seed indices (default: 0..5)")
    ap.add_argument("--deadline-h", type=float, default=1e9,
                    help="stop cleanly after this many hours (resume skips finished clips)")
    ap.add_argument("--repeat-first", action="store_true",
                    help="regenerate the first clip with the same seed; require identical PCM")
    args = ap.parse_args(argv)

    t_start = time.time()
    out = pathlib.Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    manifest = out / f"manifest_{TAG}.csv"
    home = pathlib.Path(args.magenta_home)

    frames = int(round(args.duration * FRAME_RATE))
    assert frames * SAMPLES_PER_FRAME == int(round(args.duration * SAMPLE_RATE))

    P = pd.read_csv(args.prompts)
    assert len(P) == 50 and P.prompt_id.is_unique, "frozen matrix must be 50 unique prompts"
    assert P.prompt_text.str.startswith("instrumental ").all(), "'instrumental' prefix missing"
    if args.only:
        keep = args.only.split(",")
        assert set(keep) <= set(P.prompt_id), set(keep) - set(P.prompt_id)
        P = P[P.prompt_id.isin(keep)]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(N_SEEDS))

    todo = [(r.prompt_id, r.prompt_text, k) for r in P.itertuples() for k in seeds
            if not (out / "audio" / clip_name(r.prompt_id, k)).exists()]
    print(f"[mrt2] {len(todo)} clips to make ({args.size}, rev {args.revision[:10]}, "
          f"{args.duration:g} s = {frames} frames)", flush=True)
    if not todo and not args.repeat_first:
        return 0

    t0 = time.time()
    download_assets(home, args.size, args.revision)
    t_dl = time.time() - t0
    t0 = time.time()
    mrt = build_system(home, args.size)
    init = make_seeded_init(mrt)
    t_build = time.time() - t0
    import jax
    print(f"[mrt2] devices={jax.devices()} jax={jax.__version__} "
          f"download {t_dl:.0f}s, load+compile {t_build:.0f}s", flush=True)

    new_file = not manifest.exists()
    with open(manifest, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new_file:
            w.writeheader()
        made = 0
        t_loop = time.time()
        for pid, text, k in todo:
            if (time.time() - t_start) / 3600 > args.deadline_h:
                print("[mrt2] DEADLINE reached, stopping cleanly for resume", flush=True)
                break
            name = clip_name(pid, k)
            seed = clip_seed(name)
            tc = time.time()
            pcm = generate_pcm(mrt, init, text, seed, frames)
            gen_s = time.time() - tc
            path = out / "audio" / name
            tmp = path.with_suffix(".tmp.wav")
            sf.write(str(tmp), pcm, SAMPLE_RATE, subtype="PCM_16")
            meas = audit(tmp, args.duration)
            tmp.rename(path)
            w.writerow(dict(clip=name, model=HF_REPO, size=args.size, hf_revision=args.revision,
                            prompt_id=pid, prompt_text=text, seed_index=k, seed=seed,
                            mapper_seed=seed, frames=frames, gen_s=round(gen_s, 3),
                            pcm_sha256=pcm_sha256(path), **meas))
            fh.flush()
            made += 1
            rate = (time.time() - t_loop) / made
            print(f"[mrt2] {made}/{len(todo)} {name} {gen_s:.1f}s | mean {rate:.1f}s/clip | "
                  f"ETA {(len(todo) - made) * rate / 60:.0f} min", flush=True)

    if args.repeat_first:
        pid, text, k = (todo[0] if todo else (P.iloc[0].prompt_id, P.iloc[0].prompt_text, seeds[0]))
        name = clip_name(pid, k)
        pcm2 = generate_pcm(mrt, init, text, clip_seed(name), frames)
        a, _ = sf.read(str(out / "audio" / name), dtype="int16", always_2d=True)
        same = np.array_equal(a, pcm2)
        print(f"[mrt2] REPEAT {name}: {'IDENTICAL' if same else 'DIFFERENT'} "
              f"(max |diff| {int(np.abs(a.astype(np.int32) - pcm2.astype(np.int32)).max())})",
              flush=True)
        if not same:
            return 2
    print(f"[mrt2] done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
