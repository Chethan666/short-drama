"""Pipeline 4 on open models: one drama episode made on a rented Vast.ai GPU.

Here (the GitHub runner, free): each character's fixed voice as a short Kokoro reference clip,
Kokoro fallback lines, the job, and afterwards the edit (TV cuts on every line, film look, big
subtitles, sound bed). On the GPU: Chatterbox voices every line cloned from its character's
voice; Qwen-Image-Edit-2511 draws a portrait per character, a plate per location and every shot
from those references; Wan2.2-S2V-14B films on-screen speakers lip-synced to their line; Wan 2.2
I2V A14B films every other shot (and any talking shot S2V couldn't make); MMAudio makes the
room's ambience. The machine is rented only from hosts that charge nothing for internet traffic.
"""

from __future__ import annotations

import filecmp
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from . import sound
from .produce import (REFERENCE_TEXT, FRAMING, STYLE, Producer, _ffmpeg, character_prompt, location_prompt,
                      motion_prompt, shot_prompt, slug, subtitles)

log = logging.getLogger(__name__)

PICTURE_MODEL = "Qwen/Qwen-Image-Edit-2511"
CLIP_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TALK_MODEL = "Wan-AI/Wan2.2-S2V-14B"
TALK_FRAMINGS = {"close_up", "extreme_close_up", "medium", "over_shoulder"}
HOLDS = {"establishing": 3.0, "insert": 2.2, "wide": 3.0}
RATE = 24_000

REMOTE_SCRIPT = r"""#!/usr/bin/env bash
# Runs detached on the rented machine; every step logs to run.log; DONE marks the end.
set -uo pipefail
cd /workspace/job
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
step() { echo "$(date +%H:%M:%S) == $*"; }
fetch() { python -c "from huggingface_hub import snapshot_download as d; d('$1'$2)" 2>&1 | tail -n 1; }
python -c "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'ok')"
step installing
(apt-get update -qq && apt-get install -y -qq git ffmpeg) >/dev/null 2>&1
# Run 84: a background install upgraded torch mid-run and broke torchaudio for both Wan models.
# Every install below is held to the image's own torch, torchvision and torchaudio.
python -c "import torch, torchvision, torchaudio as a; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\ntorchaudio=={a.__version__}')" > /workspace/pins.txt
cat /workspace/pins.txt
export PIP_CONSTRAINT=/workspace/pins.txt
pip install -q -U "git+https://github.com/huggingface/diffusers" transformers accelerate sentencepiece \
  ftfy imageio imageio-ffmpeg peft soundfile librosa 2>&1 | tail -n 3
step "downloading the models in the background"
(fetch @PICTURE_MODEL@ "" > dl_picture.log 2>&1; touch PICTURE_READY) &
(fetch @TALK_MODEL@ ", local_dir='@TALK_DIR@'" > dl_talk.log 2>&1; touch TALK_MODEL_READY) &
(fetch @CLIP_MODEL@ "" > dl_clip.log 2>&1; touch CLIP_READY) &
(
  git clone -q https://github.com/Wan-Video/Wan2.2 /workspace/Wan2.2
  # Its own environment, not sharing the system's packages: run 86's install crashed (pip
  # "expected string ... got 'NoneType'") while MMAudio was installing into the shared packages.
  unset PIP_CONSTRAINT
  python -m venv /workspace/wanenv
  /workspace/wanenv/bin/pip install -q -U pip
  TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  VV=$(python -c "import torchvision; print(torchvision.__version__.split('+')[0])")
  AV=$(python -c "import torchaudio; print(torchaudio.__version__.split('+')[0])")
  /workspace/wanenv/bin/pip install -q "torch==$TV" "torchvision==$VV" "torchaudio==$AV" \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple 2>&1 | tail -n 3
  printf 'torch==%s\ntorchvision==%s\ntorchaudio==%s\n' "$TV" "$VV" "$AV" > /tmp/wan-pins.txt
  cd /workspace/Wan2.2
  # One file at a time with a newline after each: run 85 glued them ("numpy>=1.23.5,<2decord"),
  # pip refused the whole list and wan could not import. No editable installs, no torch, and no
  # flash-attn (it would compile for many paid minutes; the worker falls back to PyTorch attention).
  for f in requirements.txt requirements_s2v.txt requirements_animate.txt; do
    [ -e "$f" ] && { cat "$f"; echo; }
  done | grep -vE '^\s*(#|-e|$)' | grep -viE '^(torch|torchvision|torchaudio|flash.attn|pyworld)([<>=! ]|$)' > /tmp/wan-req.txt
  # Run 87: pyworld (only for Wan's own text-to-speech, unused here) failed to compile, and pip
  # then installs nothing. It is left out, and if the list still fails, each package is
  # installed on its own so one bad package can never block the rest.
  if ! /workspace/wanenv/bin/pip install -q -c /tmp/wan-pins.txt -r /tmp/wan-req.txt 2>&1 | tail -n 8; then :; fi
  if ! /workspace/wanenv/bin/python -c "import easydict, decord, librosa" 2>/dev/null; then
    echo "installing Wan's packages one at a time"
    while read -r req; do
      /workspace/wanenv/bin/pip install -q -c /tmp/wan-pins.txt "$req" >/dev/null 2>&1 || echo "  not installed: $req"
    done < /tmp/wan-req.txt
  fi
  touch /workspace/job/WAN_ENV_READY
) > wan_env.log 2>&1 &
(
  git clone -q https://github.com/hkchengrex/MMAudio /workspace/MMAudio && pip install -q -e /workspace/MMAudio
  touch /workspace/job/MMAUDIO_READY
) > mmaudio.log 2>&1 &
step "voices (Chatterbox)"
# Its own environment with all its pinned packages, then a torch build that knows this GPU
# (Chatterbox pins an older torch without Blackwell support). Run 84 lacked librosa here.
python -m venv /workspace/tts
( unset PIP_CONSTRAINT
  /workspace/tts/bin/pip install -q chatterbox-tts soundfile 2>&1 | tail -n 3
  TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  AV=$(python -c "import torchaudio; print(torchaudio.__version__.split('+')[0])")
  # With its CUDA libraries this time (run 85: --no-deps left out libcufile, so torch would not load).
  /workspace/tts/bin/pip install -q "torch==$TV" "torchaudio==$AV" \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple 2>&1 | tail -n 3 )
/workspace/tts/bin/python -c "import chatterbox, librosa, torch; print('chatterbox ready, torch', torch.__version__)"
/workspace/tts/bin/python vast_worker.py job.json voices
for f in out/kline_*.wav; do t="${f/kline_/line_}"; [ -e "$t" ] || { cp "$f" "$t"; echo "line ${t##*/}: Kokoro"; }; done
step "waiting for the picture model"
while [ ! -e PICTURE_READY ]; do sleep 10; done
step pictures
python vast_worker.py job.json pictures
step "waiting for the talking model"
while [ ! -e TALK_MODEL_READY ] || [ ! -e WAN_ENV_READY ]; do sleep 10; done
tail -n 15 wan_env.log
# Run 85: every S2V shot failed while importing wan; show why before spending on it.
(cd /workspace/Wan2.2 && /workspace/wanenv/bin/python -c "import wan; print('wan imports fine')" 2>&1 | tail -n 25)
step "talking shots (Wan2.2-S2V)"
python vast_worker.py job.json talk
step "waiting for the clip model"
while [ ! -e CLIP_READY ]; do sleep 10; done
step "moving shots (Wan 2.2 I2V)"
python vast_worker.py job.json move
step "waiting for MMAudio"
while [ ! -e MMAUDIO_READY ]; do sleep 10; done
step sound
python vast_worker.py job.json sound
step finished
touch /workspace/job/DONE
"""


def remote_script(job: dict[str, Any]) -> str:
    return (REMOTE_SCRIPT.replace("@PICTURE_MODEL@", job["picture_model"]).replace("@CLIP_MODEL@", job["clip_model"])
            .replace("@TALK_MODEL@", job["talk_model"]).replace("@TALK_DIR@", job["talk_dir"]))


def talks(shot: dict[str, Any]) -> bool:
    """An on-screen speaker in a framing where the mouth reads: filmed lip-synced to the line."""
    return bool(str(shot.get("line", "")).strip() and shot.get("speaker") in (shot.get("characters") or [])
                and not shot.get("inner_voice") and shot.get("shot", "medium") in TALK_FRAMINGS)


def voice_owner(bible: dict[str, Any], name: str) -> str:
    """Whose voice a character speaks with (a double can share the original's)."""
    cast = {c["name"]: c for c in bible.get("cast", [])}
    return str(cast.get(name, {}).get("voice_of") or name)


def build_job(bible: dict[str, Any], shots: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    genre = bible.get("genre", "drama")
    refs = []
    cast = sorted(bible.get("cast", []), key=lambda c: bool(c.get("based_on")))  # bases first
    for c in cast:
        if c.get("based_on"):
            prompt = (f"Picture 1 shows {c['based_on']}. {STYLE}. Show this exact same person, the same face, "
                      f"features and build: {c.get('look', '')}. Wearing {c.get('wardrobe', '')}. Medium "
                      "close-up portrait, facing the camera, plain dark background.")
        else:
            prompt = character_prompt(c, genre)
        refs.append({"name": c["name"], "file": f"ref_{slug(c['name'])}.png", "prompt": prompt,
                     **({"based_on": c["based_on"]} if c.get("based_on") else {})})
    for loc in bible.get("locations", []):
        refs.append({"name": loc["name"], "file": f"ref_{slug(loc['name'])}.png", "prompt": location_prompt(loc, genre)})
    job: dict[str, Any] = {
        "picture_model": cfg.get("picture_model", PICTURE_MODEL), "clip_model": cfg.get("clip_model", CLIP_MODEL),
        "talk_model": cfg.get("talk_model", TALK_MODEL), "talk_dir": "/workspace/models/Wan2.2-S2V-14B",
        "wan_repo": "/workspace/Wan2.2", "wan_python": "/workspace/wanenv/bin/python",
        "picture_size": cfg.get("picture_size", [928, 1664]), "picture_steps": int(cfg.get("picture_steps", 40)),
        "clip_size": cfg.get("clip_size", [720, 1280]), "clip_steps": int(cfg.get("clip_steps", 30)),
        "talk_size": cfg.get("talk_size", "1024*704"), "talk_steps": int(cfg.get("talk_steps", 40)),
        "references": refs, "shots": [], "lines": [], "uploads": [],
        # MMAudio names its output file after the prompt; run 85's long prompt was too long a file name.
        "audio": [{"name": f"room_{slug(loc['name'])}", "duration": 10,
                   "prompt": (loc.get("sound") or f"{loc['name']} at night, quiet room tone")[:90]}
                  for loc in bible.get("locations", [])[:4]],
    }
    known = {r["name"] for r in refs}
    for i, shot in enumerate(shots):
        names = [n for n in shot.get("characters", []) if n in known][:2]
        if shot.get("location") in known:
            names.append(shot["location"])
        mention = " ".join(f"{{ref:{n}}} is the reference for {n}." for n in names)
        mode = "talk" if talks(shot) else "move"
        if mode == "talk":
            prompt = (f"{shot['speaker']} speaks with a {shot.get('emotion', 'tense')} expression, natural lip "
                      f"sync, blinking, subtle head and shoulder movement. {shot.get('action', '')}. Realistic, "
                      "cinematic, prestige TV drama, shallow depth of field, steady camera.")
        else:
            prompt = motion_prompt(shot)
        job["shots"].append({"index": i, "mode": mode, "refs": names,
                             "picture_prompt": f"{mention} {shot_prompt(shot, bible)}".strip(),
                             "motion_prompt": prompt, "seconds": hold(shot)})
        if str(shot.get("line", "")).strip() and shot.get("speaker"):
            owner = voice_owner(bible, shot["speaker"])
            job["lines"].append({"index": i, "text": str(shot["line"]).strip(), "emotion": shot.get("emotion", ""),
                                 "reference": f"voice_{slug(owner)}.wav"})
    return job


def hold(shot: dict[str, Any]) -> float:
    return float(shot.get("hold") or HOLDS.get(shot.get("shot", ""), 2.5))


def kokoro_voices(bible: dict[str, Any], kokoro: Any, folder: Path) -> dict[str, str]:
    """Each voice owner's fixed Kokoro voice, and a reference clip of it for Chatterbox to clone."""
    pools = {"female": ["af_heart", "bf_emma", "af_nicole", "af_sarah"], "male": ["bm_george", "am_michael", "am_fenrir"]}
    chosen: dict[str, str] = {}
    for c in bible.get("cast", []):
        if c.get("voice_of"):
            continue
        pool = [v for v in pools.get(c.get("gender", "male"), pools["male"]) if v not in chosen.values()]
        chosen[c["name"]] = c.get("voice") or (pool[0] if pool else "am_michael")
        kokoro.voice = chosen[c["name"]]
        soundfile.write(folder / f"voice_{slug(c['name'])}.wav", kokoro.say(REFERENCE_TEXT, 1.0), kokoro.sample_rate)
    return chosen


def _pad(audio: np.ndarray, rate: int) -> np.ndarray:
    piece = np.concatenate([np.zeros(int(0.15 * rate), np.float32), audio.astype(np.float32),
                            np.zeros(int(0.45 * rate), np.float32)])
    least = int(2.0 * rate)
    return np.concatenate([piece, np.zeros(max(0, least - piece.size), np.float32)])


def _read(path: Path) -> np.ndarray:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        audio = np.interp(np.linspace(0, audio.size - 1, int(audio.size * RATE / rate)), np.arange(audio.size),
                          audio).astype(np.float32)
    return audio


def produce(bible: dict[str, Any], episode: dict[str, Any], out_dir: Path, kokoro: Any, cfg: dict[str, Any],
            vast: Any, worker: Path, run_job=None, full_cfg: dict | None = None,
            reuse: Path | None = None) -> dict[str, Any]:
    from . import vast as vastmod

    started = time.time()
    run_job = run_job or vastmod.run_job
    out_dir = out_dir.resolve()
    work = out_dir / "work"
    upload = work / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    shots = episode.get("shots") or []
    if not shots:
        raise ValueError("the episode has no shots")

    # 1. Voices here: references to clone, and Kokoro lines in case Chatterbox can't read one.
    voices = kokoro_voices(bible, kokoro, upload)
    for i, shot in enumerate(shots):
        if str(shot.get("line", "")).strip() and shot.get("speaker"):
            kokoro.voice = voices.get(voice_owner(bible, shot["speaker"]), "am_michael")
            soundfile.write(upload / f"kline_{i:03d}.wav", _pad(kokoro.say(str(shot["line"]), 1.0), kokoro.sample_rate),
                            kokoro.sample_rate)

    # 2. The GPU.
    job = build_job(bible, shots, cfg)
    if reuse is not None and reuse.exists():  # pictures an earlier run already made, from the same prompts
        wanted = {r["file"] for r in job["references"]} | {f"shot_{s['index']:03d}.png" for s in job["shots"]}
        # Every clip and line too. A talking shot an earlier run could only make without lip sync
        # (no .talk marker) is still uploaded: the talking stage replaces it, and if S2V fails
        # again it stays as the moving fallback, so nothing is paid for twice.
        for s in job["shots"]:
            i = s["index"]
            wanted |= {f"clip_{i:03d}.mp4", f"clip_{i:03d}.talk", f"line_{i:03d}.wav"}
        kept = [p for p in reuse.rglob("*") if p.is_file() and p.name in wanted]
        for p in kept:
            shutil.copy(p, upload / p.name)
        log.info("Reusing %d pictures and clips from %s", len(kept), reuse)
    job["uploads"] = [str(p) for p in sorted(upload.glob("*.wav")) + sorted(upload.glob("*.png"))
                      + sorted(upload.glob("clip_*"))]
    gpu = run_job(vast, job, worker, work / "gpu", float(cfg.get("max_usd", 15.0)),
                  disk=int(cfg.get("disk", 450)), max_hours=float(cfg.get("max_hours", 5.0)),
                  script=remote_script(job))
    made = work / "gpu" / "out"
    made.mkdir(parents=True, exist_ok=True)

    # 3. Each shot's line and length.
    lines: list[np.ndarray] = []
    durations: list[float] = []
    engines: dict[str, int] = {"chatterbox": 0, "kokoro": 0}
    for i, shot in enumerate(shots):
        if str(shot.get("line", "")).strip() and shot.get("speaker"):
            gpu_line, fallback = made / f"line_{i:03d}.wav", upload / f"kline_{i:03d}.wav"
            cloned = gpu_line.exists() and not filecmp.cmp(gpu_line, fallback, shallow=False)
            audio = _read(gpu_line if gpu_line.exists() else fallback)
            engines["chatterbox" if cloned else "kokoro"] += 1
            if shot.get("inner_voice"):  # a thought: softer, with a little room
                echo = np.zeros_like(audio)
                lag = int(0.06 * RATE)
                echo[lag:] = audio[:-lag] * 0.35
                audio = (audio + echo) * 0.75
            lines.append(audio)
        else:
            lines.append(np.zeros(int(hold(shot) * RATE), np.float32))
        durations.append(lines[-1].size / RATE)

    # 4. Pictures (a missing one borrows its nearest neighbour) and clips.
    pictures: dict[int, Path] = {i: made / f"shot_{i:03d}.png" for i in range(len(shots))
                                 if (made / f"shot_{i:03d}.png").exists()}
    if not pictures:
        raise RuntimeError(f"the GPU made no pictures; see the log: {str(gpu.get('log', ''))[-1500:]}")
    for i in range(len(shots)):
        if i not in pictures:
            pictures[i] = pictures[min(pictures, key=lambda j: abs(j - i))]
    for i in range(len(shots)):  # a run stopped by an older worker mid-shot left the moving clip aside
        if not (made / f"clip_{i:03d}.mp4").exists() and (made / f"clip_{i:03d}.move.mp4").exists():
            (made / f"clip_{i:03d}.move.mp4").replace(made / f"clip_{i:03d}.mp4")
    clips = {i: made / f"clip_{i:03d}.mp4" for i in range(len(shots)) if (made / f"clip_{i:03d}.mp4").exists()}
    talking = sorted(i for i in clips if (made / f"clip_{i:03d}.talk").exists())  # made by S2V itself

    # 5. Sound: the lines, the room, the bed.
    track = np.concatenate(lines)
    starts = np.cumsum([0.0] + durations[:-1]).tolist()
    speech = [(s + 0.12, s + d - 0.28) for s, d, shot in zip(starts, durations, shots) if shot.get("line")]
    scenes = [{"mood": shot.get("emotion", ""), "sfx": []} for shot in shots]
    mixed = sound.mix(track, RATE, scenes, list(zip(starts, [s + d for s, d in zip(starts, durations)])), speech)
    room = sorted(made.glob("room_*.flac"))
    if room:
        amb = _read(room[0])
        amb = np.tile(amb, int(np.ceil(mixed.size / max(amb.size, 1))))[: mixed.size]
        amb *= 0.12 / (float(np.max(np.abs(amb))) or 1.0)
        mixed = mixed + amb
        mixed *= min(1.0, 0.95 / (float(np.max(np.abs(mixed))) or 1.0))
    soundfile.write(work / "mix.wav", mixed.astype(np.float32), RATE)

    # 6. The edit.
    editor = Producer(None, full_cfg or {"drama_video": {}}, work, None, kokoro)
    (work / "subs.ass").write_text(subtitles(shots, starts, durations, (editor.w, editor.h)))
    parts = [editor.clip(work, i, shot, pictures[i], clips.get(i), durations[i]) for i, shot in enumerate(shots)]
    (work / "clips.txt").write_text("".join(f"file '{c}'\n" for c in parts))
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", "clips.txt", "-i", "mix.wav", "-vf", "subtitles=subs.ass",
             "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-b:a", "192k", "-shortest", "-movflags", "+faststart", "episode.mp4"], work)
    shutil.move(str(work / "episode.mp4"), out_dir / "episode.mp4")
    reveal = next((i for i in range(len(shots) - 1, -1, -1) if shots[i].get("shot") in ("wide", "close_up")), 0)
    shutil.copy(pictures[reveal], out_dir / "thumbnail.png")
    review = out_dir / "pictures"
    review.mkdir(exist_ok=True)
    for p in list(made.glob("ref_*.png")) + list(made.glob("shot_*.png")):
        shutil.copy(p, review / p.name)
    raw = out_dir / "raw"  # the GPU's own clips and lines, so a later run can reuse them (reuse_run)
    raw.mkdir(exist_ok=True)
    for p in list(made.glob("clip_*")) + list(made.glob("line_*.wav")):
        shutil.copy(p, raw / p.name)

    summary = {
        "mode": "drama-vast", "title": episode.get("title"), "series": bible.get("title"),
        "seconds": round(sum(durations), 1), "shots": len(shots), "pictures_from_gpu": len(list(made.glob("shot_*.png"))),
        "moving_shots": len(clips), "lip_synced_shots": len(talking), "stills_with_camera_moves": len(shots) - len(clips),
        "voices": engines, "kokoro_voices": voices, "gpu": {k: v for k, v in gpu.items() if k != "log"},
        "usd": gpu.get("usd", 0.0), "render_minutes": round((time.time() - started) / 60, 1),
        "models": {"voices": "Chatterbox (cloned from Kokoro reference voices)", "pictures": job["picture_model"],
                   "talking": job["talk_model"], "moving": job["clip_model"], "ambience": "MMAudio"},
    }
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "gpu.log").write_text(str(gpu.get("log", "")))
    if not cfg.get("keep_work"):
        shutil.rmtree(work, ignore_errors=True)
    return summary
