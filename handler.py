"""RunPod Serverless worker: Wolof text -> WhatsApp voice note (OGG/Opus), with Oolel-Voices.

Input  {"text": str, "voice_b64": str (reference recording, any format ffmpeg reads),
        optional tuning (clamped to safe ranges): "exaggeration", "cfg_weight", "temperature" (0.1-1.0),
        "pause" (seconds between sentence groups, 0.1-1.0), "ref_start" and "ref_seconds" (which part of
        the reference recording to imitate, in seconds)}
Output {"audio_b64": str (audio/ogg; codecs=opus), "seconds": float}

Privacy: the text and the voice are health/personal data. Nothing is written to disk except
temporary files deleted right after use, and nothing about the content is logged.
"""

import base64
import hashlib
import os
import re
import subprocess
import tempfile
import time

import runpod
import torch

MODEL_DIR = os.environ.get("OOLEL_MODEL_DIR", "/models/oolel")
MAX_CHARS = 2500
CHUNK_CHARS = 280  # the model emits at most ~40 s per call: speak long replies sentence group by group

import sys  # noqa: E402

sys.path.insert(0, MODEL_DIR)
import modeling_oolel_voices as oolel  # noqa: E402  (pinned, reviewed copy baked into the image)

MODEL = oolel.OolelVoicesForInference.from_pretrained(MODEL_DIR, device_map="cuda" if torch.cuda.is_available() else "cpu")
_voice_sha: str | None = None  # conditioning is cached per worker for the last voice + settings used

DEFAULTS = {"exaggeration": 0.3, "cfg_weight": 0.5, "temperature": 0.3, "pause": 0.35, "ref_start": 0.0, "ref_seconds": 12.0}
LIMITS = {"exaggeration": (0.1, 1.0), "cfg_weight": (0.1, 1.0), "temperature": (0.1, 1.0), "pause": (0.1, 1.0),
          "ref_start": (0.0, 60.0), "ref_seconds": (5.0, 20.0)}


def _settings(inp: dict) -> dict:
    out = {}
    for k, default in DEFAULTS.items():
        try:
            v = float(inp.get(k, default))
        except (TypeError, ValueError):
            v = default
        lo, hi = LIMITS[k]
        out[k] = min(hi, max(lo, v))
    return out


def _chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > CHUNK_CHARS:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _prepare_voice(voice_b64: str, workdir: str, cfg: dict) -> None:
    global _voice_sha
    raw = base64.b64decode(voice_b64)
    sha = hashlib.sha256(raw + repr((cfg["exaggeration"], cfg["ref_start"], cfg["ref_seconds"])).encode()).hexdigest()
    if sha == _voice_sha:
        return
    src, wav = os.path.join(workdir, "voice_in"), os.path.join(workdir, "voice.wav")
    with open(src, "wb") as f:
        f.write(raw)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(cfg["ref_start"]), "-t", str(cfg["ref_seconds"]),
                    "-i", src, "-ac", "1", "-ar", "24000", wav], check=True)
    MODEL.prepare_conditionals(wav, exaggeration=cfg["exaggeration"])
    _voice_sha = sha


def handler(job):
    inp = job.get("input") or {}
    text, voice_b64 = (inp.get("text") or "").strip(), inp.get("voice_b64") or ""
    if not text or len(text) > MAX_CHARS:
        return {"error": f"text must be 1-{MAX_CHARS} characters"}
    if not voice_b64:
        return {"error": "voice_b64 is required"}
    cfg = _settings(inp)
    started = time.time()
    with tempfile.TemporaryDirectory() as workdir:
        _prepare_voice(voice_b64, workdir, cfg)
        pieces = []
        with torch.inference_mode():
            for chunk in _chunks(text):
                pieces.append(MODEL.generate(chunk, cfg_weight=cfg["cfg_weight"], exaggeration=cfg["exaggeration"],
                                             temperature=cfg["temperature"]).cpu())
                pieces.append(torch.zeros(1, int(MODEL.sr * cfg["pause"])))  # short pause between chunks
        wav = torch.cat(pieces, dim=-1)
        pcm = os.path.join(workdir, "out.f32")
        wav.squeeze(0).numpy().astype("float32").tofile(pcm)
        ogg = os.path.join(workdir, "out.ogg")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(MODEL.sr), "-ac", "1", "-i", pcm,
             "-c:a", "libopus", "-b:a", "32k", "-ar", "48000", "-application", "voip", ogg],
            check=True,
        )
        with open(ogg, "rb") as f:
            audio = f.read()
    seconds = wav.shape[-1] / MODEL.sr
    print(f"generated {seconds:.1f}s of audio in {time.time() - started:.1f}s ({len(text)} chars)")  # no content
    return {"audio_b64": base64.b64encode(audio).decode(), "seconds": round(seconds, 1)}


runpod.serverless.start({"handler": handler})
