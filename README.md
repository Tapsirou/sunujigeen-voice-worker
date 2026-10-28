# Suñu Jigeen – Wolof voice worker

RunPod Serverless worker that turns a Wolof text into a WhatsApp voice note (OGG/Opus), for the
Suñu Jigeen Santé WhatsApp assistant.

- Model: [Oolel-Voices](https://huggingface.co/soynade-research/Oolel-Voices) by Soynade Research,
  used **unmodified**, pinned to revision `fb8a693e0d86e00672a04810cc5f331789cc0fb4`.
- Input: `{"text": "...", "voice_b64": "<reference recording, base64>"}`
- Output: `{"audio_b64": "<OGG/Opus, base64>", "seconds": 12.3}`

The image contains no secrets and no voice recording: the reference voice is sent by the app with
each request. Nothing about the text or the voice is stored or logged.

## Deploy on RunPod

Serverless → New Endpoint → GitHub repo → this repository, branch `main`, Dockerfile `Dockerfile`.
GPU with 16 GB or more, 0 active workers (scale to zero), execution timeout 300 s.

## License

Oolel-Voices is distributed under the **GNU AGPL-3.0**. This worker uses it without modification;
its source code is available at the link above. The code in this repository (handler.py,
Dockerfile) is also published under the AGPL-3.0.
