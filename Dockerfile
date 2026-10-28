# Wolof voice worker for RunPod Serverless (GPU). Contains only the public model and this handler:
# no secrets, no voice recording (the app sends the voice with each request).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Oolel-Voices, pinned to the revision reviewed on 2026-10-27 (license AGPL-3.0, see LICENSE notice in README.md).
ARG OOLEL_REVISION=fb8a693e0d86e00672a04810cc5f331789cc0fb4
# The Wolof fine-tuned weights are the ones at the repository root (t3_cfg.safetensors, sha256 checked below).
# pretrained_model_download/ holds the base model before Wolof training: do NOT use it.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('soynade-research/Oolel-Voices', revision='${OOLEL_REVISION}', local_dir='/models/oolel', \
allow_patterns=['*.py', '*.json', 'models/**', 't3_cfg.safetensors', 's3gen.safetensors', 've.safetensors', 'conds.pt'])" \
 && sha256sum /models/oolel/t3_cfg.safetensors | grep -q '^7c3e0ab39b32e45c88a8cfe8844d8dd244c612c697b5623d0a82e511b7d19b64' \
 || (echo 'Wrong t3_cfg.safetensors: expected the Wolof fine-tuned weights' && exit 1)

# Everything the model needs is in the image: no downloads at run time.
ENV HF_HUB_OFFLINE=1 OOLEL_MODEL_DIR=/models/oolel
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
