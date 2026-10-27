# Wolof voice worker for RunPod Serverless (GPU). Contains only the public model and this handler:
# no secrets, no voice recording (the app sends the voice with each request).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Oolel-Voices, pinned to the revision reviewed on 2026-10-27 (license AGPL-3.0, see LICENSE notice in README.md).
ARG OOLEL_REVISION=fb8a693e0d86e00672a04810cc5f331789cc0fb4
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('soynade-research/Oolel-Voices', revision='${OOLEL_REVISION}', local_dir='/models/oolel', \
allow_patterns=['*.py', '*.json', 'models/**', 'pretrained_model_download/**'])" \
 && cd /models/oolel && for f in pretrained_model_download/*; do ln -s "$f" "$(basename "$f")"; done

# Everything the model needs is in the image: no downloads at run time.
ENV HF_HUB_OFFLINE=1 OOLEL_MODEL_DIR=/models/oolel
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
