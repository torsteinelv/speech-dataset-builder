# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models_cache \
    TRANSFORMERS_CACHE=/models_cache \
    XDG_CACHE_HOME=/models_cache \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      ffmpeg git curl ca-certificates \
      build-essential pkg-config \
      libsndfile1 \
      patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" \
 && "$VIRTUAL_ENV/bin/python" -m ensurepip --upgrade \
 && "$VIRTUAL_ENV/bin/python" -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

ARG CUDNN8_WHEEL_VERSION=8.9.7.29
RUN <<'BASH'
set -euo pipefail

mkdir -p /tmp/cudnn8
pip download --no-deps --extra-index-url https://pypi.nvidia.com \
  -d /tmp/cudnn8 "nvidia-cudnn-cu12==${CUDNN8_WHEEL_VERSION}"

python - <<'PY'
import glob, os, zipfile, shutil

whls = glob.glob("/tmp/cudnn8/nvidia_cudnn_cu12-*.whl")
if not whls:
    raise SystemExit("Fant ikke nvidia_cudnn_cu12 wheel i /tmp/cudnn8")

whl = whls[0]
extract_dir = "/tmp/cudnn8/extract"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(whl) as z:
    z.extractall(extract_dir)

libs = sorted({
    p for p in glob.glob(os.path.join(extract_dir, "**", "libcudnn*.so.8*"), recursive=True)
    if os.path.isfile(p)
})
if not libs:
    raise SystemExit("Fant ingen libcudnn*.so.8* inne i cudnn8 wheel")

outdir = "/usr/local/lib/cudnn8"
os.makedirs(outdir, exist_ok=True)
for p in libs:
    shutil.copy2(p, outdir)

print(f"cuDNN8: kopierte {len(libs)} filer til {outdir}")
PY

echo "/usr/local/lib/cudnn8" > /etc/ld.so.conf.d/cudnn8.conf
ldconfig

python - <<'PY'
import ctypes
ctypes.CDLL("libcudnn_ops_infer.so.8")
print("OK: libcudnn_ops_infer.so.8 kan lastes")
PY

rm -rf /tmp/cudnn8
BASH

RUN python - <<'PY'
import sysconfig, glob, os, subprocess
purelib = sysconfig.get_paths()["purelib"]
libs = glob.glob(os.path.join(purelib, "**", "libctranslate2*.so*"), recursive=True)
for so in libs:
    subprocess.run(["patchelf", "--clear-execstack", so], check=False,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(f"Patchet execstack på {len(libs)} filer")
PY

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY . /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-u", "/app/src/main.py"]
