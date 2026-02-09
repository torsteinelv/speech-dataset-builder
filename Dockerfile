# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models_cache \
    TRANSFORMERS_CACHE=/models_cache \
    XDG_CACHE_HOME=/models_cache

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg git curl ca-certificates \
    build-essential pkg-config \
    libsndfile1 \
    patchelf \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer python deps
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install -r /app/requirements.txt

# ------------------------------------------------------------
# FIX: legg inn cuDNN 8 runtime libs side-by-side
# (for pakker som forventer libcudnn_ops_infer.so.8)
# Vi installerer via NVIDIA sin pip-index, og kopierer kun *.so.8*
# til en separat mappe for å unngå å overstyre cuDNN9.
# ------------------------------------------------------------
RUN python3 -m pip install --extra-index-url https://pypi.nvidia.com "nvidia-cudnn-cu12==8.9.7.29" \
 && python3 - <<'PY'
import sysconfig, glob, os, shutil

purelib = sysconfig.get_paths()["purelib"]

# Finn cuDNN8 bibliotekene som ble installert av nvidia-cudnn-cu12
hits = glob.glob(os.path.join(purelib, "**", "libcudnn_ops_infer.so.8*"), recursive=True)
if not hits:
    hits = glob.glob(os.path.join(purelib, "**", "libcudnn*.so.8*"), recursive=True)

if not hits:
    raise RuntimeError(
        "Fant ikke cuDNN8-bibliotek etter install av nvidia-cudnn-cu12. "
        "Sjekk egress/tilgang til https://pypi.nvidia.com"
    )

libdir = os.path.dirname(hits[0])
outdir = "/usr/local/lib/cudnn8"
os.makedirs(outdir, exist_ok=True)

# Kopier kun cuDNN8-filer (so.8*) til outdir
copied = 0
for f in glob.glob(os.path.join(libdir, "libcudnn*.so.8*")):
    if os.path.isfile(f):
        shutil.copy2(f, outdir)
        copied += 1

print(f"cuDNN8: kopierte {copied} filer til {outdir}")
PY \
 && echo "/usr/local/lib/cudnn8" > /etc/ld.so.conf.d/cudnn8.conf \
 && ldconfig

# ------------------------------------------------------------
# (Valgfritt, men anbefalt): Fjern execstack-flag fra CTranslate2 .so
# for å unngå 'cannot enable executable stack' på hardnede kernels.
# ------------------------------------------------------------
RUN python3 - <<'PY'
import sysconfig, glob, os, subprocess

purelib = sysconfig.get_paths()["purelib"]
libs = glob.glob(os.path.join(purelib, "**", "libctranslate2*.so*"), recursive=True)

if not libs:
    print("Fant ingen libctranslate2*.so* - hopper over execstack patch")
else:
    patched = 0
    for so in libs:
        # patchelf returnerer !=0 om det ikke er ELF; vi ignorerer det.
        subprocess.run(["patchelf", "--clear-execstack", so], check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        patched += 1
    print(f"Patchet execstack på {patched} filer")
PY

# Entrypoint som verifiserer cuDNN8-fila før appen starter
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Kopier resten av appen
COPY . /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-u", "/app/src/main.py"]
