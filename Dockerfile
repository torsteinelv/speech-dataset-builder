# VIKTIG: Vi bytter til et image som allerede har ALT av GPU-biblioteker installert
# Dette inneholder Python, PyTorch, CUDA 11.8 og cuDNN 8
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Vi må fortsatt installere git og ffmpeg (for lyd), og pax-utils for patchen vår
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    ffmpeg \
    tzdata \
    pax-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installer Python-pakker (WhisperX osv)
# Vi bruker --no-deps på torch-relaterte ting hvis mulig, men pip håndterer dette greit nå
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# --- SIKKERHETS-PATCH (Execstack) ---
# Selv på proffe images kan CTranslate2 krangle med Kubernetes sikkerhet.
# Vi kjører denne under bygging for å være 100% sikre.
RUN LIB_FILE=$(find /opt/conda/lib/python3.10/site-packages -name "libctranslate2*.so*" | head -n 1) && \
    if [ -n "$LIB_FILE" ]; then \
        echo "Patcher $LIB_FILE"; \
        scanelf --clear-execstack "$LIB_FILE"; \
    fi

COPY src/ ./src/

CMD ["python", "src/main.py"]
