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



COPY src/ ./src/

CMD ["python", "src/main.py"]
