# Vi bruker nyeste STABILE versjon: CUDA 12.6.3 (CUDA 13 finnes ikke enda)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

# 1. Installerer Python og verktøy
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    tzdata \
    pax-utils \
    && rm -rf /var/lib/apt/lists/*

# Fix for å bruke 'python' kommandoen
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

# 2. Installer pakker - MED TVANGSFLAGG (--break-system-packages)
# Dette er nøkkelen for å få det til å virke på Ubuntu 24.04
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY src/ ./src/

CMD ["python", "src/main.py"]
