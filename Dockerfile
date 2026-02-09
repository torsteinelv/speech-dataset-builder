# Vi bruker NVIDIAs offisielle image. Dette har cuDNN 8 ferdig installert i systemstiene.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 1. Installerer Python 3.10 og nødvendige verktøy
# (Ubuntu 22.04 kommer med python3.10 som standard)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    tzdata \
    pax-utils \
    && rm -rf /var/lib/apt/lists/*

# Lag en symlink så "python" peker på "python3"
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

# 2. Installer Python-pakker
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# 3. SIKKERHETS-PATCH (Execstack)
# Vi kjører denne for sikkerhets skyld, selv om NVIDIA-imaget ofte er snillere.
# Denne søker gjennom både dist-packages og site-packages for å finne synderen.
RUN LIB_FILE=$(find /usr/local/lib/python3.10 /usr/lib/python3 -name "libctranslate2*.so*" 2>/dev/null | head -n 1) && \
    if [ -n "$LIB_FILE" ]; then \
        echo "Patcher $LIB_FILE"; \
        scanelf --clear-execstack "$LIB_FILE"; \
    fi

COPY src/ ./src/

CMD ["python", "src/main.py"]
