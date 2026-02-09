# Vi hopper helt frem til Ubuntu 24.04 og CUDA 12.4.1
# Dette er optimalt for RTX 4000 Ada-serien.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

# 1. Installer Python 3.12 (standard i Ubuntu 24.04) og verktøy
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

# 2. Installer pakker
# Vi fjerner '--upgrade pip' her da Ubuntu 24.04 er veldig streng på system-python.
# Vi bruker --break-system-packages fordi vi er inne i en container (det er trygt).
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/main.py"]
