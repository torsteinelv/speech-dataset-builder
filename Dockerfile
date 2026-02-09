# Bruk et rent, lett Python-image i stedet for det gamle PyTorch-imaget
FROM python:3.10-slim

WORKDIR /app

# Installer systemavhengigheter (git for whisperx, ffmpeg for lyd)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Kopier og installer python-pakker
COPY requirements.txt .
# Oppgrader pip først for å håndtere moderne wheels korrekt
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Kopier kildekoden
COPY src/ ./src/

CMD ["python", "src/main.py"]
