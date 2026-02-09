FROM python:3.10-slim

WORKDIR /app

# Installer systemavhengigheter + pax-utils (for scanelf)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    ffmpeg \
    tzdata \
    pax-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# --- FIKS: Kjører scanelf på HELE site-packages for å være sikker ---
# Dette fjerner "executable stack" flagget fra alle biblioteker som måtte ha det
RUN scanelf --recursive --clear-execstack /usr/local/lib/python3.10/site-packages || true

COPY src/ ./src/

CMD ["python", "src/main.py"]
