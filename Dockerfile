# Vi bruker en offisiell PyTorch image med CUDA 11.8 eller 12.1 støtte
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Sett arbeidsmappe
WORKDIR /app

# VIKTIG ENDRING HER:
# Vi setter DEBIAN_FRONTEND=noninteractive for å unngå at tzdata stopper bygget
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Kopier requirements og installer python-pakker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopier kildekoden
COPY src/ ./src/

# Standard kommando når containeren starter
CMD ["python", "src/main.py"]
