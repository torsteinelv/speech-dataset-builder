# Vi bruker en offisiell PyTorch image med CUDA 11.8 eller 12.1 støtte
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Sett arbeidsmappe
WORKDIR /app

# Installer systemavhengigheter (git og ffmpeg er kritiske)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Kopier requirements og installer python-pakker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopier kildekoden
COPY src/ ./src/

# Standard kommando når containeren starter
CMD ["python", "src/main.py"]
