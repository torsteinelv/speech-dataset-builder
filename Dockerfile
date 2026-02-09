# Vi bytter til 'bullseye' fordi den har verktøyet 'execstack' tilgjengelig
FROM python:3.10-bullseye

WORKDIR /app

# Installer systemavhengigheter + execstack
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    ffmpeg \
    tzdata \
    execstack \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# --- VIKTIG FIX: Fjern "executable stack" kravet fra CTranslate2 ---
# Dette søker etter biblioteket som feilet og fjerner flagget som Kubernetes blokkerer
RUN find /usr/local/lib/python3.10/site-packages -name "libctranslate2*.so*" -exec execstack -c {} \;

COPY src/ ./src/

CMD ["python", "src/main.py"]
