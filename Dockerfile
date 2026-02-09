# Vi bruker slim-versjonen for et mindre og raskere image
FROM python:3.10-slim

WORKDIR /app

# Installer systemavhengigheter
# Vi erstatter 'execstack' med 'pax-utils' som er tilgjengelig i nyere Debian
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    ffmpeg \
    tzdata \
    pax-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Oppgrader pip først for å håndtere moderne wheels korrekt
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# --- VIKTIG FIX: Fjern "executable stack" kravet fra CTranslate2 ---
# scanelf --clear-execstack gjør det samme som execstack -c gjorde før, og er støttet i Debian 11+
RUN find /usr/local/lib/python3.10/site-packages -name "libctranslate2*.so*" -exec scanelf --clear-execstack {} \;

COPY src/ ./src/

CMD ["python", "src/main.py"]
