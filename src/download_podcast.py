import feedparser
import requests
import os
import re
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- KONFIGURASJON ---
RSS_URL = os.getenv("RSS_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BASE_FOLDER = "raw/"  # Rot-mappen for all lyd

# Temp mappe
TEMP_DIR = Path("temp_downloads")

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def clean_filename(title):
    # Fjerner ugyldige tegn og mellomrom
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
    return cleaned.strip().replace(" ", "_") # Bytter mellomrom med understrek for sikkerhets skyld

def download_and_upload():
    if not RSS_URL or not S3_BUCKET:
        print("FEIL: Mangler RSS_URL eller S3_BUCKET i .env")
        return

    s3 = get_s3_client()
    TEMP_DIR.mkdir(exist_ok=True)

    print(f"Henter RSS: {RSS_URL} ...")
    feed = feedparser.parse(RSS_URL)

    if not feed.entries:
        print("Fant ingen episoder.")
        return

    # 1. Hent og vask podcast-navnet for å bruke som mappenavn
    podcast_title = clean_filename(feed.feed.get('title', 'Ukjent_Podcast'))
    print(f"Podcast: {podcast_title}")
    print(f"Fant {len(feed.entries)} episoder.")

    for entry in feed.entries:
        episode_title = clean_filename(entry.title)
        
        mp3_url = None
        for link in entry.links:
            if link.rel == 'enclosure' and 'audio' in link.type:
                mp3_url = link.href
                break
        
        if not mp3_url: continue

        filename = f"{episode_title}.mp3"
        
        # HER ER MAGIEN: Vi legger podcast-navnet inn i stien
        # Resultat: raw/Min_Podcast/Min_Episode.mp3
        s3_key = f"{S3_BASE_FOLDER}{podcast_title}/{filename}"
        
        local_path = TEMP_DIR / filename

        # Sjekk om filen finnes i S3 (så vi slipper å laste ned på nytt)
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            print(f"SKIP (Finnes): {podcast_title}/{filename}")
            continue
        except:
            pass # Filen finnes ikke, fortsett

        print(f"LASTER NED: {episode_title} ...")
        try:
            # Last ned lokalt først
            with requests.get(mp3_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Last opp til riktig undermappe i S3
            print(f" -> Laster opp til: {s3_key}")
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)

        except Exception as e:
            print(f" -> FEILET: {e}")
        
        finally:
            if local_path.exists(): local_path.unlink()

    if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
        TEMP_DIR.rmdir()
    print("\nFerdig!")

if __name__ == "__main__":
    download_and_upload()
