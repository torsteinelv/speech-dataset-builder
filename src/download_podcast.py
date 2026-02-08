import feedparser
import requests
import os
import re
import boto3
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- KONFIGURASJON ---
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Hent prosjektmappen fra .env (Default: 002_speech_dataset)
S3_BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
S3_TARGET_ROOT = f"{S3_BASE_PATH}/raw/" 

TEMP_DIR = Path("temp_downloads")

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def clean_filename(title):
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
    return cleaned.strip().replace(" ", "_")

def process_single_feed(rss_url):
    if not S3_BUCKET:
        print("FEIL: Mangler S3_BUCKET i .env")
        return

    s3 = get_s3_client()
    TEMP_DIR.mkdir(exist_ok=True)

    print(f"--- Behandler RSS: {rss_url} ---")
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"Klarte ikke lese RSS: {e}")
        return

    if not feed.entries:
        print("Fant ingen episoder eller ugyldig RSS.")
        return

    podcast_title = clean_filename(feed.feed.get('title', 'Ukjent_Podcast'))
    print(f"Podcast: {podcast_title} ({len(feed.entries)} episoder)")

    for entry in feed.entries:
        episode_title = clean_filename(entry.title)
        
        mp3_url = None
        for link in entry.links:
            if link.rel == 'enclosure' and 'audio' in link.type:
                mp3_url = link.href
                break
        
        if not mp3_url: continue

        filename = f"{episode_title}.mp3"
        # Struktur: 002_speech_dataset/raw/PodkastNavn/Episode.mp3
        s3_key = f"{S3_TARGET_ROOT}{podcast_title}/{filename}"
        local_path = TEMP_DIR / filename

        try:
            s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            # print(f"SKIP (Finnes): {filename}") 
            continue
        except:
            pass 

        print(f"LASTER NED: {episode_title} ...")
        try:
            with requests.get(mp3_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print(f" -> Laster opp til: {s3_key}")
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)

        except Exception as e:
            print(f" -> FEILET: {e}")
        
        finally:
            if local_path.exists(): local_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="Last ned podkaster til S3.")
    parser.add_argument("--url", type=str, help="URL til én RSS feed")
    parser.add_argument("--file", type=str, help="Fil med liste over RSS feeder")
    args = parser.parse_args()

    if args.url:
        process_single_feed(args.url)
    elif args.file:
        file_path = Path(args.file)
        if file_path.exists():
            with open(file_path, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
            for url in urls:
                process_single_feed(url)
    else:
        print("Bruk --url eller --file.")

    if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
        TEMP_DIR.rmdir()

if __name__ == "__main__":
    main()
