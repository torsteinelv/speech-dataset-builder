import feedparser
import requests
import os
import re
import boto3
import argparse
import html
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- KONFIGURASJON ---
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Prosjektmappe
S3_BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
S3_TARGET_ROOT = f"{S3_BASE_PATH}/raw/" 

# Navnet på filen i S3 som holder oversikten
SUBSCRIPTION_FILE_KEY = f"{S3_TARGET_ROOT}subscriptions.txt"

TEMP_DIR = Path("temp_downloads")

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def clean_filename(title):
    title = html.unescape(title)
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
    cleaned = cleaned.strip().replace(" ", "_")
    return re.sub(r'_+', '_', cleaned)

# --- ABONNEMENTS-HÅNDTERING ---
def load_subscriptions(s3):
    """Laster ned listen over lagrede podkaster fra S3."""
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=SUBSCRIPTION_FILE_KEY)
        content = response['Body'].read().decode('utf-8')
        urls = {line.strip() for line in content.splitlines() if line.strip()}
        print(f"--- Hentet {len(urls)} abonnementer fra S3 ---")
        return urls
    except s3.exceptions.NoSuchKey:
        print("--- Ingen abonnementer funnet i S3 (ny fil vil bli opprettet) ---")
        return set()
    except Exception as e:
        print(f"Advarsel: Kunne ikke lese subscriptions.txt: {e}")
        return set()

def save_subscription(s3, new_url):
    """Legger til en ny URL i listen og laster opp til S3."""
    current_urls = load_subscriptions(s3)
    
    if new_url in current_urls:
        return # URLen finnes allerede, trenger ikke gjøre noe

    current_urls.add(new_url)
    
    # Lagre tilbake til S3
    file_content = "\n".join(sorted(current_urls))
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=SUBSCRIPTION_FILE_KEY, Body=file_content.encode('utf-8'))
        print(f"--- Lagret ny podkast i listen på S3 ---")
    except Exception as e:
        print(f"FEIL: Klarte ikke oppdatere abonnement-listen i S3: {e}")

# --- HOVEDLOGIKK ---
def process_single_feed(rss_url, s3):
    TEMP_DIR.mkdir(exist_ok=True)

    print(f"\nSjekker RSS: {rss_url} ...")
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"Klarte ikke lese RSS: {e}")
        return

    if not feed.entries:
        print(" -> Ingen episoder funnet.")
        return

    podcast_title = clean_filename(feed.feed.get('title', 'Ukjent_Podcast'))
    print(f" -> Podcast: {podcast_title} ({len(feed.entries)} episoder)")

    for entry in feed.entries:
        episode_title = clean_filename(entry.title)
        
        mp3_url = None
        for link in entry.links:
            if link.rel == 'enclosure' and 'audio' in link.type:
                mp3_url = link.href
                break
        
        if not mp3_url: continue

        filename = f"{episode_title}.mp3"
        s3_key = f"{S3_TARGET_ROOT}{podcast_title}/{filename}"
        local_path = TEMP_DIR / filename

        try:
            s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            # print(f"SKIP (Finnes): {filename}") 
            continue
        except:
            pass 

        print(f" -> LASTER NED: {episode_title} ...")
        try:
            with requests.get(mp3_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print(f"    -> Laster opp til S3...")
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)

        except Exception as e:
            print(f"    -> FEILET: {e}")
        
        finally:
            if local_path.exists(): local_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="Last ned podkaster til S3.")
    parser.add_argument("--url", type=str, help="Legg til og last ned en ny RSS feed")
    parser.add_argument("--file", type=str, help="Importer en liste med RSS feeder fra lokal fil")
    args = parser.parse_args()

    if not S3_BUCKET:
        print("FEIL: Mangler S3_BUCKET i .env")
        return

    s3 = get_s3_client()

    # SCENARIO 1: Brukeren vil legge til en ny podcast
    if args.url:
        print(f"Starter behandling av ny podcast...")
        process_single_feed(args.url, s3)
        save_subscription(s3, args.url)
    
    # SCENARIO 2: Brukeren vil importere fra fil
    elif args.file:
        file_path = Path(args.file)
        if file_path.exists():
            with open(file_path, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
            for url in urls:
                process_single_feed(url, s3)
                save_subscription(s3, url)
    
    # SCENARIO 3: Ingen argumenter - oppdater alle abonnementer
    else:
        saved_urls = load_subscriptions(s3)
        if not saved_urls:
            print("Ingen lagrede podkaster funnet.")
            print("Bruk --url '...' for å legge til den første!")
        else:
            print(f"Sjekker {len(saved_urls)} lagrede podkaster for nye episoder...")
            for url in saved_urls:
                process_single_feed(url, s3)

    # Rydd opp
    if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
        TEMP_DIR.rmdir()
    print("\nFerdig!")

if __name__ == "__main__":
    main()
