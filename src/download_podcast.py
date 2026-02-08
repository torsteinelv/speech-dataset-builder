import feedparser
import requests
import os
import re
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()

# --- KONFIGURASJON ---
RSS_URL = os.getenv("RSS_URL")
# S3 / Ceph Config
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_Target_Folder = "raw/"  # Mappen i S3 der filene skal ligge

# Midlertidig mappe for nedlasting før opplasting
TEMP_DIR = Path("temp_downloads")

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def clean_filename(title):
    # Rens filnavn for ulovlige tegn og mellomrom
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
    return cleaned.strip()

def download_and_upload():
    if not RSS_URL or not S3_BUCKET:
        print("FEIL: Mangler RSS_URL eller S3_BUCKET i .env filen.")
        return

    s3 = get_s3_client()
    TEMP_DIR.mkdir(exist_ok=True)

    print(f"Henter podcast-liste fra: {RSS_URL} ...")
    feed = feedparser.parse(RSS_URL)

    if not feed.entries:
        print("Fant ingen episoder.")
        return

    print(f"Fant {len(feed.entries)} episoder. Sjekker mot S3...")

    for entry in feed.entries:
        title = clean_filename(entry.title)
        
        # Finn mp3-lenke
        mp3_url = None
        for link in entry.links:
            if link.rel == 'enclosure' and 'audio' in link.type:
                mp3_url = link.href
                break
        
        if not mp3_url:
            continue

        filename = f"{title}.mp3"
        s3_key = f"{S3_Target_Folder}{filename}"
        local_path = TEMP_DIR / filename

        # 1. SJEKK OM FILEN ALLEREDE FINNES I S3
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            print(f"SKIP (Finnes i S3): {title}")
            continue
        except:
            # Filen finnes ikke i S3, så vi fortsetter
            pass

        # 2. LAST NED LOKALT
        print(f"LASTER NED: {title} ...")
        try:
            response = requests.get(mp3_url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 3. LAST OPP TIL S3
            print(f" -> Laster opp til s3://{S3_BUCKET}/{s3_key} ...")
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)
            print(" -> Ferdig!")

        except Exception as e:
            print(f" -> FEILET: {e}")
        
        finally:
            # 4. RYDD OPP (Slett lokal fil)
            if local_path.exists():
                local_path.unlink()

    # Fjern temp-mappen til slutt
    if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
        TEMP_DIR.rmdir()

    print("\nAlle nye episoder er lastet opp til S3!")

if __name__ == "__main__":
    download_and_upload()
