import os
import json
import subprocess
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("S3_BUCKET", "ml-data")

# ==========================================
# ⚙️ KONFIGURASJON
# ==========================================
MIN_DURATION = 2.0  
MAX_DURATION = 15.0 
BASE_PATH = "002_speech_dataset"
OUT_BASE = "003_final_dataset/parquet"

def get_processed_files(s3):
    """Henter en liste over alle episoder som er ferdig prosessert av Jobb 3/4."""
    print("🔍 Leter etter prosesserte episoder i S3...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed_global/")
    return sorted([obj['Key'] for page in pages if 'Contents' in page for obj in page['Contents'] if obj['Key'].endswith(".jsonl")])

def get_exported_parquets(s3):
    """Sjekker hvilke Parquet-filer vi allerede har bygget (Smart Resume)."""
    print("🔍 Sjekker hvilke episoder som allerede er eksportert...")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{OUT_BASE}/")
        # Vi forventer at filnavnene er: {podcast_navn}___{episode_navn}.parquet
        return {obj['Key'].split('/')[-1] for page in pages if 'Contents' in page for obj in page['Contents'] if obj['Key'].endswith(".parquet")}
    except Exception:
        return set()

def main():
    print("🚀 Starter PARQUET EXPORT for Multi-Speaker Datasett (Kun Speaker IDs)!")
    
    try:
        s3 = boto3.client('s3', endpoint_url=S3_ENDPOINT, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    except Exception as e:
        print(f"❌ Klarte ikke koble til S3: {e}")
        return
        
    all_episodes = get_processed_files(s3)
    exported_files = get_exported_parquets(s3)
    
    print(f"📦 Fant {len(all_episodes)} episoder. {len(exported_files)} er allerede eksportert.")

    Path("temp_audio").mkdir(exist_ok=True)
    
    total_new_clips = 0
    total_seconds = 0.0
    
    for ep_index, ep_key in enumerate(all_episodes, 1):
        parts = ep_key.split('/')
        podcast_name = parts[2] if len(parts) >= 4 else "Ukjent"
        ep_name_base = parts[-1].replace('.jsonl', '')
        
        # Sjekk Smart Resume
        parquet_filename = f"{podcast_name}___{ep_name_base}.parquet"
        
        if parquet_filename in exported_files:
            continue
            
        print(f"⏳ Prosesserer [{ep_index}/{len(all_episodes)}]: {ep_name_base}...")
        
        # 1. Les JSONL
        try:
            response = s3.get_object(Bucket=BUCKET, Key=ep_key)
            lines = response['Body'].read().decode('utf-8').splitlines()
        except Exception as e:
            print(f"❌ Feil ved lesing av {ep_key}: {e}")
            continue
        
        segments = []
        for line in lines:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                spk_id = data.get("global_speaker_id")
                text = data.get("text", "").strip()
                start = data.get("start", 0.0)
                end = data.get("end", 0.0)
                dur = end - start
                
                # Bruk KUN raw spk_id
                if spk_id and MIN_DURATION <= dur <= MAX_DURATION and len(text) > 2:
                    segments.append({
                        "speaker": spk_id,
                        "text": text,
                        "start": start,
                        "end": end,
                        "dur": dur
                    })
            except Exception:
                continue
                
        if not segments:
            # Lag en tom parquet-fil så vi vet at vi har sjekket denne episoden
            try:
                empty_table = pa.Table.from_arrays([pa.array([], type=pa.binary()), pa.array([], type=pa.string()), pa.array([], type=pa.string())], names=['audio', 'speaker', 'text'])
                local_empty = f"temp_audio/{parquet_filename}"
                pq.write_table(empty_table, local_empty)
                s3.upload_file(local_empty, BUCKET, f"{OUT_BASE}/{parquet_filename}")
                os.remove(local_empty)
            except Exception as e:
                print(f"⚠️ Kunne ikke lage tom fil for {parquet_filename}: {e}")
            continue

        # 2. Last ned original-lyden
        s3_audio_key = f"{BASE_PATH}/audio/{podcast_name}/{ep_name_base}.mp3"
        local_mp3 = "temp_audio/temp_ep.mp3"
        try:
            s3.download_file(BUCKET, s3_audio_key, local_mp3)
        except Exception:
            print(f"❌ Fant ikke lyd for {ep_name_base}. Hopper over.")
            continue

        # 3. Klipp lyden til minnet
        audio_data = []
        speaker_data = []
        text_data = []
        
        for i, seg in enumerate(segments):
            local_wav = f"temp_audio/clip_{i}.wav"
            
            cmd = [
                "ffmpeg", "-y", "-i", local_mp3,
                "-ss", str(seg["start"]), "-to", str(seg["end"]),
                "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
                local_wav
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(local_wav):
                with open(local_wav, "rb") as f:
                    wav_bytes = f.read()
                    
                # Hugging Face Audio Feature format: en dict med "bytes" og valgfri "path"
                audio_data.append({"bytes": wav_bytes, "path": f"clip_{i}.wav"})
                speaker_data.append(seg["speaker"])
                text_data.append(seg["text"])
                
                total_seconds += seg["dur"]
                total_new_clips += 1
                os.remove(local_wav)
            
        if os.path.exists(local_mp3):
            os.remove(local_mp3)

        # 4. Bygg og last opp Parquet
        if audio_data:
            try:
                # Definer skjemaet eksplisitt for å unngå Arrow-feil
                schema = pa.schema([
                    pa.field('audio', pa.struct([
                        pa.field('bytes', pa.binary()),
                        pa.field('path', pa.string())
                    ])),
                    pa.field('speaker', pa.string()),
                    pa.field('text', pa.string())
                ])
                
                table = pa.Table.from_arrays(
                    [pa.array(audio_data), pa.array(speaker_data), pa.array(text_data)],
                    schema=schema
                )
                
                local_parquet = f"temp_audio/{parquet_filename}"
                pq.write_table(table, local_parquet)
                
                s3.upload_file(local_parquet, BUCKET, f"{OUT_BASE}/{parquet_filename}")
                os.remove(local_parquet)
                print(f"✅ Lagret {parquet_filename} med {len(audio_data)} klipp.")
            except Exception as e:
                print(f"❌ Feil ved lagring av parquet {parquet_filename}: {e}")

    print("\n" + "="*60)
    print("🎉 PARQUET EXPORT FERDIG!")
    print(f"📂 Nye lydklipp eksportert : {total_new_clips}")
    print(f"⏱️ Ny eksportert taletid   : {total_seconds / 3600:.2f} timer")
    print(f"📍 Alt ligger klart i S3   : s3://{BUCKET}/{OUT_BASE}/")
    print("="*60)

if __name__ == "__main__":
    main()
