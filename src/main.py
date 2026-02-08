import os
import gc
import json
import torch
import whisperx
import pandas as pd
import subprocess
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()

# --- KONFIGURASJON ---
# S3 / Ceph Config
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL") # VIKTIG FOR CEPH!
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_INPUT_PREFIX = "raw/"       # Mappen i bøtta der MP3-filene ligger
S3_OUTPUT_PREFIX = "dataset/"  # Mappen i bøtta der ferdig JSONL havner

# Modell Config
NBAILAB_MODEL_ID = "NbAiLab/nb-whisper-large"
DEVICE = "cuda"
BATCH_SIZE = 32         
COMPUTE_TYPE = "float16" 
HF_TOKEN = os.getenv("HF_TOKEN")

# Filter Config
MIN_CONFIDENCE = 0.90   
BUFFER_ZONE = 0.5       
MIN_DURATION = 1.5      
NUM_SPEAKERS = 3        

# Lokale stier (Midlertidig lagring)
LOCAL_TEMP_DIR = Path("/tmp/processing")
MODEL_CACHE_DIR = Path("/app/models")

def get_s3_client():
    """Oppretter tilkobling til Ceph S3"""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT, # Her er magien for Ceph
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def get_norwegian_model_path():
    """Sjekker/Konverterer NbAiLab-modellen til WhisperX format"""
    model_name_safe = NBAILAB_MODEL_ID.replace("/", "_")
    ct2_output_dir = MODEL_CACHE_DIR / f"{model_name_safe}_ct2"

    if ct2_output_dir.exists():
        print(f"--- Fant bufret modell: {ct2_output_dir} ---")
        return str(ct2_output_dir)
    
    print(f"--- Laster ned og konverterer {NBAILAB_MODEL_ID}... ---")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ct2-transformers-converter",
        "--model", NBAILAB_MODEL_ID,
        "--output_dir", str(ct2_output_dir),
        "--quantization", COMPUTE_TYPE,
        "--low_cpu_mem_usage"
    ]
    try:
        subprocess.run(cmd, check=True)
        return str(ct2_output_dir)
    except Exception as e:
        print(f"Modell-konvertering feilet ({e}). Bruker standard 'large-v3'.")
        return "large-v3"

def process_audio():
    if not HF_TOKEN: raise ValueError("Mangler HF_TOKEN!")
    if not S3_BUCKET: raise ValueError("Mangler S3_BUCKET!")

    # 1. Klargjør
    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    
    # 2. Last modell
    model_path = get_norwegian_model_path()
    print(f"--- Laster WhisperX med modell: {model_path} ---")
    model = whisperx.load_model(model_path, DEVICE, compute_type=COMPUTE_TYPE, language="no")
    
    # 3. Finn filer i Ceph
    print(f"Lister filer i s3://{S3_BUCKET}/{S3_INPUT_PREFIX}...")
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_INPUT_PREFIX)
    except Exception as e:
        print(f"Klarte ikke koble til S3: {e}")
        return

    if 'Contents' not in response:
        print("Ingen filer funnet i raw-mappen.")
        return

    all_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.mp3')]
    print(f"Fant {len(all_files)} mp3-filer som skal prosesseres.")

    # 4. Prosesser hver fil
    for i, s3_key in enumerate(all_files):
        filename = Path(s3_key).name
        result_key = f"{S3_OUTPUT_PREFIX}{filename.replace('.mp3', '.jsonl')}"
        
        # Sjekk om ferdig (Resume capability)
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=result_key)
            print(f"Skipper {filename} (allerede ferdig).")
            continue
        except:
            pass 

        print(f"\n[{i+1}/{len(all_files)}] Laster ned {filename}...")
        local_mp3 = LOCAL_TEMP_DIR / filename
        local_json = LOCAL_TEMP_DIR / f"{filename}.jsonl"

        try:
            # A. Last ned
            s3.download_file(S3_BUCKET, s3_key, str(local_mp3))

            # B. Transkriber
            audio = whisperx.load_audio(str(local_mp3))
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language="no")

            # C. Align
            model_a, metadata = whisperx.load_align_model(language_code="no", device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            del model_a, metadata
            gc.collect(); torch.cuda.empty_cache()

            # D. Diarize
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
            diarize_segments = diarize_model(audio, min_speakers=NUM_SPEAKERS, max_speakers=NUM_SPEAKERS)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            gc.collect(); torch.cuda.empty_cache()

            # E. Filtrer og Lagre
            clean_data = filter_data(result, diarize_segments, filename)
            
            with open(local_json, "w", encoding="utf-8") as f:
                for entry in clean_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")

            # F. Last opp til Ceph
            print(f"Laster opp resultat til {result_key}...")
            s3.upload_file(str(local_json), S3_BUCKET, result_key)

        except Exception as e:
            print(f"FEIL på {filename}: {e}")
        
        finally:
            # Rydd opp disk
            if local_mp3.exists(): local_mp3.unlink()
            if local_json.exists(): local_json.unlink()
            gc.collect(); torch.cuda.empty_cache()

def filter_data(result, diarize_segments, filename):
    """Vasker dataene for støy og overlapp"""
    bad_zones = []
    df = pd.DataFrame(diarize_segments)
    if not df.empty:
        df = df.sort_values(by="start")
        for idx in range(len(df) - 1):
            curr = df.iloc[idx]
            nxt = df.iloc[idx+1]
            if curr["end"] > nxt["start"]:
                zone_start = max(curr["start"], nxt["start"]) - BUFFER_ZONE
                zone_end = min(curr["end"], nxt["end"]) + BUFFER_ZONE
                bad_zones.append((zone_start, zone_end))

    clean_segments = []
    for seg in result["segments"]:
        if "speaker" not in seg: continue
        if (seg["end"] - seg["start"]) < MIN_DURATION: continue

        # Sjekk overlapp
        is_bad = False
        for b_start, b_end in bad_zones:
            if (seg["start"] < b_end) and (seg["end"] > b_start):
                is_bad = True
                break
        if is_bad: continue

        # Sjekk confidence
        words = seg.get("words", [])
        if not words: continue
        avg_score = sum(w.get("score", 0) for w in words) / len(words)
        if avg_score < MIN_CONFIDENCE: continue

        clean_segments.append({
            "audio_file": filename,
            "speaker": seg["speaker"],
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
            "score": round(avg_score, 3)
        })
    return clean_segments

if __name__ == "__main__":
    process_audio()
