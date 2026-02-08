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

load_dotenv()

# --- S3 KONFIGURASJON ---
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# --- PROSJEKT STIER ---
base_path = os.getenv("S3_BASE_PATH", "002_speech_dataset")
S3_INPUT_PREFIX = f"{base_path}/raw/"
S3_OUTPUT_PREFIX = f"{base_path}/processed/"

# --- MODELL & HARDWARE ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "NbAiLab/nb-whisper-large") 
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- FILTER ---
MIN_CONFIDENCE = 0.90   
BUFFER_ZONE = 0.5       
MIN_DURATION = 1.5      
NUM_SPEAKERS = 3        

LOCAL_TEMP_DIR = Path("temp_processing")
MODEL_CACHE_DIR = Path("models_cache")

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

def get_model_path():
    if "/" not in WHISPER_MODEL:
        return WHISPER_MODEL

    model_name_safe = WHISPER_MODEL.replace("/", "_")
    ct2_output_dir = MODEL_CACHE_DIR / f"{model_name_safe}_ct2"

    if ct2_output_dir.exists():
        print(f"--- Fant bufret modell: {ct2_output_dir} ---")
        return str(ct2_output_dir)
    
    print(f"--- Konverterer {WHISPER_MODEL}... ---")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ct2-transformers-converter",
        "--model", WHISPER_MODEL,
        "--output_dir", str(ct2_output_dir),
        "--quantization", COMPUTE_TYPE,
        "--low_cpu_mem_usage"
    ]
    try:
        subprocess.run(cmd, check=True)
        return str(ct2_output_dir)
    except Exception as e:
        print(f"Konvertering feilet ({e}). Fallback til 'large-v3'.")
        return "large-v3"

def process_audio():
    if not HF_TOKEN or not S3_BUCKET: 
        raise ValueError("Mangler HF_TOKEN eller S3_BUCKET i .env!")

    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    
    model_path = get_model_path()
    print(f"--- Initialiserer WhisperX ({DEVICE}) ---")
    print(f"--- Prosjekt: {base_path} ---")
    
    model = whisperx.load_model(model_path, DEVICE, compute_type=COMPUTE_TYPE, language="no")
    
    print(f"Lister filer i s3://{S3_BUCKET}/{S3_INPUT_PREFIX}...")
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_INPUT_PREFIX)
    except Exception as e:
        print(f"S3 Feil: {e}")
        return

    if 'Contents' not in response:
        print("Ingen filer funnet.")
        return

    all_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.mp3')]
    print(f"Fant {len(all_files)} mp3-filer.")

    for i, s3_key in enumerate(all_files):
        # Beregn relative stier
        # Input:  002_speech_dataset/raw/Podcast/ep.mp3
        # Output: 002_speech_dataset/processed/Podcast/ep.jsonl
        relative_path = s3_key.replace(S3_INPUT_PREFIX, "", 1)
        result_key = f"{S3_OUTPUT_PREFIX}{relative_path.replace('.mp3', '.jsonl')}"
        filename = Path(s3_key).name

        try:
            s3.head_object(Bucket=S3_BUCKET, Key=result_key)
            print(f"Skipper {relative_path} (Ferdig).")
            continue
        except:
            pass 

        print(f"\n[{i+1}/{len(all_files)}] Laster ned {filename}...")
        local_mp3 = LOCAL_TEMP_DIR / filename
        local_json = LOCAL_TEMP_DIR / f"{filename}.jsonl"

        try:
            s3.download_file(S3_BUCKET, s3_key, str(local_mp3))

            audio = whisperx.load_audio(str(local_mp3))
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language="no")

            model_a, metadata = whisperx.load_align_model(language_code="no", device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            del model_a, metadata
            
            if DEVICE == "cuda":
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
                diarize_segments = diarize_model(audio, min_speakers=NUM_SPEAKERS, max_speakers=NUM_SPEAKERS)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
            else:
                diarize_segments = []

            clean_data = filter_data(result, diarize_segments, relative_path)
            
            with open(local_json, "w", encoding="utf-8") as f:
                for entry in clean_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")

            print(f"Laster opp til {result_key}")
            s3.upload_file(str(local_json), S3_BUCKET, result_key)

        except Exception as e:
            print(f"FEIL på {filename}: {e}")
        
        finally:
            if local_mp3.exists(): local_mp3.unlink()
            if local_json.exists(): local_json.unlink()
            if DEVICE == "cuda":
                gc.collect(); torch.cuda.empty_cache()

def filter_data(result, diarize_segments, source_name):
    if not isinstance(diarize_segments, list) and not hasattr(diarize_segments, 'itertracks'):
         return [{"text": s["text"].strip(), "start": s["start"], "end": s["end"], "speaker": "Unknown", "file": source_name} for s in result["segments"]]

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

        is_bad = False
        for b_start, b_end in bad_zones:
            if (seg["start"] < b_end) and (seg["end"] > b_start):
                is_bad = True
                break
        if is_bad: continue

        words = seg.get("words", [])
        if not words: continue
        avg_score = sum(w.get("score", 0) for w in words) / len(words)
        if avg_score < MIN_CONFIDENCE: continue

        clean_segments.append({
            "audio_file": source_name,
            "speaker": seg["speaker"],
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
            "score": round(avg_score, 3)
        })
    return clean_segments

if __name__ == "__main__":
    process_audio()
