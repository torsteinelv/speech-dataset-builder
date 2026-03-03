import os
import json
import subprocess
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import io
import wave
import hashlib
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

BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
OUT_BASE = f"{BASE_PATH}/parquet"
DONE_BASE = f"{BASE_PATH}/parquet_done"

SR = 24000
SAMPWIDTH = 2  # int16
NCH = 1

# 🟢 FIKSET: Kolonnen heter nå "speaker_id" i stedet for "speaker"
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("audio", pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])),
    pa.field("speaker_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("start", pa.float32()),
    pa.field("end", pa.float32()),
    pa.field("dur", pa.float32()),
    pa.field("source", pa.string()),
    pa.field("episode_key", pa.string()),
    pa.field("audio_key", pa.string()),
])

def wav_bytes_from_int16(mono_int16: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(NCH)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(SR)
        wf.writeframes(mono_int16.tobytes())
    return buf.getvalue()

def get_processed_files(s3):
    print(f"🔍 Leter etter prosesserte episoder i {BASE_PATH}/processed_global/...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed_global/")
    return sorted([obj['Key'] for page in pages if 'Contents' in page for obj in page['Contents'] if obj['Key'].endswith(".jsonl")])

def get_exported_markers(s3):
    print(f"🔍 Sjekker hvilke episoder som allerede er ferdige i {DONE_BASE}/...")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{DONE_BASE}/")
        return {obj['Key'].split('/')[-1].replace('.done', '') for page in pages if 'Contents' in page for obj in page['Contents'] if obj['Key'].endswith(".done")}
    except Exception:
        return set()

def main():
    print("🚀 Starter LYNKJAPP PARQUET EXPORT for Multi-Speaker Datasett (med speaker_id)!")
    
    try:
        s3 = boto3.client('s3', endpoint_url=S3_ENDPOINT, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    except Exception as e:
        print(f"❌ Klarte ikke koble til S3: {e}")
        return
        
    all_episodes = get_processed_files(s3)
    exported_markers = get_exported_markers(s3)
    
    print(f"📦 Fant {len(all_episodes)} episoder. {len(exported_markers)} er allerede eksportert.")

    Path("temp_audio").mkdir(exist_ok=True)
    
    total_new_clips = 0
    total_seconds = 0.0
    
    for ep_index, ep_key in enumerate(all_episodes, 1):
        rel_path = ep_key.split(f"{BASE_PATH}/processed_global/", 1)[-1]
        
        rel_no_jsonl = rel_path[:-6] if rel_path.endswith(".jsonl") else rel_path
        
        if not rel_no_jsonl.endswith(".mp3"):
            s3_audio_key = f"{BASE_PATH}/raw/{rel_no_jsonl}.mp3"
        else:
            s3_audio_key = f"{BASE_PATH}/raw/{rel_no_jsonl}"
            
        safe_base_name = rel_no_jsonl.replace(".mp3", "").replace("/", "___")
        source_name = rel_path.split("/")[0] if "/" in rel_path else "Ukjent"
        
        ep_hash = hashlib.sha1(ep_key.encode()).hexdigest()[:10]
        
        if safe_base_name in exported_markers:
            continue
            
        print(f"⏳ Prosesserer [{ep_index}/{len(all_episodes)}]: {safe_base_name}...")
        
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
                
                if spk_id and MIN_DURATION <= dur <= MAX_DURATION and len(text) > 2:
                    segments.append({
                        "speaker_id": spk_id, # Bruker konsekvent speaker_id lokalt også
                        "text": text,
                        "start": start,
                        "end": end,
                        "dur": dur
                    })
            except Exception:
                continue
                
        if not segments:
            s3.put_object(Bucket=BUCKET, Key=f"{DONE_BASE}/{safe_base_name}.done", Body=b"done")
            continue

        local_mp3 = f"temp_audio/{ep_hash}_original.mp3"
        try:
            s3.download_file(BUCKET, s3_audio_key, local_mp3)
        except Exception as e:
            print(f"❌ Fant ikke lyd for {safe_base_name} ({s3_audio_key}). Hopper over.")
            continue

        local_pcm = f"temp_audio/{ep_hash}.s16"
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", local_mp3,
            "-vn", "-map", "0:a:0",
            "-ar", str(SR), "-ac", str(NCH),
            "-f", "s16le", local_pcm
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ FFmpeg feilet på {safe_base_name}. Hopper over.")
            os.remove(local_mp3)
            continue

        try:
            pcm = np.memmap(local_pcm, dtype=np.int16, mode="r")
        except Exception as e:
            print(f"❌ Klarte ikke lese PCM for {safe_base_name}: {e}")
            os.remove(local_mp3)
            if os.path.exists(local_pcm): os.remove(local_pcm)
            continue

        id_data, audio_data, speaker_data, text_data = [], [], [], []
        start_data, end_data, dur_data, source_data, ep_data, audio_key_data = [], [], [], [], [], []
        
        for i, seg in enumerate(segments):
            s = max(0, int(seg["start"] * SR))
            e = max(s + 1, int(seg["end"] * SR))
            
            if s >= len(pcm): continue
            e = min(e, len(pcm))
            
            clip = np.asarray(pcm[s:e])
            wb = wav_bytes_from_int16(clip)
            
            clip_path = f"{safe_base_name}___{i:06d}.wav"
            
            clip_id = hashlib.sha1(
                f"{ep_key}:{seg['start']:.3f}:{seg['end']:.3f}:{seg['speaker_id']}:{len(seg['text'])}".encode()
            ).hexdigest()[:16]
            
            id_data.append(clip_id)
            audio_data.append({"bytes": wb, "path": clip_path})
            speaker_data.append(seg["speaker_id"])
            text_data.append(seg["text"])
            start_data.append(seg["start"])
            end_data.append(seg["end"])
            dur_data.append(seg["dur"])
            source_data.append(source_name)
            ep_data.append(ep_key)
            audio_key_data.append(s3_audio_key)
            
            total_seconds += seg["dur"]
            total_new_clips += 1

        del pcm
        os.remove(local_mp3)
        os.remove(local_pcm)

        if audio_data:
            try:
                table = pa.Table.from_arrays(
                    [
                        pa.array(id_data, type=pa.string()),
                        pa.array(audio_data, type=SCHEMA.field('audio').type),
                        pa.array(speaker_data, type=pa.string()), # Her putter vi inn speaker_id-dataene
                        pa.array(text_data, type=pa.string()),
                        pa.array(start_data, type=pa.float32()),
                        pa.array(end_data, type=pa.float32()),
                        pa.array(dur_data, type=pa.float32()),
                        pa.array(source_data, type=pa.string()),
                        pa.array(ep_data, type=pa.string()),
                        pa.array(audio_key_data, type=pa.string()),
                    ],
                    schema=SCHEMA
                )
                
                local_parquet = f"temp_audio/{ep_hash}.parquet"
                
                pq.write_table(
                    table, 
                    local_parquet, 
                    compression="zstd", 
                    compression_level=3,
                    use_dictionary=["speaker_id", "source"] # Oppdatert dictionary referanse
                )
                
                parquet_s3_key = f"{OUT_BASE}/{safe_base_name}.parquet"
                s3.upload_file(local_parquet, BUCKET, parquet_s3_key)
                
                s3.put_object(Bucket=BUCKET, Key=f"{DONE_BASE}/{safe_base_name}.done", Body=b"done")
                
                os.remove(local_parquet)
                print(f"✅ Lagret {safe_base_name}.parquet med {len(audio_data)} klipp.")
            except Exception as e:
                print(f"❌ Feil ved lagring av parquet for {safe_base_name}: {e}")

    print("\n" + "="*60)
    print("🎉 PARQUET EXPORT FERDIG!")
    print(f"📂 Nye lydklipp eksportert : {total_new_clips}")
    print(f"⏱️ Ny eksportert taletid   : {total_seconds / 3600:.2f} timer")
    print("="*60)

if __name__ == "__main__":
    main()
