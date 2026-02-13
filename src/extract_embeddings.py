import os
import json
import boto3
import pickle
import botocore

# ==========================================
# MAGISK FIKS (MONKEY PATCH)
# Fikser pyannote-krasjen UTEN å endre noen versjoner.
# ==========================================
import huggingface_hub
_original_hf_download = huggingface_hub.hf_hub_download

def _patched_hf_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_download(*args, **kwargs)

huggingface_hub.hf_hub_download = _patched_hf_download
# ==========================================

from pyannote.audio import Model, Inference
from pydub import AudioSegment

def main():
    # Hent miljøvariabler
    BUCKET = os.getenv("S3_BUCKET", "ml-data")
    BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
    EMBEDDINGS_KEY = f"{BASE_PATH}/metadata/embeddings.pkl"
    
    print("🔌 Kobler til S3...")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    print("🤖 Laster inn pyannote/embedding modell...")
    
    # Koden under bruker det pyannote forventer, mens patchen vår i toppen rydder opp for Hugging Face!
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HF_TOKEN"))
    inference = Inference(model, window="whole")

    # --- GJENOPPTA FREMDRIFT FRA S3 ---
    all_embeddings = {}
    
    try:
        print(f"📥 Sjekker om {EMBEDDINGS_KEY} finnes i S3...")
        response = s3.get_object(Bucket=BUCKET, Key=EMBEDDINGS_KEY)
        all_embeddings = pickle.loads(response['Body'].read())
        print(f"   -> Fant eksisterende fil! Lastet inn {len(all_embeddings)} fingeravtrykk.")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            print("🆕 Fant ingen tidligere data. Starter med blanke ark.")
        else:
            raise e

    print("🔍 Leter etter prosesserte episoder...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed/")

    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if not obj['Key'].endswith('.jsonl'): continue
            
            jsonl_key = obj['Key']
            audio_key = jsonl_key.replace("processed/", "raw/").replace(".jsonl", ".mp3")
            ep_name = jsonl_key.split("/")[-1].replace(".jsonl", "")

            # Sjekk om episoden allerede er prosessert i minnet
            already_processed = any(key.startswith(f"{ep_name}_") for key in all_embeddings.keys())
            if already_processed:
                print(f"⏩ Hopper over: {ep_name} (Allerede prosessert)")
                continue

            print(f"\n🎧 Behandler: {ep_name}")
            
            # 1. Les JSONL og finn det lengste klippet per speaker
            response = s3.get_object(Bucket=BUCKET, Key=jsonl_key)
            content = response['Body'].read().decode('utf-8')
            
            speaker_clips = {}
            for line in content.splitlines():
                if line.strip():
                    data = json.loads(line)
                    spk = data['speaker']
                    duration = data['end'] - data['start']
                    
                    if spk not in speaker_clips or duration > speaker_clips[spk]['duration']:
                        speaker_clips[spk] = {'start': data['start'], 'end': data['end'], 'duration': duration}

            if not speaker_clips: continue

            # 2. Last ned lydfil midlertidig
            temp_audio = "temp_audio.mp3"
            try:
                s3.download_file(BUCKET, audio_key, temp_audio)
            except Exception as e:
                print(f"⚠️ Fant ikke lydfil for {audio_key}. Hopper over.")
                continue

            audio = AudioSegment.from_file(temp_audio)

            # 3. Lag fingeravtrykk
            for spk, times in speaker_clips.items():
                start_ms = int(times['start'] * 1000)
                end_ms = int(times['end'] * 1000)
                clip = audio[start_ms:end_ms]
                
                temp_clip = "temp_clip.wav"
                clip.export(temp_clip, format="wav")
                
                unique_id = f"{ep_name}_{spk}"
                embedding = inference(temp_clip)
                all_embeddings[unique_id] = embedding
                print(f"  👉 Hentet ut avtrykk for {spk} ({times['duration']:.1f} sek)")

            # Rydd opp midlertidige filer lokalt
            if os.path.exists(temp_audio): os.remove(temp_audio)
            if os.path.exists("temp_clip.wav"): os.remove("temp_clip.wav")

            # --- LAGRE PROGRESS TIL S3 ---
            s3.put_object(
                Bucket=BUCKET, 
                Key=EMBEDDINGS_KEY, 
                Body=pickle.dumps(all_embeddings)
            )

    print(f"\n✅ Jobb A er ferdig! Fingeravtrykkene ligger trygt i S3: {EMBEDDINGS_KEY}")

if __name__ == "__main__":
    main()
