import os
import json
import boto3
import torch
import numpy as np
import pickle
import time
from io import BytesIO
from collections import defaultdict
from dotenv import load_dotenv
from pydub import AudioSegment

# --- MONKEY PATCH START (Må være øverst) ---
# Dette fikser "TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'"
import huggingface_hub

# Vi tar vare på den originale funksjonen
_original_hf_hub_download = huggingface_hub.hf_hub_download

def _patched_hf_hub_download(*args, **kwargs):
    # Hvis gamle biblioteker sender 'use_auth_token', bytter vi navn til 'token'
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_hub_download(*args, **kwargs)

# Vi erstatter funksjonen i biblioteket med vår fikset versjon
huggingface_hub.hf_hub_download = _patched_hf_hub_download
# --- MONKEY PATCH SLUTT ---

from pyannote.audio import Model, Inference

def main():
    load_dotenv()
    
    # --- KONFIGURASJON ---
    BUCKET = os.getenv("S3_BUCKET", "ml-data")
    BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    NUM_SAMPLES = 5       # Hvor mange klipp skal vi basere "snittet" på?
    BATCH_SIZE = 10       # Hvor mange episoder behandler vi før vi lagrer til S3?
    # ---------------------

    s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    print("🚀 Laster Pyannote embedding-modell...")
    # Vi bruker 'token' her (det er riktig i nyere versjoner), 
    # og monkey-patchen over fikser de interne kallene som bruker feil navn.
    model = Model.from_pretrained("pyannote/embedding", token=HF_TOKEN)
    inference = Inference(model, window="whole")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Modell lastet på {device} | Batch-størrelse: {BATCH_SIZE}")

    embeddings_map = {}
    
    # Prøv å laste eksisterende fil (Resume-funksjon)
    try:
        print("🔍 Ser etter eksisterende embeddings...")
        resp = s3.get_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/embeddings.pkl")
        embeddings_map = pickle.loads(resp['Body'].read())
        print(f"📥 Resume: Lastet {len(embeddings_map)} eksisterende embeddings.")
    except:
        print("✨ Starter med blanke ark (Ingen embeddings funnet i metadata-mappen).")

    # Hent liste over alle ferdige episoder
    processed_files = []
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed/")
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.jsonl'):
                    processed_files.append(obj['Key'])

    print(f"📂 Fant {len(processed_files)} episoder totalt.")

    episodes_since_save = 0
    start_time = time.time()

    for jsonl_key in processed_files:
        ep_name = jsonl_key.split("/")[-1].replace(".jsonl", "")
        
        # Sjekk om episoden allerede er i minnet (Resume-sjekk)
        if any(k.startswith(ep_name) for k in embeddings_map.keys()):
            continue

        print(f"\n🎧 Behandler: {ep_name}")
        
        try:
            # Last tekst (JSONL)
            jsonl_obj = s3.get_object(Bucket=BUCKET, Key=jsonl_key)
            transcript_lines = jsonl_obj['Body'].read().decode('utf-8').splitlines()
            
            # Last lyd (MP3)
            audio_key = jsonl_key.replace("processed/", "raw/").replace(".jsonl", ".mp3")
            audio_obj = s3.get_object(Bucket=BUCKET, Key=audio_key)
            # Laster hele filen i minnet (raskere for slicing med pydub)
            audio = AudioSegment.from_file(BytesIO(audio_obj['Body'].read()))
        except Exception as e:
            print(f"   ❌ Feil ved nedlasting av {ep_name}: {e}")
            continue

        # Grupper klipp per speaker
        speaker_segments = defaultdict(list)
        for line in transcript_lines:
            data = json.loads(line)
            duration = data['end'] - data['start']
            # Vi ignorerer klipp under 2 sekunder for embedding (for dårlig kvalitet/for lite data)
            if duration > 2.0: 
                speaker_segments[data['speaker']].append((data['start'], data['end'], duration))

        # Prosesser hver speaker i episoden
        for speaker, segments in speaker_segments.items():
            # Sorter etter lengde (lengst først -> best kvalitet)
            segments.sort(key=lambda x: x[2], reverse=True)
            
            # Ta de N beste klippene
            top_segments = segments[:NUM_SAMPLES]
            
            vectors = []
            total_duration_used = 0
            
            for start, end, dur in top_segments:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                
                # Klipp ut lyden
                chunk = audio[start_ms:end_ms]
                # Pyannote forventer 16kHz mono
                chunk = chunk.set_frame_rate(16000).set_channels(1)
                
                # Lagre til midlertidig fil i RAM-disk (/tmp)
                chunk.export("/tmp/clip.wav", format="wav")
                
                # Kjør AI-modellen
                try:
                    with torch.no_grad():
                        embedding = inference("/tmp/clip.wav")
                        vectors.append(embedding)
                        total_duration_used += dur
                except Exception as e:
                    print(f"      ⚠️ Feil ved embedding av klipp: {e}")

            if vectors:
                # ✨ MAGIEN: Regn ut gjennomsnittet av alle vektorene (Centroid)
                # Dette lager et mye mer stabilt "bilde" av stemmen enn bare ett klipp.
                mean_vector = np.mean(vectors, axis=0)
                
                # Lagre resultatet med unik ID
                unique_id = f"{ep_name}_{speaker}"
                embeddings_map[unique_id] = mean_vector
                print(f"   👉 {speaker}: Snitt av {len(vectors)} klipp ({total_duration_used:.1f}s)")

        episodes_since_save += 1

        # --- BATCH LAGRING ---
        if episodes_since_save >= BATCH_SIZE:
            print(f"💾 Lagrer checkpoint til S3 ({len(embeddings_map)} embeddings totalt)...")
            try:
                s3.put_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/embeddings.pkl", Body=pickle.dumps(embeddings_map))
                episodes_since_save = 0
                
                elapsed = time.time() - start_time
                print(f"⏱️  Tid brukt så langt: {int(elapsed//60)} minutter.")
            except Exception as e:
                print(f"❌ Kunne ikke lagre til S3: {e}")

    # Lagre helt til slutt hvis det er usavede endringer
    if episodes_since_save > 0:
        print("💾 Lagrer siste rest til S3...")
        s3.put_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/embeddings.pkl", Body=pickle.dumps(embeddings_map))

    print("\n✅ JOBB A FERDIG! Embeddings er lagret og klare for Jobb B.")

if __name__ == "__main__":
    main()
