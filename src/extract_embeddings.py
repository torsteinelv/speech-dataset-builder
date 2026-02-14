import os
import json
import boto3
import torch
import numpy as np
import pickle
from pyannote.audio import Model, Inference
from pydub import AudioSegment
from io import BytesIO
from collections import defaultdict

def main():
    # Oppsett
    BUCKET = os.getenv("S3_BUCKET", "ml-data")
    BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Hvor mange klipp skal vi basere gjennomsnittet på?
    NUM_SAMPLES = 5 

    s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    print("🚀 Laster Pyannote embedding-modell...")
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
    inference = Inference(model, window="whole")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Modell lastet på {device}")

    # Last eksisterende embeddings hvis vi skal fortsette en jobb
    embeddings_map = {}
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=f"{BASE_PATH}/embeddings.pkl")
        embeddings_map = pickle.loads(resp['Body'].read())
        print(f"📥 Lastet {len(embeddings_map)} eksisterende embeddings.")
    except:
        print("✨ Starter ny embeddings-database.")

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed/")

    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if not obj['Key'].endswith('.jsonl'): continue
            
            jsonl_key = obj['Key']
            ep_name = jsonl_key.split("/")[-1].replace(".jsonl", "")
            
            # Sjekk om vi allerede har behandlet denne episoden
            # Vi sjekker om noen nøkler i kartet starter med ep_name
            if any(k.startswith(ep_name) for k in embeddings_map.keys()):
                print(f"⏩ Skipper {ep_name} (allerede ferdig)")
                continue

            print(f"🎧 Behandler: {ep_name}")
            
            # Last data
            jsonl_obj = s3.get_object(Bucket=BUCKET, Key=jsonl_key)
            transcript_lines = jsonl_obj['Body'].read().decode('utf-8').splitlines()
            
            audio_key = jsonl_key.replace("processed/", "raw/").replace(".jsonl", ".mp3")
            try:
                audio_obj = s3.get_object(Bucket=BUCKET, Key=audio_key)
                audio = AudioSegment.from_file(BytesIO(audio_obj['Body'].read()))
            except:
                print(f"⚠️ Fant ikke lydfil for {ep_name}")
                continue

            # Samle alle klipp per speaker
            speaker_segments = defaultdict(list)
            for line in transcript_lines:
                data = json.loads(line)
                duration = data['end'] - data['start']
                if duration > 1.5: # Ignorer veldig korte lyder
                    speaker_segments[data['speaker']].append((data['start'], data['end'], duration))

            # Prosesser hver speaker
            for speaker, segments in speaker_segments.items():
                # Sorter etter lengde (lengst først) og ta de N beste
                segments.sort(key=lambda x: x[2], reverse=True)
                top_segments = segments[:NUM_SAMPLES]
                
                vectors = []
                total_duration = 0
                
                for start, end, dur in top_segments:
                    start_ms = int(start * 1000)
                    end_ms = int(end * 1000)
                    
                    # Trekk ut lyden og konverter til formatet modellen liker
                    chunk = audio[start_ms:end_ms].set_frame_rate(16000).set_channels(1)
                    
                    # Lagre midlertidig wav for inference (Pyannote krever filsti eller tensor)
                    chunk.export("/tmp/clip.wav", format="wav")
                    
                    # Generer embedding
                    with torch.no_grad():
                        embedding = inference("/tmp/clip.wav")
                        vectors.append(embedding)
                        total_duration += dur

                if vectors:
                    # ✨ MAGIEN: Regn ut gjennomsnittet av alle vektorene
                    mean_vector = np.mean(vectors, axis=0)
                    
                    # Lagre i kartet
                    unique_id = f"{ep_name}_{speaker}"
                    embeddings_map[unique_id] = mean_vector
                    print(f"   👉 {speaker}: Snitt av {len(vectors)} klipp ({total_duration:.1f} sek)")

            # Lagre checkpoint til S3 etter hver episode
            s3.put_object(Bucket=BUCKET, Key=f"{BASE_PATH}/embeddings.pkl", Body=pickle.dumps(embeddings_map))

if __name__ == "__main__":
    main()
