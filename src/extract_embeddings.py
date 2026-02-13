import os
import json
import boto3
import pickle
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
from pydub import AudioSegment

def main():
    load_dotenv()
    BUCKET = os.getenv("S3_BUCKET", "ml-data")
    
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    print("🤖 Loading pyannote/embedding model...")
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HF_TOKEN"))
    inference = Inference(model, window="whole")

    # --- GJENOPPTA FREMDRIFT (RESUME) ---
    all_embeddings = {}
    pickle_file = "embeddings.pkl"
    
    if os.path.exists(pickle_file):
        print(f"📥 Found existing '{pickle_file}'. Loading previous progress...")
        with open(pickle_file, "rb") as f:
            all_embeddings = pickle.load(f)
        print(f"   -> Loaded {len(all_embeddings)} fingerprints.")
    else:
        print("🆕 Starting fresh. No previous fingerprints found.")

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix="002_speech_dataset/processed/")

    print("🔍 Scanning episodes...")
    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if not obj['Key'].endswith('.jsonl'): continue
            
            jsonl_key = obj['Key']
            audio_key = jsonl_key.replace("processed/", "raw/").replace(".jsonl", ".mp3")
            ep_name = jsonl_key.split("/")[-1].replace(".jsonl", "")

            # Sjekk om vi allerede har behandlet denne episoden
            # (Hvis noen av nøklene i ordboken starter med episodens navn)
            already_processed = any(key.startswith(f"{ep_name}_") for key in all_embeddings.keys())
            if already_processed:
                print(f"⏩ Skipping: {ep_name} (Already processed)")
                continue

            print(f"\n🎧 Processing: {ep_name}")
            
            # 1. Read JSONL and find the longest clip per speaker
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

            # 2. Download audio temporarily
            temp_audio = "temp_audio.mp3"
            try:
                s3.download_file(BUCKET, audio_key, temp_audio)
            except Exception as e:
                print(f"⚠️ Could not find audio for {audio_key}. Skipping.")
                continue

            audio = AudioSegment.from_file(temp_audio)

            # 3. Create embeddings
            for spk, times in speaker_clips.items():
                start_ms = int(times['start'] * 1000)
                end_ms = int(times['end'] * 1000)
                clip = audio[start_ms:end_ms]
                
                temp_clip = "temp_clip.wav"
                clip.export(temp_clip, format="wav")
                
                unique_id = f"{ep_name}_{spk}"
                embedding = inference(temp_clip)
                all_embeddings[unique_id] = embedding
                print(f"  👉 Extracted {spk} ({times['duration']:.1f} sec)")

            # Cleanup files
            os.remove(temp_audio)
            if os.path.exists("temp_clip.wav"): os.remove("temp_clip.wav")

            # --- LAGRE ETTER HVER EPISODE ---
            with open(pickle_file, "wb") as f:
                pickle.dump(all_embeddings, f)

    print("\n✅ Extraction complete! Saved to 'embeddings.pkl'.")

if __name__ == "__main__":
    main()
