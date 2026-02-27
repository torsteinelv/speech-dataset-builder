import boto3
import json
import os
from collections import defaultdict
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()

ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL") 
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")          
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")      
BUCKET = os.getenv("S3_BUCKET", "ml-data")
PREFIX = "002_speech_dataset/processed_global/"

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}t {m}m {s}s"
    return f"{m}m {s}s"

def main():
    print(f"🚀 Fyrer opp S3-motoren og analyserer {BUCKET}...\n")
    try:
        s3 = boto3.client('s3', endpoint_url=ENDPOINT_URL, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    except Exception as e:
        print(f"❌ Klarte ikke koble til S3: {e}")
        return

    # Ordbøker for å samle all verdens statistikk
    speaker_stats = defaultdict(lambda: defaultdict(lambda: {"episodes": set(), "seconds": 0.0}))
    podcast_stats = defaultdict(lambda: {"seconds": 0.0, "episodes": set(), "speakers": set()})
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)
    
    file_count = 0
    total_global_seconds = 0.0
    
    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith(".jsonl"): continue
            
            file_count += 1
            if file_count % 200 == 0:
                print(f"   ⏳ Tygger data... ({file_count} episoder lest)")
            
            # Hent info fra stien
            parts = key.split('/')
            podcast_navn = parts[2] if len(parts) >= 4 else "Ukjent"
            episode_navn = parts[-1]
            
            podcast_stats[podcast_navn]["episodes"].add(episode_navn)
            
            # Les filen
            response = s3.get_object(Bucket=BUCKET, Key=key)
            lines = response['Body'].read().decode('utf-8').splitlines()

            for line in lines:
                if not line.strip(): continue
                data = json.loads(line)
                
                spk_id = data.get("global_speaker_id")
                if not spk_id: continue
                
                start = data.get("start", 0.0)
                end = data.get("end", 0.0)
                dur = max(0.0, end - start)
                
                if dur > 0:
                    speaker_stats[spk_id][podcast_navn]["episodes"].add(episode_navn)
                    speaker_stats[spk_id][podcast_navn]["seconds"] += dur
                    podcast_stats[podcast_navn]["seconds"] += dur
                    podcast_stats[podcast_navn]["speakers"].add(spk_id)
                    total_global_seconds += dur

    if file_count == 0:
        print("❌ Fant ingen filer! Venter på at Jobb 3 skal gjøre noe...")
        return

    # ==========================================
    # 🎨 UTGIFT OG FORMATERING AV DASHBOARD
    # ==========================================
    print("\n" + "━"*65)
    print("🎙️   D A T A S E T T   D A S H B O A R D   🎙️".center(65))
    print("━"*65)
    
    print(f" 📂 Totalt antall episoder prosessert : {file_count}")
    print(f" 👥 Totalt antall unike stemmer funnet: {len(speaker_stats)}")
    print(f" ⏱️  Totalt utvunnet ren taletid      : {format_time(total_global_seconds)}")
    print("━"*65 + "\n")

    print(" 🏆 TOPP 5 PODCASTER (Mest taletid utvunnet)")
    print(" ┄"*21)
    sorted_podcasts = sorted(podcast_stats.items(), key=lambda x: x[1]["seconds"], reverse=True)
    for rank, (pod, data) in enumerate(sorted_podcasts[:5], 1):
        print(f"  {rank}. {pod[:30].ljust(32)} | {format_time(data['seconds']).ljust(11)} | {len(data['speakers'])} stemmer")
    print("\n")

    # Klargjør stemme-data for de to neste listene
    flat_speakers = []
    for spk, shows in speaker_stats.items():
        tot_sec = sum(s["seconds"] for s in shows.values())
        tot_eps = sum(len(s["episodes"]) for s in shows.values())
        num_shows = len(shows)
        flat_speakers.append((spk, tot_sec, tot_eps, num_shows, shows))

    print(" 👑 THE MARATHON TALKERS (De 5 mest snakkesalige personene)")
    print(" ┄"*21)
    flat_speakers.sort(key=lambda x: x[1], reverse=True)
    for rank, (spk, sec, eps, shows_count, shows) in enumerate(flat_speakers[:5], 1):
        print(f"  {rank}. {spk} | {format_time(sec).ljust(11)} | (Lyd fra {eps} episoder)")
    print("\n")

    print(" 🌍 THE CROSSOVER KINGS (Stemmer funnet i FLERE podcaster)")
    print(" ┄"*21)
    # Sorter på flest shows, deretter tid
    flat_speakers.sort(key=lambda x: (x[3], x[1]), reverse=True)
    crossover_found = False
    for rank, (spk, sec, eps, shows_count, shows) in enumerate(flat_speakers[:5], 1):
        if shows_count > 1:
            crossover_found = True
            shows_list = ", ".join([pod for pod in shows.keys()])
            print(f"  {rank}. {spk} ({format_time(sec)})")
            print(f"     ↳ Hørt i {shows_count} serier: {shows_list[:50]}...")
            print()
            
    if not crossover_found:
        print("  Ingen crossover-stemmer oppdaget ennå!")
        
    print("━"*65 + "\n")

if __name__ == "__main__":
    main()
