import os
import json
import boto3
import requests
from collections import defaultdict
from dotenv import load_dotenv

def rens_llm_json(tekst):
    tekst = tekst.strip()
    if tekst.startswith("```json"): tekst = tekst[7:]
    elif tekst.startswith("```"): tekst = tekst[3:]
    if tekst.endswith("```"): tekst = tekst[:-3]
    return json.loads(tekst.strip())

def main():
    load_dotenv()
    BUCKET = os.getenv("S3_BUCKET", "ml-data")
    BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
    
    API_BASE = os.getenv("OPENAI_API_BASE", "[https://api.openai.com/v1](https://api.openai.com/v1)").rstrip('/')
    API_URL = f"{API_BASE}/chat/completions"
    API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key-for-local-vllm")
    LLM_MODEL = os.getenv("LLM_MODEL", "vllm/model1")

    s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    print("📥 Laster inn klynger fra Jobb B...")
    try:
        response = s3.get_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/global_speakers.json")
        global_speakers = json.loads(response['Body'].read().decode('utf-8'))
    except:
        print("❌ Fant ikke global_speakers.json.")
        return

    # Omvendt oppslag
    klipp_til_global = {}
    for global_id, klipp_liste in global_speakers.items():
        for klipp in klipp_liste:
            klipp_til_global[klipp] = global_id

    # Her lagrer vi alle forslagene for hver GLOBAL_SPEAKER_ID
    # Struktur: { "GLOBAL_SPEAKER_1": {"Torstein": 5, "Ukjent": 2} }
    navne_forslag = defaultdict(lambda: defaultdict(int))

    print("🔍 Analyserer manuset (første 10 min) per episode...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed/")

    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if not obj['Key'].endswith('.jsonl'): continue
            
            jsonl_key = obj['Key']
            ep_name = jsonl_key.split("/")[-1].replace(".jsonl", "")
            
            response = s3.get_object(Bucket=BUCKET, Key=jsonl_key)
            lines = response['Body'].read().decode('utf-8').splitlines()
            
            manus = ""
            for line in lines:
                data = json.loads(line)
                if data['start'] > 600: break # 10 minutter vindu
                manus += f"{data['speaker']}: {data['text']}\n"

            prompt = f"Analyser podcast-episoden '{ep_name}'. Hvem er SPEAKER_00, SPEAKER_01 osv? Svar KUN JSON. Bruk 'Ukjent' hvis usikker.\n\nManus:\n{manus}"

            try:
                res = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, 
                                    json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}, timeout=60)
                identifiserte = rens_llm_json(res.json()["choices"][0]["message"]["content"])
                
                for spk_kode, navn in identifiserte.items():
                    unik_id = f"{ep_name}_{spk_kode}"
                    if unik_id in klipp_til_global:
                        glob_id = klipp_til_global[unik_id]
                        navne_forslag[glob_id][navn] += 1
                        print(f"   💡 {ep_name}: {spk_kode} foreslått som '{navn}'")
            except:
                continue

    # --- DEMOKRATI-LOGIKK ---
    endelig_mapping = {}
    print("\n⚖️ Løser navnekonflikter...")
    
    for glob_id, forslag in navne_forslag.items():
        # Fjern "Ukjent" fra avstemningen hvis det finnes andre alternativer
        reelle_navn = {n: c for n, c in forslag.items() if n.lower() != "ukjent"}
        
        if reelle_navn:
            vinner = max(reelle_navn, key=reelle_navn.get)
            endelig_mapping[glob_id] = vinner
            if len(reelle_navn) > 1:
                print(f"   ⚠️ Konflikt for {glob_id}: {dict(forslag)} -> Valgte '{vinner}'")
        else:
            endelig_mapping[glob_id] = glob_id # Behold original ID hvis alt er ukjent

    # Bygg gylne fasit
    gyllen_fasit = {}
    for glob_id, klipp_liste in global_speakers.items():
        ekte_navn = endelig_mapping.get(glob_id, glob_id)
        if ekte_navn not in gyllen_fasit:
            gyllen_fasit[ekte_navn] = []
        gyllen_fasit[ekte_navn].extend(klipp_liste)

    s3.put_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/named_speakers.json",
                  Body=json.dumps(gyllen_fasit, indent=4))
    
    print(f"\n✅ Ferdig! Navngitt fasit lagret med {len(gyllen_fasit)} unike personer.")

if __name__ == "__main__":
    main()
