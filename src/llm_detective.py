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

    print(f"🤖 Starter LLM-Detektiven (Vindu: 10 min | Modell: {LLM_MODEL})")

    s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    try:
        response = s3.get_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/global_speakers.json")
        global_speakers = json.loads(response['Body'].read().decode('utf-8'))
    except:
        print("❌ Fant ikke global_speakers.json.")
        return

    klipp_til_global = {}
    for global_id, klipp_liste in global_speakers.items():
        for klipp in klipp_liste:
            klipp_til_global[klipp] = global_id

    navne_stemmer = {} 

    print("🔍 Analyserer episoder...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed/")

    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if not obj['Key'].endswith('.jsonl'): continue
            
            jsonl_key = obj['Key']
            ep_name = jsonl_key.split("/")[-1].replace(".jsonl", "")
            print(f"\n🕵️‍♂️ Analyserer: {ep_name}")
            
            response = s3.get_object(Bucket=BUCKET, Key=jsonl_key)
            content = response['Body'].read().decode('utf-8')
            
            # Vi henter nå de første 10 minuttene (600 sekunder)
            manus = ""
            for line in content.splitlines():
                if not line.strip(): continue
                data = json.loads(line)
                if data['start'] > 600: break 
                manus += f"{data['speaker']}: {data['text']}\n"

            prompt = f"""Analyser dette podcast-manuset og tittelen: '{ep_name}'.
Finn navnet på personene bak SPEAKER_00, SPEAKER_01 osv.

Regler:
1. Svar KUN med JSON.
2. Hvis du er helt sikker på navnet ut fra tekst/tittel, skriv navnet.
3. Hvis navnet IKKE nevnes eller du er usikker, skriv "Ukjent".
4. Bruk kun navn som faktisk forekommer i teksten.

Manus:
{manus}
"""

            try:
                res = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, 
                                    json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}, timeout=60)
                identifiserte_folk = rens_llm_json(res.json()["choices"][0]["message"]["content"])
                
                for spk_kode, navn in identifiserte_folk.items():
                    if navn.lower() != "ukjent":
                        print(f"   💡 Identifisert: {spk_kode} = {navn}")
                        unik_id = f"{ep_name}_{spk_kode}"
                        if unik_id in klipp_til_global:
                            navne_stemmer[klipp_til_global[unik_id]] = navn
                    else:
                        print(f"   ❔ {spk_kode} forblir ukjent.")
                        
            except Exception as e:
                print(f"   ⚠️ Feil: {e}")

    # Bygg fasiten
    gyllen_fasit = {}
    for glob_id, klipp_liste in global_speakers.items():
        # Hvis navnet er "Ukjent" eller ikke funnet, beholder vi GLOBAL_SPEAKER_X navnet
        ekte_navn = navne_stemmer.get(glob_id, glob_id)
        if ekte_navn not in gyllen_fasit:
            gyllen_fasit[ekte_navn] = []
        gyllen_fasit[ekte_navn].extend(klipp_liste)

    s3.put_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/named_speakers.json",
                  Body=json.dumps(gyllen_fasit, indent=4))
    
    print("\n✅ Ferdig! Navngitt fasit er lagret.")

if __name__ == "__main__":
    main()
