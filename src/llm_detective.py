import os
import json
import boto3
import requests
from collections import defaultdict
from dotenv import load_dotenv

def rens_llm_json(tekst):
    """Fjerner eventuell markdown-formatering rundt JSON-svaret fra LLM-en."""
    tekst = tekst.strip()
    if tekst.startswith("```json"): tekst = tekst[7:]
    elif tekst.startswith("```"): tekst = tekst[3:]
    if tekst.endswith("```"): tekst = tekst[:-3]
    return json.loads(tekst.strip())

def main():
    load_dotenv()
    
    # S3 Variabler
    BUCKET = os.getenv("S3_BUCKET", "ml-data")
    BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
    
    # LLM Variabler (Standard = OpenAI, men kan overstyres til din vLLM)
    # Eks overstyring: OPENAI_API_BASE="[https://vllm-api.thorsland.no/v1](https://vllm-api.thorsland.no/v1)"
    API_BASE = os.getenv("OPENAI_API_BASE", "[https://api.openai.com/v1](https://api.openai.com/v1)").rstrip('/')
    API_URL = f"{API_BASE}/chat/completions"
    API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key-for-local-vllm")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini") # Bytt til f.eks "vllm/model1" i .env

    print(f"🤖 Starter LLM-Detektiven (Modell: {LLM_MODEL} | API: {API_BASE})")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    print("📥 Laster inn gruppefasiten (Jobb B) fra S3...")
    try:
        response = s3.get_object(Bucket=BUCKET, Key=f"{BASE_PATH}/metadata/global_speakers.json")
        global_speakers = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print("❌ Fant ikke global_speakers.json. Har du kjørt Jobb B?")
        return

    # Omvendt ordbok for kjappe oppslag
    klipp_til_global = {}
    for global_id, klipp_liste in global_speakers.items():
        for klipp in klipp_liste:
            klipp_til_global[klipp] = global_id

    navne_stemmer = {} # GLOBAL_SPEAKER_X -> "Erna Solberg"

    print("🔍 Henter episoder for analyse...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{BASE_PATH}/processed/")

    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if not obj['Key'].endswith('.jsonl'): continue
            
            jsonl_key = obj['Key']
            ep_name = jsonl_key.split("/")[-1].replace(".jsonl", "")

            print(f"\n🕵️‍♂️ Sender til LLM: {ep_name}")
            
            # Hent transkripsjonen
            response = s3.get_object(Bucket=BUCKET, Key=jsonl_key)
            content = response['Body'].read().decode('utf-8')
            
            # Bygg manuskriptet for de første 3 minuttene (180 sek)
            manus = ""
            for line in content.splitlines():
                if not line.strip(): continue
                data = json.loads(line)
                if data['start'] > 600: break
                manus += f"{data['speaker']}: {data['text']}\n"

            if not manus: continue

            prompt = f"""Du er en super-detektiv som analyserer podcast-manuskripter. 
Her er de første 3 minuttene av episoden: '{ep_name}'.

Oppgave:
Identifiser alle personene som snakker (SPEAKER_00, SPEAKER_01, osv.). 
Bruk informasjonen i tittelen og manuset (hvem som introduserer hvem, hvem som blir stilt spørsmål, hvem som styrer ordet) for å finne deres fulle navn.

Regler:
1. Svar KUN med et gyldig JSON-objekt.
2. Nøkkelen skal være SPEAKER-koden (f.eks. "SPEAKER_01").
3. Verdien skal være personens fulle navn (f.eks. "Torstein Thorsland" eller "Erna Solberg").
4. Hvis du er usikker på et navn, gjett basert på det mest sannsynlige ut fra kontekst.

Manuskript:
{manus}
"""

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }

            try:
                res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                res.raise_for_status() # Fanger opp HTTP-feil (f.eks 401 Unauthorized)
                svar_tekst = res.json()["choices"][0]["message"]["content"]
                
                identifiserte_folk = rens_llm_json(svar_tekst)
                
                for spk_kode, navn in identifiserte_folk.items():
                    print(f"   💡 Fant: {spk_kode} = {navn}")
                    
                    unik_id = f"{ep_name}_{spk_kode}"
                    if unik_id in klipp_til_global:
                        glob_id = klipp_til_global[unik_id]
                        navne_stemmer[glob_id] = navn
                        
            except Exception as e:
                print(f"   ⚠️ Klarte ikke å analysere med LLM: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"      Feilmelding fra API: {e.response.text}")

    # Bygg den nye fasiten
    gyllen_fasit = {}
    for glob_id, klipp_liste in global_speakers.items():
        ekte_navn = navne_stemmer.get(glob_id, glob_id) 
        # Hvis navnet allerede finnes (flere GLOBAL_SPEAKERS ble gjettet til samme person), slå dem sammen
        if ekte_navn not in gyllen_fasit:
            gyllen_fasit[ekte_navn] = []
        gyllen_fasit[ekte_navn].extend(klipp_liste)

    final_key = f"{BASE_PATH}/metadata/named_speakers.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=final_key,
        Body=json.dumps(gyllen_fasit, indent=4),
        ContentType="application/json"
    )
    
    print("\n" + "="*40)
    print("🏆 LLM-DETEKTIVEN ER FERDIG! 🏆")
    print("="*40)
    print(f"Den nye, navngitte fasiten er lagret i S3 som: {final_key}")

if __name__ == "__main__":
    main()
