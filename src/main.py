import os
import gc
import json
import torch
import whisperx
import pandas as pd
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()

# --- KONFIGURASJON ---
INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
MODEL_CACHE_DIR = Path("/app/output/models") # Vi lagrer modellen på disken så vi slipper å konvertere hver gang
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda"
BATCH_SIZE = 32         
COMPUTE_TYPE = "float16" 

# Valg av modell: Her bruker vi NbAiLab sin
NBAILAB_MODEL_ID = "NbAiLab/nb-whisper-large" 

# Filter-innstillinger
MIN_CONFIDENCE = 0.90   
BUFFER_ZONE = 0.5       
MIN_DURATION = 1.5      
NUM_SPEAKERS = 3        

def get_norwegian_model_path():
    """
    Sjekker om NbAiLab-modellen finnes i CT2-format.
    Hvis ikke, laster den ned og konverterer den automatisk.
    """
    model_name_safe = NBAILAB_MODEL_ID.replace("/", "_")
    ct2_output_dir = MODEL_CACHE_DIR / f"{model_name_safe}_ct2"

    if ct2_output_dir.exists():
        print(f"--- Fant ferdig konvertert modell: {ct2_output_dir} ---")
        return str(ct2_output_dir)
    
    print(f"--- Fant ingen konvertert modell. Laster ned og konverterer {NBAILAB_MODEL_ID}... ---")
    print("Dette tar litt tid første gang (ca. 2-5 minutter).")
    
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kommando for å konvertere fra HuggingFace til CTranslate2 (WhisperX format)
    # Vi kjører dette som en subprocess
    cmd = [
        "ct2-transformers-converter",
        "--model", NBAILAB_MODEL_ID,
        "--output_dir", str(ct2_output_dir),
        "--quantization", COMPUTE_TYPE,
        "--low_cpu_mem_usage"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("--- Konvertering vellykket! ---")
        return str(ct2_output_dir)
    except subprocess.CalledProcessError as e:
        print(f"FEIL under modell-konvertering: {e}")
        print("Faller tilbake til standard 'large-v3' modell fra OpenAI.")
        return "large-v3"

def process_audio():
    if not HF_TOKEN:
        raise ValueError("Mangler HF_TOKEN i .env filen!")

    # 1. Hent riktig modell (NbAiLab eller fallback)
    model_path = get_norwegian_model_path()
    
    print(f"--- Laster modell fra: {model_path} ---")
    # Merk: Når vi laster en lokal path, trenger vi ikke 'language' argumentet i load_model, 
    # men vi setter det i transcribe()
    model = whisperx.load_model(model_path, DEVICE, compute_type=COMPUTE_TYPE, language="no")
    
    files = list(INPUT_DIR.glob("*.mp3"))
    print(f"Fant {len(files)} filer i input-mappen.")

    for i, file_path in enumerate(files):
        filename = file_path.name
        output_path = OUTPUT_DIR / f"{file_path.stem}.jsonl"

        if output_path.exists():
            print(f"Skipper {filename} (allerede ferdig).")
            continue

        print(f"\nProsesserer {i+1}/{len(files)}: {filename}...")

        try:
            # A. Transkribering
            audio = whisperx.load_audio(str(file_path))
            # VIKTIG: Sett language="no" eksplisitt her
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language="no")

            # B. Alignment
            model_a, metadata = whisperx.load_align_model(language_code="no", device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            del model_a, metadata
            gc.collect()
            torch.cuda.empty_cache()

            # C. Diarization
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
            diarize_segments = diarize_model(audio, min_speakers=NUM_SPEAKERS, max_speakers=NUM_SPEAKERS)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()

            # D. Avansert Filtrering
            clean_data = filter_data(result, diarize_segments, filename)

            # E. Lagring
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in clean_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")
            
            print(f"Ferdig! Lagret {len(clean_data)} rene setninger.")

        except Exception as e:
            print(f"FEIL på fil {filename}: {e}")

        gc.collect()
        torch.cuda.empty_cache()

def filter_data(result, diarize_segments, filename):
    # (Samme filtrerings-kode som før - ingen endringer her)
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
            "audio_file": filename,
            "speaker": seg["speaker"],
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
            "score": round(avg_score, 3)
        })
    
    return clean_segments

if __name__ == "__main__":
    process_audio()
