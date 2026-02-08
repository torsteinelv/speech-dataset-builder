import os
import gc
import json
import torch
import whisperx
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()

# --- KONFIGURASJON ---
INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda"
BATCH_SIZE = 32         # Økt for 20GB VRAM
COMPUTE_TYPE = "float16" 

# Filter-innstillinger
MIN_CONFIDENCE = 0.90   
BUFFER_ZONE = 0.5       
MIN_DURATION = 1.5      
NUM_SPEAKERS = 3        # Sett til None hvis det varierer

def process_audio():
    # Sjekk at vi har token
    if not HF_TOKEN:
        raise ValueError("Mangler HF_TOKEN i .env filen!")

    # 1. Last modeller
    print("--- Laster modeller (WhisperX Large-v3) ---")
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language="no")
    
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
            result = model.transcribe(audio, batch_size=BATCH_SIZE)

            # B. Alignment
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
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

        # Tøm GPU fullstendig før neste fil
        gc.collect()
        torch.cuda.empty_cache()

def filter_data(result, diarize_segments, filename):
    """
    Filtrerer bort overlapp, støy og korte setninger.
    """
    # 1. Finn overlappende soner
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

        # Sjekk overlapp
        is_bad = False
        for b_start, b_end in bad_zones:
            if (seg["start"] < b_end) and (seg["end"] > b_start):
                is_bad = True
                break
        if is_bad: continue

        # Sjekk score
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
