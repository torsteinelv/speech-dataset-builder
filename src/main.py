from __future__ import annotations

import gc
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv


# WhisperX diarization API: i nyere versjoner ligger dette i whisperx.diarize
# (ikke whisperx.DiarizationPipeline)
try:
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
except Exception:
    DiarizationPipeline = None
    assign_word_speakers = None


@dataclass
class Config:
    # --- S3 ---
    s3_bucket: str
    s3_endpoint: Optional[str]
    aws_access_key: Optional[str]
    aws_secret_key: Optional[str]
    s3_base_path: str

    # --- Modell / GPU ---
    whisper_model: str
    device: str
    batch_size: int
    compute_type: str
    language: str
    hf_token: Optional[str]

    # --- WhisperX steg ---
    enable_alignment: bool
    enable_diarization: bool
    diarization_model: str
    num_speakers: int
    min_speakers: int
    max_speakers: int

    # --- Filter ---
    min_confidence: float
    buffer_zone: float
    min_duration: float

    # --- Paths ---
    local_temp_dir: Path
    model_cache_dir: Path
    fallback_model: str

    @property
    def s3_input_prefix(self) -> str:
        return f"{self.s3_base_path}/raw/"

    @property
    def s3_output_prefix(self) -> str:
        return f"{self.s3_base_path}/processed/"


def env_bool(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> Config:
    load_dotenv()

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # Bruk /models_cache hvis den finnes (matcher Docker ENV i imaget)
    default_cache = Path(os.getenv("MODEL_CACHE_DIR", "/models_cache"))
    if not default_cache.exists():
        default_cache = Path("models_cache")

    cfg = Config(
        s3_bucket=os.getenv("S3_BUCKET", "").strip(),
        s3_endpoint=os.getenv("S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL".lower()),
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_base_path=os.getenv("S3_BASE_PATH", "002_speech_dataset").strip(),
        whisper_model=os.getenv("WHISPER_MODEL", "NbAiLab/nb-whisper-base").strip(),
        device=os.getenv("DEVICE", "cuda").strip(),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        compute_type=os.getenv("COMPUTE_TYPE", "float16").strip(),
        language=os.getenv("LANGUAGE", "no").strip(),
        hf_token=hf_token,
        enable_alignment=env_bool("ENABLE_ALIGNMENT", "1"),
        enable_diarization=env_bool("ENABLE_DIARIZATION", "1"),
        diarization_model=os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1").strip(),
        num_speakers=int(os.getenv("NUM_SPEAKERS", "0")),       # 0 = auto
        min_speakers=int(os.getenv("MIN_SPEAKERS", "0")),       # 0 = ikke brukt
        max_speakers=int(os.getenv("MAX_SPEAKERS", "0")),       # 0 = ikke brukt
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.90")),
        buffer_zone=float(os.getenv("BUFFER_ZONE", "0.5")),
        min_duration=float(os.getenv("MIN_DURATION", "1.5")),
        local_temp_dir=Path(os.getenv("LOCAL_TEMP_DIR", "temp_processing")),
        model_cache_dir=default_cache,
        fallback_model=os.getenv("FALLBACK_MODEL", "large-v3").strip(),
    )

    # Valider det viktigste
    if not cfg.s3_bucket:
        raise ValueError("Mangler S3_BUCKET i .env / env!")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("ADVARSEL: DEVICE=cuda men torch.cuda.is_available()=False. Faller tilbake til cpu.")
        cfg.device = "cpu"

    return cfg


def get_s3_client(cfg: Config):
    kwargs: Dict[str, Any] = {}
    if cfg.s3_endpoint:
        kwargs["endpoint_url"] = cfg.s3_endpoint

    # Hvis du bruker IAM-role i clusteret kan keys være tomme – boto3 håndterer det.
    if cfg.aws_access_key and cfg.aws_secret_key:
        kwargs["aws_access_key_id"] = cfg.aws_access_key
        kwargs["aws_secret_access_key"] = cfg.aws_secret_key

    return boto3.client("s3", **kwargs)


def list_mp3_keys(cfg: Config, s3) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"Bucket": cfg.s3_bucket, "Prefix": cfg.s3_input_prefix}
        if token:
            params["ContinuationToken"] = token

        resp = s3.list_objects_v2(**params)
        for obj in resp.get("Contents", []):
            k = obj.get("Key", "")
            if k.lower().endswith(".mp3"):
                keys.append(k)

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    return keys


def looks_like_ct2_model(model_id: str) -> bool:
    # Mange CT2-modeller på HF bruker -ct2 eller faster-whisper i navnet.
    m = model_id.lower()
    return m.endswith("-ct2") or "faster-whisper" in m


def get_model_path(cfg: Config) -> str:
    # "large-v3", "base", osv (ingen /) → bruk direkte
    if "/" not in cfg.whisper_model:
        return cfg.whisper_model

    # Hvis det ser ut som CT2 repo → ikke konverter
    if looks_like_ct2_model(cfg.whisper_model) or env_bool("SKIP_CT2_CONVERSION", "0"):
        return cfg.whisper_model

    model_name_safe = cfg.whisper_model.replace("/", "_")
    out_dir = cfg.model_cache_dir / f"{model_name_safe}_ct2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hvis allerede konvertert → bruk cache
    if (out_dir / "model.bin").exists() or (out_dir / "config.json").exists():
        print(f"--- Fant bufret CT2-modell: {out_dir} ---")
        return str(out_dir)

    print(f"--- Konverterer {cfg.whisper_model} til CT2... ---")
    cmd = [
        "ct2-transformers-converter",
        "--model", cfg.whisper_model,
        "--output_dir", str(out_dir),
        "--quantization", cfg.compute_type,
        "--low_cpu_mem_usage",
    ]

    try:
        subprocess.run(cmd, check=True)
        return str(out_dir)
    except Exception as e:
        print(f"Konvertering feilet ({e}). Fallback til '{cfg.fallback_model}'.")
        return cfg.fallback_model


def s3_result_key(cfg: Config, s3_input_key: str) -> Tuple[str, str]:
    # Input: 002_speech_dataset/raw/Podcast/ep.mp3
    # Output: 002_speech_dataset/processed/Podcast/ep.jsonl
    relative_path = s3_input_key.replace(cfg.s3_input_prefix, "", 1)
    out_key = f"{cfg.s3_output_prefix}{relative_path[:-4]}.jsonl"
    return relative_path, out_key


def compute_avg_word_score(seg: Dict[str, Any]) -> Optional[float]:
    words = seg.get("words") or []
    scores: List[float] = []
    for w in words:
        s = w.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))
    if not scores:
        return None
    return sum(scores) / len(scores)


def filter_data(
    cfg: Config,
    result: Dict[str, Any],
    diarize_df: Optional[pd.DataFrame],
    source_name: str,
) -> List[Dict[str, Any]]:
    # Overlapp-filter basert på diarization segments
    bad_zones: List[Tuple[float, float]] = []

    if diarize_df is not None and isinstance(diarize_df, pd.DataFrame) and not diarize_df.empty:
        df = diarize_df.copy()
        df = df.sort_values(by="start").reset_index(drop=True)

        for i in range(len(df) - 1):
            curr_start = float(df.loc[i, "start"])
            curr_end = float(df.loc[i, "end"])
            next_start = float(df.loc[i + 1, "start"])
            next_end = float(df.loc[i + 1, "end"])

            if curr_end > next_start:
                zone_start = max(curr_start, next_start) - cfg.buffer_zone
                zone_end = min(curr_end, next_end) + cfg.buffer_zone
                bad_zones.append((zone_start, zone_end))

    clean_segments: List[Dict[str, Any]] = []

    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker", "Unknown")

        if not text:
            continue
        if (end - start) < cfg.min_duration:
            continue

        # Skip “bad overlap zones”
        is_bad = False
        for b_start, b_end in bad_zones:
            if (start < b_end) and (end > b_start):
                is_bad = True
                break
        if is_bad:
            continue

        avg_score = compute_avg_word_score(seg)
        if avg_score is not None and avg_score < cfg.min_confidence:
            continue

        clean_segments.append(
            {
                "audio_file": source_name,
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end,
                "score": None if avg_score is None else round(avg_score, 3),
            }
        )

    return clean_segments


def process_audio() -> None:
    cfg = load_config()
    s3 = get_s3_client(cfg)

    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = get_model_path(cfg)

    print(f"--- Initialiserer WhisperX ({cfg.device}) ---")
    print(f"--- Prosjekt: {cfg.s3_base_path} ---")
    model = whisperx.load_model(model_path, cfg.device, compute_type=cfg.compute_type, language=cfg.language)

    # Alignment: last én gang
    align_model = None
    align_metadata = None
    if cfg.enable_alignment:
        print("--- Laster alignment-modell (én gang) ---")
        align_model, align_metadata = whisperx.load_align_model(language_code=cfg.language, device=cfg.device)

    # Diarization: last én gang
    diarize_model = None
    if cfg.enable_diarization:
        if not cfg.hf_token:
            print("ADVARSEL: ENABLE_DIARIZATION=1 men HF_TOKEN mangler. Skipper diarization.")
        elif DiarizationPipeline is None or assign_word_speakers is None:
            raise RuntimeError(
                "WhisperX diarization API ikke tilgjengelig. "
                "Du må ha en WhisperX-versjon som har whisperx.diarize."
            )
        else:
            print(f"--- Laster diarization-modell: {cfg.diarization_model} (én gang) ---")
            diarize_model = DiarizationPipeline(
                model_name=cfg.diarization_model,
                use_auth_token=cfg.hf_token,
                device=cfg.device,
            )

    print(f"Lister filer i s3://{cfg.s3_bucket}/{cfg.s3_input_prefix}...")
    all_files = list_mp3_keys(cfg, s3)
    print(f"Fant {len(all_files)} mp3-filer.")

    for idx, s3_key in enumerate(all_files, start=1):
        relative_path, out_key = s3_result_key(cfg, s3_key)

        # Skip ferdige
        try:
            s3.head_object(Bucket=cfg.s3_bucket, Key=out_key)
            print(f"Skipper {relative_path} (Ferdig).")
            continue
        except Exception:
            pass

        print(f"\n[{idx}/{len(all_files)}] Laster ned {relative_path}...")

        local_mp3 = cfg.local_temp_dir / relative_path
        local_json = cfg.local_temp_dir / f"{relative_path}.jsonl"
        local_mp3.parent.mkdir(parents=True, exist_ok=True)
        local_json.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3.download_file(cfg.s3_bucket, s3_key, str(local_mp3))

            audio = whisperx.load_audio(str(local_mp3))

            # 1) Transcribe
            result = model.transcribe(audio, batch_size=cfg.batch_size, language=cfg.language)

            # 2) Align (valgfritt)
            if cfg.enable_alignment and align_model is not None and align_metadata is not None:
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    cfg.device,
                    return_char_alignments=False,
                )

            # 3) Diarize + assign speakers (valgfritt)
            diarize_df: Optional[pd.DataFrame] = None
            if diarize_model is not None:
                try:
                    dia_kwargs: Dict[str, Any] = {}
                    if cfg.num_speakers > 0:
                        dia_kwargs["num_speakers"] = cfg.num_speakers
                    else:
                        if cfg.min_speakers > 0:
                            dia_kwargs["min_speakers"] = cfg.min_speakers
                        if cfg.max_speakers > 0:
                            dia_kwargs["max_speakers"] = cfg.max_speakers

                    diarize_df = diarize_model(audio, **dia_kwargs)
                    result = assign_word_speakers(diarize_df, result)
                except Exception as e:
                    print(f"ADVARSEL: diarization feilet for {relative_path}, fortsetter uten speaker labels: {e}")
                    diarize_df = None

            # 4) Filter + skriv JSONL
            clean_data = filter_data(cfg, result, diarize_df, relative_path)

            with open(local_json, "w", encoding="utf-8") as f:
                for entry in clean_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")

            print(f"Laster opp til s3://{cfg.s3_bucket}/{out_key}")
            s3.upload_file(str(local_json), cfg.s3_bucket, out_key)

        except Exception as e:
            print(f"FEIL på {relative_path}: {e}")

        finally:
            try:
                if local_mp3.exists():
                    local_mp3.unlink()
            except Exception:
                pass
            try:
                if local_json.exists():
                    local_json.unlink()
            except Exception:
                pass

            # Prøv å rydde tomme mapper i temp
            try:
                if local_mp3.parent.exists():
                    local_mp3.parent.rmdir()
            except Exception:
                pass

            if cfg.device.startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    process_audio()
