from __future__ import annotations

import gc
import inspect
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv

# -----------------------------
# WhisperX diarization API (nyere WhisperX)
# -----------------------------
try:
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
except Exception:
    DiarizationPipeline = None
    assign_word_speakers = None


def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


# -----------------------------
# HF Hub compat patch:
# huggingface_hub v1.x bruker "token=" i stedet for "use_auth_token="
# mens pyannote/whisperx ofte sender fortsatt "use_auth_token="
# -----------------------------
def patch_hf_hub_download_use_auth_token() -> None:
    try:
        import huggingface_hub
        from huggingface_hub import file_download as hf_file_download
    except Exception:
        return

    # Hvis hf_hub_download fortsatt støtter use_auth_token -> ingen patch
    try:
        sig = inspect.signature(huggingface_hub.hf_hub_download)
        if "use_auth_token" in sig.parameters:
            return
    except Exception:
        pass

    orig = huggingface_hub.hf_hub_download

    def compat_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            tok = kwargs.pop("use_auth_token")
            if tok is True:
                tok = (
                    os.getenv("HF_TOKEN")
                    or os.getenv("HUGGINGFACE_TOKEN")
                    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                )
            kwargs["token"] = tok
        return orig(*args, **kwargs)

    # Patch top-level + file_download (noen libs importerer via file_download)
    huggingface_hub.hf_hub_download = compat_hf_hub_download
    try:
        hf_file_download.hf_hub_download = compat_hf_hub_download
    except Exception:
        pass

    # Patch pyannote sin modulvariabel (unngår å måtte scanne sys.modules)
    try:
        import pyannote.audio.core.pipeline as pa_pipeline

        pa_pipeline.hf_hub_download = compat_hf_hub_download
    except Exception:
        pass


@dataclass
class Config:
    # S3
    s3_bucket: str
    s3_endpoint: Optional[str]
    aws_access_key: Optional[str]
    aws_secret_key: Optional[str]
    s3_base_path: str

    # Modell
    whisper_model: str
    fallback_model: str
    skip_ct2_conversion: bool
    ct2_force: bool

    # GPU/compute
    device: str
    batch_size: int
    compute_type: str
    language: str

    # WhisperX steg
    enable_alignment: bool
    enable_diarization: bool
    strict_diarization: bool
    diarization_model: str
    hf_token: Optional[str]

    # speaker hints
    num_speakers: int
    min_speakers: int
    max_speakers: int

    # Filter
    min_confidence: float
    buffer_zone: float
    min_duration: float

    # Paths
    local_temp_dir: Path
    model_cache_dir: Path

    @property
    def s3_input_prefix(self) -> str:
        return f"{self.s3_base_path}/raw/"

    @property
    def s3_output_prefix(self) -> str:
        return f"{self.s3_base_path}/processed/"


def load_config() -> Config:
    load_dotenv()

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # cache dir
    default_cache = Path(os.getenv("MODEL_CACHE_DIR", "/models_cache"))
    if not default_cache.exists():
        default_cache = Path("models_cache")

    cfg = Config(
        s3_bucket=os.getenv("S3_BUCKET", "").strip(),
        s3_endpoint=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_base_path=os.getenv("S3_BASE_PATH", "002_speech_dataset").strip(),
        whisper_model=os.getenv("WHISPER_MODEL", "TheStigh/nb-whisper-large-ct2").strip(),
        fallback_model=os.getenv("FALLBACK_MODEL", "large-v3").strip(),
        skip_ct2_conversion=env_bool("SKIP_CT2_CONVERSION", "1"),
        ct2_force=env_bool("CT2_FORCE", "0"),
        device=os.getenv("DEVICE", "cuda").strip(),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        compute_type=os.getenv("COMPUTE_TYPE", "float16").strip(),
        language=os.getenv("LANGUAGE", "no").strip(),
        enable_alignment=env_bool("ENABLE_ALIGNMENT", "1"),
        enable_diarization=env_bool("ENABLE_DIARIZATION", "1"),
        strict_diarization=env_bool("STRICT_DIARIZATION", "0"),
        diarization_model=os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1").strip(),
        hf_token=hf_token,
        num_speakers=int(os.getenv("NUM_SPEAKERS", "0")),
        min_speakers=int(os.getenv("MIN_SPEAKERS", "0")),
        max_speakers=int(os.getenv("MAX_SPEAKERS", "0")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.90")),
        buffer_zone=float(os.getenv("BUFFER_ZONE", "0.50")),
        min_duration=float(os.getenv("MIN_DURATION", "1.50")),
        local_temp_dir=Path(os.getenv("LOCAL_TEMP_DIR", "temp_processing")),
        model_cache_dir=default_cache,
    )

    if not cfg.s3_bucket:
        raise ValueError("Mangler S3_BUCKET i env/secret!")

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("ADVARSEL: DEVICE=cuda men torch.cuda.is_available()=False. Bruker cpu.")
        cfg.device = "cpu"

    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def get_s3_client(cfg: Config):
    kwargs: Dict[str, Any] = {}
    if cfg.s3_endpoint:
        kwargs["endpoint_url"] = cfg.s3_endpoint
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


def looks_like_ct2_model_id(model_id: str) -> bool:
    m = model_id.lower()
    return m.endswith("-ct2") or "faster-whisper" in m or m.endswith("_ct2")


def map_nbailab_to_ct2(model_id: str) -> Optional[str]:
    """
    Mapper NbAiLab transformer-modeller til kjente CT2 repoer (kvalitet først).
    """
    m = model_id.strip()
    mo = re.match(r"(?i)^NbAiLab/nb-whisper-(tiny|base|small|medium|large)$", m)
    if mo:
        size = mo.group(1).lower()
        return f"TheStigh/nb-whisper-{size}-ct2"
    return None


def resolve_whisper_model_path(cfg: Config) -> str:
    m = cfg.whisper_model.strip()

    # Lokal sti?
    if Path(m).exists():
        return m

    # Innebygd faster-whisper sizes: "base", "large-v3", etc
    if "/" not in m:
        return m

    # Allerede CT2 repo-id
    if looks_like_ct2_model_id(m):
        return m

    # SKIP_CT2_CONVERSION=1: map til CT2 hvis mulig
    if cfg.skip_ct2_conversion:
        mapped = map_nbailab_to_ct2(m)
        if mapped:
            print(f"--- SKIP_CT2_CONVERSION=1: Mapper {m} -> {mapped} ---")
            return mapped

        print(
            f"ADVARSEL: SKIP_CT2_CONVERSION=1 men WHISPER_MODEL={m} er ikke CT2 og har ingen mapping. "
            f"Fallback til '{cfg.fallback_model}'."
        )
        return cfg.fallback_model

    # Hvis du virkelig vil konvertere i runtime (ikke anbefalt), kan du legge det inn her igjen.
    print(
        "ADVARSEL: SKIP_CT2_CONVERSION=0 men runtime-konvertering er slått av i denne 'robuste' profilen. "
        f"Fallback til '{cfg.fallback_model}'."
    )
    return cfg.fallback_model


def s3_result_key(cfg: Config, s3_input_key: str) -> Tuple[str, str]:
    relative = s3_input_key.replace(cfg.s3_input_prefix, "", 1)
    out_key = f"{cfg.s3_output_prefix}{relative[:-4]}.jsonl"
    return relative, out_key


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
    bad_zones: List[Tuple[float, float]] = []
    if diarize_df is not None and isinstance(diarize_df, pd.DataFrame) and not diarize_df.empty:
        df = diarize_df.sort_values("start").reset_index(drop=True)
        for i in range(len(df) - 1):
            curr_end = float(df.loc[i, "end"])
            next_start = float(df.loc[i + 1, "start"])
            if curr_end > next_start:
                zone_start = next_start - cfg.buffer_zone
                zone_end = curr_end + cfg.buffer_zone
                bad_zones.append((zone_start, zone_end))

    clean: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker", "Unknown")

        if not text:
            continue
        if (end - start) < cfg.min_duration:
            continue
        if any((start < z_end and end > z_start) for z_start, z_end in bad_zones):
            continue

        avg_score = compute_avg_word_score(seg)
        if avg_score is not None and avg_score < cfg.min_confidence:
            continue

        clean.append(
            {
                "audio_file": source_name,
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end,
                "score": None if avg_score is None else round(avg_score, 3),
            }
        )
    return clean


def init_diarization(cfg: Config):
    """
    Laster diarization pipeline én gang.
    Robust: hvis gated/mangler token/ikke akseptert -> disable diarization (med mindre STRICT_DIARIZATION=1)
    """
    if not cfg.enable_diarization:
        return None

    if not cfg.hf_token:
        msg = "ENABLE_DIARIZATION=1 men HF_TOKEN mangler. Skipper diarization."
        if cfg.strict_diarization:
            raise RuntimeError(msg)
        print("ADVARSEL:", msg)
        return None

    if DiarizationPipeline is None or assign_word_speakers is None:
        msg = "WhisperX diarization API (whisperx.diarize) er ikke tilgjengelig i denne installasjonen."
        if cfg.strict_diarization:
            raise RuntimeError(msg)
        print("ADVARSEL:", msg)
        return None

    patch_hf_hub_download_use_auth_token()

    print(f"--- Laster diarization-modell: {cfg.diarization_model} (én gang) ---")
    try:
        diarize_model = DiarizationPipeline(
            model_name=cfg.diarization_model,
            use_auth_token=cfg.hf_token,
            device=cfg.device,
        )
        # Ekstra sanity: noen ganger kan intern model ende opp None hvis nedlasting feiler
        if getattr(diarize_model, "model", None) is None:
            raise RuntimeError(
                "DiarizationPipeline lastet, men underliggende model er None. "
                "Dette skjer typisk hvis modellen er gated og vilkår ikke er akseptert."
            )
        return diarize_model

    except Exception as e:
        print("\nADVARSEL: Kunne ikke laste diarization pipeline.")
        print("Årsak:", repr(e))
        print(
            "\nFor å aktivere diarization må du:\n"
            "  1) logge inn på Hugging Face med kontoen som HF_TOKEN tilhører\n"
            "  2) åpne modellen og akseptere vilkår:\n"
            "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3) sørge for at token har 'read' og at det er samme bruker som aksepterte vilkårene.\n"
        )
        if cfg.strict_diarization:
            raise
        print("Fortsetter uten diarization (speaker='Unknown').\n")
        return None


def process_audio() -> None:
    cfg = load_config()
    s3 = get_s3_client(cfg)

    model_path = resolve_whisper_model_path(cfg)

    print(f"[entry] DEVICE={cfg.device}  COMPUTE_TYPE={cfg.compute_type}  MODEL={model_path}")
    print(f"--- Initialiserer WhisperX ({cfg.device}) ---")
    print(f"--- Prosjekt: {cfg.s3_base_path} ---")

    model = whisperx.load_model(model_path, cfg.device, compute_type=cfg.compute_type, language=cfg.language)

    # Alignment én gang
    align_model = None
    align_metadata = None
    if cfg.enable_alignment:
        print("--- Laster alignment-modell (én gang) ---")
        align_model, align_metadata = whisperx.load_align_model(language_code=cfg.language, device=cfg.device)

    # Diarization én gang (robust)
    diarize_model = init_diarization(cfg)

    print(f"Lister filer i s3://{cfg.s3_bucket}/{cfg.s3_input_prefix}...")
    files = list_mp3_keys(cfg, s3)
    print(f"Fant {len(files)} mp3-filer.")

    for idx, s3_key in enumerate(files, start=1):
        relative, out_key = s3_result_key(cfg, s3_key)

        # idempotent skip
        try:
            s3.head_object(Bucket=cfg.s3_bucket, Key=out_key)
            print(f"Skipper {relative} (Ferdig).")
            continue
        except Exception:
            pass

        print(f"\n[{idx}/{len(files)}] Laster ned {relative}...")

        local_mp3 = cfg.local_temp_dir / relative
        local_jsonl = cfg.local_temp_dir / f"{relative}.jsonl"
        local_mp3.parent.mkdir(parents=True, exist_ok=True)
        local_jsonl.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3.download_file(cfg.s3_bucket, s3_key, str(local_mp3))

            audio = whisperx.load_audio(str(local_mp3))

            # 1) Transcribe
            result = model.transcribe(audio, batch_size=cfg.batch_size, language=cfg.language)

            # 2) Align
            if cfg.enable_alignment and align_model is not None and align_metadata is not None:
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    cfg.device,
                    return_char_alignments=False,
                )

            # 3) Diarize + assign speakers
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
                    print(f"ADVARSEL: diarization feilet for {relative}, fortsetter uten speaker labels: {e}")
                    diarize_df = None

            # 4) Filter + skriv JSONL
            clean = filter_data(cfg, result, diarize_df, relative)

            with open(local_jsonl, "w", encoding="utf-8") as f:
                for row in clean:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")

            print(f"Laster opp til s3://{cfg.s3_bucket}/{out_key}")
            s3.upload_file(str(local_jsonl), cfg.s3_bucket, out_key)

        except Exception as e:
            print(f"FEIL på {relative}: {e}")

        finally:
            try:
                if local_mp3.exists():
                    local_mp3.unlink()
            except Exception:
                pass
            try:
                if local_jsonl.exists():
                    local_jsonl.unlink()
            except Exception:
                pass

            if cfg.device.startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    process_audio()
