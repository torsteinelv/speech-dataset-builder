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


def cuda_cleanup() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "out of memory" in msg
        or "cuda failed with error out of memory" in msg
        or "cuda out of memory" in msg
        or "cublas" in msg and "alloc" in msg
    )


# -----------------------------
# HF Hub compat patch:
# huggingface_hub v1.x bruker "token=" i stedet for "use_auth_token="
# men pyannote/whisperx i noen stacker sender fortsatt "use_auth_token="
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

    huggingface_hub.hf_hub_download = compat_hf_hub_download
    try:
        hf_file_download.hf_hub_download = compat_hf_hub_download
    except Exception:
        pass

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

    # GPU/compute
    device: str
    batch_size: int
    compute_type: str
    language: str

    # OOM strategy
    auto_batch_cap: bool
    allow_model_downsize_on_oom: bool
    allow_cpu_fallback_on_oom: bool

    # WhisperX steg
    enable_alignment: bool
    align_device: str
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


def gpu_total_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return None


def cap_batch_size(cfg: Config) -> None:
    if not cfg.auto_batch_cap:
        return
    if not cfg.device.startswith("cuda"):
        return
    total = gpu_total_mem_gb()
    if total is None:
        return

    # konservative caps for large models
    if total <= 6:
        cap = 1
    elif total <= 8:
        cap = 2
    elif total <= 12:
        cap = 4
    elif total <= 16:
        cap = 8
    else:
        cap = 16

    if cfg.batch_size > cap:
        print(f"ADVARSEL: GPU {total:.1f}GB -> capper BATCH_SIZE {cfg.batch_size} -> {cap}")
        cfg.batch_size = cap


def load_config() -> Config:
    load_dotenv()

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

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
        device=os.getenv("DEVICE", "cuda").strip(),
        batch_size=int(os.getenv("BATCH_SIZE", "8")),
        compute_type=os.getenv("COMPUTE_TYPE", "float16").strip(),
        language=os.getenv("LANGUAGE", "no").strip(),
        auto_batch_cap=env_bool("AUTO_BATCH_CAP", "1"),
        allow_model_downsize_on_oom=env_bool("ALLOW_MODEL_DOWNSIZE_ON_OOM", "1"),
        allow_cpu_fallback_on_oom=env_bool("ALLOW_CPU_FALLBACK_ON_OOM", "0"),
        enable_alignment=env_bool("ENABLE_ALIGNMENT", "1"),
        # ✅ viktig: alignment på CPU som default (reduserer GPU OOM)
        align_device=os.getenv("ALIGN_DEVICE", "cpu").strip(),
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

    cap_batch_size(cfg)
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
    m = model_id.strip()
    mo = re.match(r"(?i)^NbAiLab/nb-whisper-(tiny|base|small|medium|large)$", m)
    if mo:
        size = mo.group(1).lower()
        return f"TheStigh/nb-whisper-{size}-ct2"
    return None


def resolve_whisper_model_path(cfg: Config) -> str:
    m = cfg.whisper_model.strip()

    if Path(m).exists():
        return m

    if "/" not in m:
        return m

    if looks_like_ct2_model_id(m):
        return m

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

    # hvis du virkelig vil runtime-konvertering kan vi legge det inn igjen, men du ville ha stabilt oppsett.
    print(
        "ADVARSEL: SKIP_CT2_CONVERSION=0 men runtime-konvertering er deaktivert i denne stabile profilen. "
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
                bad_zones.append((next_start - cfg.buffer_zone, curr_end + cfg.buffer_zone))

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
    if not cfg.enable_diarization:
        return None

    if not cfg.hf_token:
        msg = "ENABLE_DIARIZATION=1 men HF_TOKEN mangler. Skipper diarization."
        if cfg.strict_diarization:
            raise RuntimeError(msg)
        print("ADVARSEL:", msg)
        return None

    if DiarizationPipeline is None or assign_word_speakers is None:
        msg = "WhisperX diarization API (whisperx.diarize) ikke tilgjengelig."
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
        if getattr(diarize_model, "model", None) is None:
            raise RuntimeError(
                "DiarizationPipeline lastet, men underliggende model er None (typisk gated/ikke akseptert)."
            )
        return diarize_model
    except Exception as e:
        print("\nADVARSEL: Kunne ikke laste diarization pipeline.")
        print("Årsak:", repr(e))
        print(
            "\nFor å aktivere diarization må du akseptere vilkår på HF-kontoen som tokenet tilhører:\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "Deretter vil diarization fungere automatisk.\n"
        )
        if cfg.strict_diarization:
            raise
        print("Fortsetter uten diarization (speaker='Unknown').\n")
        return None


def compute_type_candidates(primary: str) -> List[str]:
    out = []
    def add(x: str):
        if x and x not in out:
            out.append(x)

    add(primary)

    # gode OOM-fallbacks for CT2/faster-whisper
    add("int8_float16")
    add("int8")

    return out


def batch_candidates(primary: int) -> List[int]:
    out = []
    b = max(1, int(primary))
    while True:
        if b not in out:
            out.append(b)
        if b == 1:
            break
        b = max(1, b // 2)
    return out


def model_downsize_chain(model_id: str) -> List[str]:
    """
    Hvis du bruker TheStigh/nb-whisper-<size>-ct2 kan vi auto-downgrade ved OOM:
      large -> medium -> small -> base -> tiny
    """
    sizes = ["large", "medium", "small", "base", "tiny"]
    m = re.match(r"^(TheStigh/nb-whisper-)(tiny|base|small|medium|large)(-ct2)$", model_id)
    if not m:
        return [model_id]

    prefix, cur, suffix = m.group(1), m.group(2), m.group(3)
    idx = sizes.index(cur)
    chain = [model_id]
    for s in sizes[idx + 1 :]:
        chain.append(f"{prefix}{s}{suffix}")
    return chain


class ASRManager:
    def __init__(self, cfg: Config, model_id: str):
        self.cfg = cfg
        self.device = cfg.device
        self.language = cfg.language
        self.model_id = model_id
        self.compute_type = cfg.compute_type
        self.model = None
        self.load(self.model_id, self.compute_type, self.device)

    def unload(self):
        try:
            if self.model is not None:
                del self.model
        except Exception:
            pass
        self.model = None
        cuda_cleanup()

    def load(self, model_id: str, compute_type: str, device: str):
        self.unload()
        self.model_id = model_id
        self.compute_type = compute_type
        self.device = device
        print(f"--- Laster ASR: model={model_id} device={device} compute_type={compute_type} ---")
        self.model = whisperx.load_model(model_id, device, compute_type=compute_type, language=self.language)

    def transcribe_oom_safe(self, audio):
        model_chain = model_downsize_chain(self.model_id) if self.cfg.allow_model_downsize_on_oom else [self.model_id]
        ct_chain = compute_type_candidates(self.compute_type)
        b_chain = batch_candidates(self.cfg.batch_size)

        last_err: Optional[Exception] = None

        for midx, m in enumerate(model_chain):
            for cidx, ct in enumerate(ct_chain):
                # last inn (eller reload) modellen for denne kombinasjonen
                if self.model is None or self.model_id != m or self.compute_type != ct or self.device != self.cfg.device:
                    self.load(m, ct, self.cfg.device)

                for b in b_chain:
                    try:
                        res = self.model.transcribe(audio, batch_size=b, language=self.language)
                        # "lås inn" suksess-parametre for resten av runnet
                        if b != self.cfg.batch_size:
                            print(f"INFO: batch_size justert for stabilitet: {self.cfg.batch_size} -> {b}")
                            self.cfg.batch_size = b
                        return res
                    except Exception as e:
                        last_err = e
                        if is_oom_error(e):
                            print(f"OOM: model={m} compute={ct} batch={b} -> prøver fallback...")
                            # Reload for å komme tilbake fra evt. korrupt GPU state etter OOM
                            self.load(m, ct, self.cfg.device)
                            continue
                        raise

        # Hvis alt på GPU feilet og CPU fallback er tillatt:
        if self.cfg.allow_cpu_fallback_on_oom and self.cfg.device.startswith("cuda"):
            print("ADVARSEL: Alle GPU-fallbacks feilet. Prøver CPU fallback (kan være tregt).")
            try:
                self.load(model_chain[-1], "int8", "cpu")
                return self.model.transcribe(audio, batch_size=1, language=self.language)
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Kunne ikke transkribere pga OOM etter alle fallbacks. Siste error: {last_err!r}")


def process_audio() -> None:
    cfg = load_config()
    s3 = get_s3_client(cfg)

    resolved_model = resolve_whisper_model_path(cfg)
    print(f"[entry] DEVICE={cfg.device} ALIGN_DEVICE={cfg.align_device} BATCH={cfg.batch_size} MODEL={resolved_model}")

    # ASR manager (OOM-safe)
    asr = ASRManager(cfg, resolved_model)

    # Alignment én gang (på CPU som default)
    align_model = None
    align_metadata = None
    if cfg.enable_alignment:
        print(f"--- Laster alignment-modell (én gang) på {cfg.align_device} ---")
        align_model, align_metadata = whisperx.load_align_model(language_code=cfg.language, device=cfg.align_device)

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

            # 1) Transcribe (OOM-safe)
            result = asr.transcribe_oom_safe(audio)

            # 2) Align (valgfritt)
            if cfg.enable_alignment and align_model is not None and align_metadata is not None:
                # align kjører på cfg.align_device (default cpu)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    cfg.align_device,
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
            cuda_cleanup()


if __name__ == "__main__":
    process_audio()
