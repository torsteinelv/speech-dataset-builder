from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import numpy as np
import torch
from dotenv import load_dotenv

# --- MONKEY PATCH START (Må være før pyannote imports) ---
# Dette fikser "TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'"
import huggingface_hub

# Vi tar vare på den originale funksjonen
_original_hf_hub_download = huggingface_hub.hf_hub_download

def _patched_hf_hub_download(*args, **kwargs):
    # Hvis gamle biblioteker sender 'use_auth_token', bytter vi navn til 'token'
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_hub_download(*args, **kwargs)

# Vi erstatter funksjonen i biblioteket med vår fikset versjon
huggingface_hub.hf_hub_download = _patched_hf_hub_download
# --- MONKEY PATCH SLUTT ---

from pyannote.audio import Model, Inference
from pyannote.core import Segment


# -----------------------------
# Utils
# -----------------------------
def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None and v.strip() else float(default)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None and v.strip() else int(default)


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v is not None and v.strip() else default


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def list_s3_keys(s3, bucket: str, prefix: str, suffix: Optional[str] = None) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj.get("Key", "")
            if suffix is None or k.endswith(suffix):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    s3_bucket: str
    s3_endpoint: Optional[str]
    aws_access_key: Optional[str]
    aws_secret_key: Optional[str]
    s3_base_path: str

    processed_prefix: str
    processed_global_prefix: str
    global_prefix: str

    # embedding
    embedding_model: str
    embedding_device: str
    hf_token: Optional[str]

    # thresholds + selection
    sim_high: float
    sim_low: float

    min_total_speech_s: float
    target_speech_s: float
    max_segments: int
    min_segment_s: float
    embed_window_s: float

    local_temp_dir: Path

    @property
    def s3_processed_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.processed_prefix}/"

    @property
    def s3_raw_prefix(self) -> str:
        return f"{self.s3_base_path}/raw/"

    @property
    def s3_processed_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.processed_global_prefix}/"

    @property
    def s3_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.global_prefix}/"

    @property
    def s3_global_registry_key(self) -> str:
        return f"{self.s3_global_prefix}registry.jsonl"


def load_config() -> Config:
    load_dotenv()

    s3_bucket = env_str("S3_BUCKET", "")
    if not s3_bucket:
        raise ValueError("Mangler S3_BUCKET")

    return Config(
        s3_bucket=s3_bucket,
        s3_endpoint=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_base_path=env_str("S3_BASE_PATH", "002_speech_dataset"),
        processed_prefix=env_str("PROCESSED_PREFIX", "processed"),
        processed_global_prefix=env_str("PROCESSED_GLOBAL_PREFIX", "processed_global"),
        global_prefix=env_str("GLOBAL_PREFIX", "global"),
        embedding_model=env_str("SPEAKER_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"),
        embedding_device=env_str("SPEAKER_EMBEDDING_DEVICE", "cuda"),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        sim_high=env_float("GLOBAL_SIM_HIGH", 0.78),
        sim_low=env_float("GLOBAL_SIM_LOW", 0.68),
        min_total_speech_s=env_float("GLOBAL_MIN_TOTAL_SPEECH_SECONDS", 12.0),
        target_speech_s=env_float("GLOBAL_TARGET_SPEECH_SECONDS", 30.0),
        max_segments=env_int("GLOBAL_MAX_SEGMENTS", 12),
        min_segment_s=env_float("GLOBAL_MIN_SEGMENT_SECONDS", 1.6),
        embed_window_s=env_float("GLOBAL_EMBED_WINDOW_SECONDS", 3.0),
        local_temp_dir=Path(env_str("LOCAL_TEMP_DIR", "temp_global_index")),
    )


def get_s3_client(cfg: Config):
    kwargs: Dict[str, Any] = {}
    if cfg.s3_endpoint:
        kwargs["endpoint_url"] = cfg.s3_endpoint
    if cfg.aws_access_key and cfg.aws_secret_key:
        kwargs["aws_access_key_id"] = cfg.aws_access_key
        kwargs["aws_secret_access_key"] = cfg.aws_secret_key
    return boto3.client("s3", **kwargs)


# -----------------------------
# Speaker embedding
# -----------------------------
class SpeakerEmbedder:
    """
    Bruker pyannote Model + Inference (window="whole") og inference.crop(...) for excerpt embeddings.
    Ref: model card for pyannote/wespeaker-voxceleb-resnet34-LM. (Basic/Advanced usage)
    """

    def __init__(self, model_id: str, device: str, hf_token: Optional[str] = None):
        kwargs: Dict[str, Any] = {}
        # Mange modeller er åpne uten token, men støtt token hvis satt.
        if hf_token:
            kwargs["token"] = hf_token

        # Her vil monkey-patchen over fikse kallet hvis pyannote bruker use_auth_token internt
        self.model = Model.from_pretrained(model_id, **kwargs)
        self.inference = Inference(self.model, window="whole")
        self.inference.to(torch.device(device))

    def embed_excerpt(self, audio_path: str, start_s: float, end_s: float) -> np.ndarray:
        excerpt = Segment(float(start_s), float(end_s))
        emb = self.inference.crop(audio_path, excerpt)  # (1 x D) numpy array
        v = np.asarray(emb).reshape(-1).astype(np.float32)
        return normalize(v)


# -----------------------------
# Registry format
# -----------------------------
def load_registry(s3, cfg: Config) -> List[Dict[str, Any]]:
    if not s3_exists(s3, cfg.s3_bucket, cfg.s3_global_registry_key):
        return []
    tmp = cfg.local_temp_dir / "registry.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(cfg.s3_bucket, cfg.s3_global_registry_key, str(tmp))
    rows = read_jsonl(tmp)
    # sørg for korrekt typing/normalisering (defensivt)
    for r in rows:
        r["embedding"] = normalize(np.asarray(r["embedding"], dtype=np.float32)).tolist()
        r["n_examples"] = int(r.get("n_examples", 0))
        r["total_seconds"] = float(r.get("total_seconds", 0.0))
    return rows


def save_registry(s3, cfg: Config, registry: List[Dict[str, Any]]) -> None:
    tmp = cfg.local_temp_dir / "registry.jsonl"
    write_jsonl(tmp, registry)
    s3.upload_file(str(tmp), cfg.s3_bucket, cfg.s3_global_registry_key)


def best_match(centroid: np.ndarray, registry: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
    if not registry:
        return None, float("-inf")
    mat = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in registry], axis=0)  # (N, D)
    sims = mat @ centroid  # cosine hvis begge er L2-normalisert
    idx = int(np.argmax(sims))
    return registry[idx], float(sims[idx])


def update_centroid(old: np.ndarray, old_weight: float, new: np.ndarray, new_weight: float) -> np.ndarray:
    w = float(max(old_weight, 1e-6) + max(new_weight, 1e-6))
    v = (old * old_weight + new * new_weight) / w
    return normalize(v)


# -----------------------------
# Segment picking
# -----------------------------
def pick_embedding_windows(
    segments: List[Dict[str, Any]],
    *,
    min_seg_s: float,
    embed_window_s: float,
    max_segments: int,
    target_speech_s: float,
) -> List[Tuple[float, float, float]]:
    """
    Returnerer liste av (start, end, weight) for embedding.
    """
    cands: List[Tuple[float, float, float, float]] = []  # start, end, score, dur
    for s in segments:
        try:
            st = float(s["start"])
            en = float(s["end"])
            dur = max(0.0, en - st)
            if dur < min_seg_s:
                continue
            score = s.get("score")
            sc = float(score) if isinstance(score, (int, float)) else 1.0
            cands.append((st, en, sc, dur))
        except Exception:
            continue

    cands.sort(key=lambda x: (x[3] * x[2]), reverse=True)

    picked: List[Tuple[float, float, float]] = []
    total = 0.0

    for st, en, sc, dur in cands:
        if len(picked) >= max_segments:
            break
        if total >= target_speech_s:
            break

        win = min(embed_window_s, dur)
        mid = (st + en) / 2.0
        wst = max(st, mid - win / 2.0)
        wen = min(en, mid + win / 2.0)

        weight = win
        picked.append((wst, wen, weight))
        total += win

    return picked


# -----------------------------
# Per-episode processing
# -----------------------------
def group_by_local_speaker(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        spk = str(r.get("speaker", "Unknown"))
        groups.setdefault(spk, []).append(r)
    return groups


def local_speaker_stats(segments: List[Dict[str, Any]]) -> float:
    total = 0.0
    for s in segments:
        try:
            total += max(0.0, float(s["end"]) - float(s["start"]))
        except Exception:
            pass
    return total


def make_global_id() -> str:
    return f"spk_{uuid.uuid4().hex[:12]}"


def process_one_episode(
    s3,
    cfg: Config,
    embedder: SpeakerEmbedder,
    registry: List[Dict[str, Any]],
    processed_key: str,
) -> None:
    relative = processed_key.replace(cfg.s3_processed_prefix, "", 1)  # podcast/episode.jsonl
    out_processed_key = f"{cfg.s3_processed_global_prefix}{relative}"
    link_key = f"{cfg.s3_global_prefix}links/{relative[:-5]}.json"  # drop ".jsonl" -> ".json"

    if s3_exists(s3, cfg.s3_bucket, out_processed_key) and s3_exists(s3, cfg.s3_bucket, link_key):
        print(f"SKIP (global ferdig): {relative}")
        return

    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)
    local_processed = cfg.local_temp_dir / "processed" / relative
    local_processed.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(cfg.s3_bucket, processed_key, str(local_processed))
    rows = read_jsonl(local_processed)
    if not rows:
        print(f"ADVARSEL: tom fil {relative}")
        return

    audio_rel = str(rows[0].get("audio_file", "")).strip()
    if not audio_rel:
        print(f"ADVARSEL: mangler audio_file i {relative}")
        return

    raw_key = f"{cfg.s3_raw_prefix}{audio_rel}"
    local_mp3 = cfg.local_temp_dir / "raw" / audio_rel
    local_mp3.parent.mkdir(parents=True, exist_ok=True)
    if not local_mp3.exists():
        s3.download_file(cfg.s3_bucket, raw_key, str(local_mp3))

    groups = group_by_local_speaker(rows)

    link_result: Dict[str, Any] = {
        "audio_file": audio_rel,
        "processed_key": processed_key,
        "links": [],
    }

    for local_spk, segs in groups.items():
        if local_spk.lower() == "unknown":
            link_result["links"].append(
                {
                    "local_speaker": local_spk,
                    "global_speaker_id": None,
                    "decision": "skip_unknown",
                    "similarity": None,
                    "total_speech_s": round(local_speaker_stats(segs), 3),
                }
            )
            continue

        total_s = local_speaker_stats(segs)
        if total_s < cfg.min_total_speech_s:
            link_result["links"].append(
                {
                    "local_speaker": local_spk,
                    "global_speaker_id": None,
                    "decision": "too_little_audio",
                    "similarity": None,
                    "total_speech_s": round(total_s, 3),
                }
            )
            continue

        windows = pick_embedding_windows(
            segs,
            min_seg_s=cfg.min_segment_s,
            embed_window_s=cfg.embed_window_s,
            max_segments=cfg.max_segments,
            target_speech_s=cfg.target_speech_s,
        )
        if not windows:
            link_result["links"].append(
                {
                    "local_speaker": local_spk,
                    "global_speaker_id": None,
                    "decision": "no_windows",
                    "similarity": None,
                    "total_speech_s": round(total_s, 3),
                }
            )
            continue

        embs: List[np.ndarray] = []
        weights: List[float] = []
        for st, en, w in windows:
            try:
                v = embedder.embed_excerpt(str(local_mp3), st, en)
                embs.append(v)
                weights.append(float(w))
            except Exception as e:
                print(f"ADVARSEL: embedding feilet local={local_spk} [{st:.2f},{en:.2f}] {e}")

        if not embs:
            link_result["links"].append(
                {
                    "local_speaker": local_spk,
                    "global_speaker_id": None,
                    "decision": "embedding_failed",
                    "similarity": None,
                    "total_speech_s": round(total_s, 3),
                }
            )
            continue

        W = float(sum(weights))
        centroid = normalize(sum(v * w for v, w in zip(embs, weights)) / max(W, 1e-6))

        best, sim = best_match(centroid, registry)

        decision: str
        assigned_id: str
        candidate_id: Optional[str] = None
        candidate_sim: Optional[float] = None

        if best is None:
            decision = "new"
            assigned_id = make_global_id()
            registry.append(
                {
                    "global_speaker_id": assigned_id,
                    "embedding": centroid.tolist(),
                    "n_examples": 1,
                    "total_seconds": float(sum(weights)),
                    "created_at": None,
                    "updated_at": None,
                }
            )

        else:
            candidate_id = str(best["global_speaker_id"])
            candidate_sim = float(sim)

            if sim >= cfg.sim_high:
                decision = "linked"
                assigned_id = candidate_id

                old_emb = np.asarray(best["embedding"], dtype=np.float32)
                old_w = float(best.get("total_seconds", 0.0))
                new_w = float(sum(weights))
                new_emb = update_centroid(old_emb, old_w, centroid, new_w)

                best["embedding"] = new_emb.tolist()
                best["n_examples"] = int(best.get("n_examples", 0)) + 1
                best["total_seconds"] = old_w + new_w

            elif sim <= cfg.sim_low:
                decision = "new"
                assigned_id = make_global_id()
                registry.append(
                    {
                        "global_speaker_id": assigned_id,
                        "embedding": centroid.tolist(),
                        "n_examples": 1,
                        "total_seconds": float(sum(weights)),
                        "created_at": None,
                        "updated_at": None,
                    }
                )

            else:
                decision = "review"
                assigned_id = make_global_id()
                registry.append(
                    {
                        "global_speaker_id": assigned_id,
                        "embedding": centroid.tolist(),
                        "n_examples": 1,
                        "total_seconds": float(sum(weights)),
                        "created_at": None,
                        "updated_at": None,
                    }
                )

        link_result["links"].append(
            {
                "local_speaker": local_spk,
                "global_speaker_id": assigned_id,
                "decision": decision,
                "similarity": None if candidate_sim is None else round(candidate_sim, 4),
                "candidate_global_speaker_id": candidate_id,
                "total_speech_s": round(total_s, 3),
                "used_for_embedding_s": round(float(sum(weights)), 3),
                "n_embedding_windows": len(weights),
            }
        )

    mapping: Dict[str, str] = {
        x["local_speaker"]: x["global_speaker_id"]
        for x in link_result["links"]
        if x.get("global_speaker_id") is not None
    }

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["global_speaker_id"] = mapping.get(str(r.get("speaker", "")))
        out_rows.append(rr)

    local_out = cfg.local_temp_dir / "processed_global" / relative
    write_jsonl(local_out, out_rows)
    s3.upload_file(str(local_out), cfg.s3_bucket, out_processed_key)

    local_link = cfg.local_temp_dir / "links" / f"{relative[:-5]}.json"
    local_link.parent.mkdir(parents=True, exist_ok=True)
    local_link.write_text(json.dumps(link_result, ensure_ascii=False, indent=2), encoding="utf-8")
    s3.upload_file(str(local_link), cfg.s3_bucket, link_key)

    print(f"OK: {relative} -> global speakers linket ({len(link_result['links'])} lokale speakere).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Global speaker linking (lokal diarization -> global speaker IDs).")
    parser.add_argument("--max-files", type=int, default=0, help="Begrens antall episoder (0=ingen grense).")
    args = parser.parse_args()

    cfg = load_config()
    s3 = get_s3_client(cfg)
    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing processed jsonl under s3://{cfg.s3_bucket}/{cfg.s3_processed_prefix}")
    keys = list_s3_keys(s3, cfg.s3_bucket, cfg.s3_processed_prefix, suffix=".jsonl")
    keys.sort()

    if args.max_files and args.max_files > 0:
        keys = keys[: args.max_files]

    registry = load_registry(s3, cfg)
    print(f"Registry loaded: {len(registry)} globale speakere.")

    embedder = SpeakerEmbedder(cfg.embedding_model, cfg.embedding_device, cfg.hf_token)

    for i, k in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] {k}")
        process_one_episode(s3, cfg, embedder, registry, k)
        save_registry(s3, cfg, registry)

    print("Ferdig.")


if __name__ == "__main__":
    main()
