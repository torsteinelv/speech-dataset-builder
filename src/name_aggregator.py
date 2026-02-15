from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from dotenv import load_dotenv


# -----------------------------
# Utils
# -----------------------------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v is not None and v.strip() else default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None and v.strip() else int(default)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None and v.strip() else float(default)


def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def norm_name(name: str) -> str:
    # Lett normalisering: trim + collapse whitespace
    s = " ".join(name.strip().split())
    return s


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
    global_prefix: str
    cache_version: str

    min_confirm_count: int
    min_confirm_score: float

    local_temp_dir: Path

    @property
    def s3_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.global_prefix}/"

    @property
    def s3_cache_prefix(self) -> str:
        return f"{self.s3_global_prefix}name_cache_{self.cache_version}/"

    @property
    def s3_names_key(self) -> str:
        return f"{self.s3_global_prefix}names.jsonl"


def load_config() -> Config:
    load_dotenv()

    bucket = env_str("S3_BUCKET", "ml-data")
    base = env_str("S3_BASE_PATH", "002_speech_dataset")

    return Config(
        s3_bucket=bucket,
        s3_endpoint=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_base_path=base,
        global_prefix=env_str("GLOBAL_PREFIX", "global"),
        cache_version=env_str("NAME_CACHE_VERSION", "v2"),
        min_confirm_count=env_int("NAME_MIN_CONFIRM_COUNT", 1),
        min_confirm_score=env_float("NAME_MIN_CONFIRM_SCORE", 0.8),
        local_temp_dir=Path(env_str("LOCAL_TEMP_DIR", "temp_name_aggregator")).resolve(),
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
# Aggregation
# -----------------------------
def load_cache_json(s3, bucket: str, key: str) -> Optional[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        txt = obj["Body"].read().decode("utf-8", errors="replace")
        return json.loads(txt)
    except Exception:
        return None


def aggregate(cfg: Config, s3) -> List[Dict[str, Any]]:
    cache_keys = list_s3_keys(s3, cfg.s3_bucket, cfg.s3_cache_prefix, suffix=".json")
    cache_keys.sort()

    print(f"📦 Fant {len(cache_keys)} cachefiler under {cfg.s3_cache_prefix}")

    # gid -> name -> {score_sum, count, examples[]}
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(lambda: {"score_sum": 0.0, "count": 0, "examples": []}))
    episodes_with_gid: Dict[str, set] = defaultdict(set)

    for k in cache_keys:
        blob = load_cache_json(s3, cfg.s3_bucket, k)
        if not blob:
            continue

        ep_key = blob.get("episode_key") or k
        audio_file = blob.get("audio_file")

        for a in blob.get("valid_assignments", []) or []:
            if not isinstance(a, dict):
                continue
            gid = str(a.get("global_speaker_id", "")).strip()
            name = norm_name(str(a.get("name", "")).strip())
            conf = safe_float(a.get("confidence", 0.0), 0.0)

            if not gid or not name:
                continue

            bucket_entry = stats[gid][name]
            bucket_entry["score_sum"] += conf
            bucket_entry["count"] += 1
            episodes_with_gid[gid].add(ep_key)

            # ta vare på litt evidens (maks 3)
            if len(bucket_entry["examples"]) < 3:
                bucket_entry["examples"].append(
                    {
                        "episode_key": ep_key,
                        "audio_file": audio_file,
                        "confidence": conf,
                        "evidence": a.get("evidence", {}),
                    }
                )

    out_rows: List[Dict[str, Any]] = []

    for gid, name_map in stats.items():
        # finn beste navn
        best_name = None
        best_score = -1.0
        best_count = -1

        for name, st in name_map.items():
            score = float(st["score_sum"])
            cnt = int(st["count"])
            # prioriter score, så count
            if (score > best_score) or (score == best_score and cnt > best_count):
                best_name = name
                best_score = score
                best_count = cnt

        if not best_name:
            continue

        n_eps = len(episodes_with_gid.get(gid, set()))
        status = "guessed"
        if best_count >= cfg.min_confirm_count and best_score >= cfg.min_confirm_score:
            status = "confirmed"

        out_rows.append(
            {
                "global_speaker_id": gid,
                "canonical_name": best_name,
                "canonical_score": round(best_score, 3),
                "canonical_status": status,
                "n_assignments": best_count,
                "n_episodes": n_eps,
                "examples": name_map[best_name]["examples"],
            }
        )

    out_rows.sort(key=lambda r: (r["canonical_status"] != "confirmed", -r["canonical_score"], -r["n_assignments"]))
    return out_rows


def write_names(cfg: Config, s3, rows: List[Dict[str, Any]]) -> None:
    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = (cfg.local_temp_dir / "names.jsonl").resolve()

    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    s3.upload_file(str(tmp), cfg.s3_bucket, cfg.s3_names_key)


def main() -> None:
    cfg = load_config()
    s3 = get_s3_client(cfg)

    rows = aggregate(cfg, s3)

    # FAILSAFE: ikke overskriv names.jsonl med tom fil
    if not rows:
        print("❌ Aggregator fant 0 speakers med navn. Skriver IKKE names.jsonl (failsafe).")
        print("   Sjekk at extractor skriver cache til samme NAME_CACHE_VERSION og at prefix matcher.")
        return

    write_names(cfg, s3, rows)
    print(f"✅ Skrev {len(rows)} speakers til {cfg.s3_names_key}")


if __name__ == "__main__":
    main()
