# src/name_aggregator.py
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from dotenv import load_dotenv


def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v is not None and v.strip() else default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None and v.strip() else int(default)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None and v.strip() else float(default)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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


@dataclass
class Config:
    s3_bucket: str
    s3_endpoint: Optional[str]
    aws_access_key: Optional[str]
    aws_secret_key: Optional[str]
    s3_base_path: str
    global_prefix: str

    min_episodes_host_intro: int
    min_score_host_intro: float
    min_self_intro_score: float
    margin_ratio: float
    margin_abs: float

    local_temp_dir: Path

    @property
    def s3_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.global_prefix}/"

    @property
    def s3_names_registry_key(self) -> str:
        return f"{self.s3_global_prefix}names.jsonl"

    @property
    def s3_registry_key(self) -> str:
        return f"{self.s3_global_prefix}registry.jsonl"

    @property
    def s3_name_cache_prefix(self) -> str:
        return f"{self.s3_global_prefix}name_cache/"


def load_config() -> Config:
    load_dotenv()
    s3_bucket = env_str("S3_BUCKET")
    if not s3_bucket:
        raise ValueError("Mangler S3_BUCKET")

    return Config(
        s3_bucket=s3_bucket,
        s3_endpoint=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_base_path=env_str("S3_BASE_PATH", "002_speech_dataset"),
        global_prefix=env_str("GLOBAL_PREFIX", "global"),
        min_episodes_host_intro=env_int("NAME_MIN_EPISODES_HOST_INTRO", 2),
        min_score_host_intro=env_float("NAME_MIN_SCORE_HOST_INTRO", 1.10),
        min_self_intro_score=env_float("NAME_MIN_SELF_INTRO_SCORE", 0.80),
        margin_ratio=env_float("NAME_MARGIN_RATIO", 1.40),
        margin_abs=env_float("NAME_MARGIN_ABS", 0.35),
        local_temp_dir=Path(env_str("LOCAL_TEMP_DIR", "temp_name_aggregator")),
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
# Name normalization + scoring
# -----------------------------
def normalize_name(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"[^a-zA-ZÆØÅæøå\-\s']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


EVIDENCE_MULT = {
    "self_intro": 1.00,
    "host_intro": 0.60,
}

# Cap per episode so one episode can't dominate even if multiple mentions
MAX_EPISODE_WEIGHT = 0.85


def candidate_weight(c: Dict[str, Any]) -> float:
    conf = clamp01(float(c.get("confidence", 0.0)))
    ev = c.get("evidence", {}) or {}
    ev_type = str(ev.get("type", "")).strip().lower()
    mult = float(EVIDENCE_MULT.get(ev_type, 0.0))
    return min(MAX_EPISODE_WEIGHT, conf * mult)


def pick_best_display_name(cands: List[Dict[str, Any]]) -> str:
    # Velg display-variant med høyest (weight, confidence) – enkel og grei.
    best = None
    best_score = -1.0
    for c in cands:
        w = candidate_weight(c)
        conf = float(c.get("confidence", 0.0))
        score = w + 0.01 * conf
        if score > best_score:
            best_score = score
            best = c
    return str(best.get("name", "")).strip() if best else ""


def finalize_one(cfg: Config, gid: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = rec.get("candidates", []) or []
    if not candidates:
        rec["canonical_name"] = None
        rec["canonical_status"] = "no_candidates"
        return rec

    # Aggregate with per-(episode,name) max weight (demokrati + anti-episode-spam)
    best_per_episode: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in candidates:
        ep = str(c.get("episode_key") or c.get("episode") or "")
        if not ep:
            continue
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        norm = normalize_name(name)
        if not norm:
            continue

        w = candidate_weight(c)
        key = (ep, norm)
        prev = best_per_episode.get(key)
        if prev is None or w > candidate_weight(prev):
            best_per_episode[key] = c

    # Sum votes across episodes
    sum_by_name: Dict[str, float] = {}
    eps_by_name: Dict[str, set] = {}
    has_self_intro: Dict[str, bool] = {}
    cands_by_name: Dict[str, List[Dict[str, Any]]] = {}

    for (ep, norm), c in best_per_episode.items():
        w = candidate_weight(c)
        sum_by_name[norm] = sum_by_name.get(norm, 0.0) + w
        eps_by_name.setdefault(norm, set()).add(ep)
        ev_type = str((c.get("evidence", {}) or {}).get("type", "")).lower()
        if ev_type == "self_intro":
            has_self_intro[norm] = True
        cands_by_name.setdefault(norm, []).append(c)

    if not sum_by_name:
        rec["canonical_name"] = None
        rec["canonical_status"] = "no_valid_votes"
        return rec

    # sort winners
    ranked = sorted(sum_by_name.items(), key=lambda kv: kv[1], reverse=True)
    best_norm, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    support_eps = len(eps_by_name.get(best_norm, set()))
    best_has_self = bool(has_self_intro.get(best_norm, False))

    # decision rules
    accepted = False
    status = "unknown"

    if best_has_self and best_score >= cfg.min_self_intro_score:
        accepted = True
        status = "confirmed_self_intro"
    else:
        # host_intro trenger fler-episode støtte
        if support_eps >= cfg.min_episodes_host_intro and best_score >= cfg.min_score_host_intro:
            # margin til neste for å unngå “Torstein vs Gunvor” usikkerhet
            ratio_ok = (second_score <= 1e-9) or (best_score / max(second_score, 1e-9) >= cfg.margin_ratio)
            abs_ok = (best_score - second_score) >= cfg.margin_abs
            if ratio_ok and abs_ok:
                accepted = True
                status = "confirmed_multi_episode"

    if accepted:
        display = pick_best_display_name(cands_by_name.get(best_norm, []))
        rec["canonical_name"] = display
        rec["canonical_status"] = status
        rec["canonical_score"] = round(best_score, 4)
        rec["canonical_support_episodes"] = support_eps
    else:
        rec["canonical_name"] = None
        rec["canonical_status"] = "ambiguous_or_insufficient"
        rec["canonical_score"] = round(best_score, 4)
        rec["canonical_support_episodes"] = support_eps

    # debug: topp 3
    rec["vote_top"] = [
        {"name_norm": n, "score": round(s, 4), "episodes": len(eps_by_name.get(n, set())), "has_self": bool(has_self_intro.get(n, False))}
        for n, s in ranked[:3]
    ]
    return rec


def load_speaker_stats(s3, cfg: Config) -> Dict[str, Dict[str, Any]]:
    """
    Leser global/registry.jsonl fra speaker_indexer for å kunne
    (valgfritt) bruke total_seconds til analyse senere.
    Ikke strengt nødvendig for voting, men nyttig for rapportering.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    try:
        obj = s3.get_object(Bucket=cfg.s3_bucket, Key=cfg.s3_registry_key)
        txt = obj["Body"].read().decode("utf-8", errors="replace")
    except Exception:
        return stats

    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gid = str(r.get("global_speaker_id", "")).strip()
        if not gid:
            continue
        stats[gid] = {
            "total_seconds": float(r.get("total_seconds", 0.0)),
            "n_examples": int(r.get("n_examples", 0)),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2: Aggreger name_cache -> global/names.jsonl (voting).")
    parser.add_argument("--max-files", type=int, default=0, help="Begrens antall cache-filer (0=ingen grense).")
    args = parser.parse_args()

    cfg = load_config()
    s3 = get_s3_client(cfg)
    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)

    cache_keys = list_s3_keys(s3, cfg.s3_bucket, cfg.s3_name_cache_prefix, suffix=".json")
    cache_keys.sort()
    if args.max_files and args.max_files > 0:
        cache_keys = cache_keys[: args.max_files]

    # Samle candidates per global speaker
    names: Dict[str, Dict[str, Any]] = {}

    for i, k in enumerate(cache_keys, start=1):
        print(f"[{i}/{len(cache_keys)}] {k}")
        obj = s3.get_object(Bucket=cfg.s3_bucket, Key=k)
        txt = obj["Body"].read().decode("utf-8", errors="replace")
        ep = json.loads(txt)

        episode_key = str(ep.get("episode_key", k))
        for a in ep.get("valid_assignments", []) or []:
            gid = str(a.get("global_speaker_id", "")).strip()
            if not gid:
                continue

            rec = names.get(gid) or {
                "global_speaker_id": gid,
                "candidates": [],
                "canonical_name": None,
                "canonical_status": "unknown",
            }

            # Dedup: episode + name + line_id
            ev = a.get("evidence", {}) or {}
            dedup_key = (episode_key, str(a.get("name", "")).strip(), str(ev.get("line_id", "")).strip())
            if "_dedup" not in rec:
                rec["_dedup"] = set()
            if dedup_key in rec["_dedup"]:
                names[gid] = rec
                continue
            rec["_dedup"].add(dedup_key)

            a2 = dict(a)
            a2["episode_key"] = episode_key
            rec["candidates"].append(a2)

            names[gid] = rec

    # Finalize voting
    speaker_stats = load_speaker_stats(s3, cfg)  # valgfritt
    for gid, rec in names.items():
        rec = finalize_one(cfg, gid, rec)
        # attach stats om tilgjengelig (nice to have)
        if gid in speaker_stats:
            rec["speaker_stats"] = speaker_stats[gid]
        # fjern dedup set før lagring
        rec.pop("_dedup", None)
        names[gid] = rec

    # skriv names.jsonl
    out_path = cfg.local_temp_dir / "names.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for gid in sorted(names.keys()):
            json.dump(names[gid], f, ensure_ascii=False)
            f.write("\n")

    s3.upload_file(str(out_path), cfg.s3_bucket, cfg.s3_names_registry_key)
    print(f"Skrev {len(names)} speakers til {cfg.s3_names_registry_key}")


if __name__ == "__main__":
    main()
