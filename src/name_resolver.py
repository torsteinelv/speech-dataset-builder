# src/name_resolver.py
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests
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


def read_jsonl_text(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def extract_first_json_object(s: str) -> str:
    """
    Modell-output kan av og til komme med litt tekst rundt JSON.
    Vi plukker ut første {...}-blokk på en defensiv måte.
    """
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Fant ikke JSON-objekt i LLM-respons.")
    return s[start : end + 1]


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

    processed_global_prefix: str
    global_prefix: str

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    temperature: float

    max_episode_lines: int
    max_speaker_lines: int

    local_temp_dir: Path

    @property
    def s3_processed_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.processed_global_prefix}/"

    @property
    def s3_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.global_prefix}/"

    @property
    def s3_names_registry_key(self) -> str:
        return f"{self.s3_global_prefix}names.jsonl"


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
        processed_global_prefix=env_str("PROCESSED_GLOBAL_PREFIX", "processed_global"),
        global_prefix=env_str("GLOBAL_PREFIX", "global"),
        llm_base_url=env_str("LLM_BASE_URL"),
        llm_api_key=env_str("LLM_API_KEY"),
        llm_model=env_str("LLM_MODEL", "gpt-4.1-mini"),
        temperature=env_float("NAME_RESOLVER_TEMPERATURE", 0.0),
        max_episode_lines=env_int("NAME_RESOLVER_MAX_EPISODE_LINES", 220),
        max_speaker_lines=env_int("NAME_RESOLVER_MAX_SPEAKER_LINES", 40),
        local_temp_dir=Path(env_str("LOCAL_TEMP_DIR", "temp_name_resolver")),
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
# LLM client (OpenAI-compatible Chat Completions)
# -----------------------------
def llm_chat(cfg: Config, messages: List[Dict[str, str]]) -> str:
    if not cfg.llm_base_url or not cfg.llm_api_key:
        raise ValueError("LLM_BASE_URL/LLM_API_KEY mangler. (Sett dem i .env)")

    url = cfg.llm_base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.llm_api_key}"}
    payload = {
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": cfg.temperature,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# -----------------------------
# Name resolution
# -----------------------------
INTRO_PATTERNS = [
    r"\bvelkommen\b",
    r"\bmed oss\b",
    r"\bgjest\b",
    r"\bi studio\b",
    r"\bi dag har vi\b",
    r"\bdu hører på\b",
    r"\bmitt navn\b",
    r"\bjeg heter\b",
]


def build_episode_prompt_rows(rows: List[Dict[str, Any]], cfg: Config) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Returnerer:
      - episode_snippet: tekst med utvalgte linjer
      - per_speaker_lines: {global_speaker_id: [linjer]}
    """
    # 1) Ta de første N linjene som “kontekst”
    first = rows[: cfg.max_episode_lines]

    # 2) Ta alle “intro-liknende” linjer i hele episoden (begrenset)
    intro_lines: List[Dict[str, Any]] = []
    pat = re.compile("|".join(INTRO_PATTERNS), re.IGNORECASE)
    for r in rows:
        txt = str(r.get("text", "")).strip()
        if not txt:
            continue
        if pat.search(txt):
            intro_lines.append(r)
        if len(intro_lines) >= cfg.max_episode_lines:
            break

    # 3) Per speaker: første M linjer + linjer med “jeg heter / mitt navn”
    per: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        gid = r.get("global_speaker_id")
        if not gid:
            continue
        per.setdefault(gid, [])
        if len(per[gid]) < cfg.max_speaker_lines:
            per[gid].append(r)

    # Ekstra: linjer som matcher sterke self-ID mønstre
    strong_pat = re.compile(r"\bjeg heter\b|\bmitt navn\b", re.IGNORECASE)
    for r in rows:
        gid = r.get("global_speaker_id")
        if not gid:
            continue
        if strong_pat.search(str(r.get("text", ""))):
            per.setdefault(gid, [])
            per[gid].append(r)

    # bygg snippet tekst
    def fmt_line(r: Dict[str, Any]) -> str:
        return (
            f"[{r.get('start', 0):.2f}-{r.get('end', 0):.2f}] "
            f"{r.get('global_speaker_id') or 'NO_GID'} "
            f"(local={r.get('speaker')}): {str(r.get('text', '')).strip()}"
        )

    lines_out: List[str] = []
    lines_out.append("=== FIRST_LINES ===")
    lines_out.extend(fmt_line(r) for r in first)

    lines_out.append("\n=== INTRO_LINES ===")
    # dedup litt
    seen = set()
    for r in intro_lines:
        k = (r.get("start"), r.get("end"), r.get("global_speaker_id"), r.get("text"))
        if k in seen:
            continue
        seen.add(k)
        lines_out.append(fmt_line(r))

    lines_out.append("\n=== PER_SPEAKER_SAMPLES ===")
    for gid, items in per.items():
        lines_out.append(f"\n-- SPEAKER {gid} --")
        # sort på tid, og ta maks M
        items_sorted = sorted(items, key=lambda x: float(x.get("start", 0.0)))
        for r in items_sorted[: cfg.max_speaker_lines]:
            lines_out.append(fmt_line(r))

    return "\n".join(lines_out), per


def resolve_names_for_episode(cfg: Config, audio_file: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    snippet, _ = build_episode_prompt_rows(rows, cfg)

    system = (
        "Du er et system som trekker ut navn på podcast-deltakere basert på tekstbevis.\n"
        "VIKTIG:\n"
        "- Du får kun et utdrag av transkripsjonen med speaker-id og tidsstempler.\n"
        "- Du MÅ kun foreslå et navn hvis navnet er eksplisitt nevnt i utdraget.\n"
        "- Du MÅ inkludere et sitat (quote) som er et eksakt utdrag fra input.\n"
        "- Hvis du ikke finner eksplisitt bevis: returner unknown.\n"
        "- Ikke gjett.\n"
        "Svar KUN med ett JSON-objekt."
    )

    user = (
        f"EPISODE: {audio_file}\n\n"
        "Her er utdraget:\n"
        f"{snippet}\n\n"
        "Oppgave:\n"
        "1) Finn hvilke global_speaker_id som kan knyttes til et personnavn.\n"
        "2) For hver: gi name, confidence (0-1), quote, start, end, og en kort reason.\n"
        "3) Returner også en liste over speaker-id som forblir unknown.\n\n"
        "JSON-skjema:\n"
        "{\n"
        '  "episode": "...",\n'
        '  "assignments": [\n'
        "    {\n"
        '      "global_speaker_id": "spk_...",\n'
        '      "name": "Fullt Navn",\n'
        '      "confidence": 0.0,\n'
        '      "evidence": {"quote": "...", "start": 0.0, "end": 0.0, "reason": "..."}\n'
        "    }\n"
        "  ],\n"
        '  "unknown": ["spk_...", "..."]\n'
        "}\n"
    )

    content = llm_chat(cfg, [{"role": "system", "content": system}, {"role": "user", "content": user}])
    obj = json.loads(extract_first_json_object(content))

    # Valider “quote” finnes faktisk i snippet (anti-hallusinasjon)
    snippet_text = snippet
    ok_assignments: List[Dict[str, Any]] = []
    for a in obj.get("assignments", []):
        ev = a.get("evidence", {}) or {}
        quote = str(ev.get("quote", "")).strip()
        if not quote:
            continue
        if quote not in snippet_text:
            # dropp hvis quote ikke er eksakt
            continue
        ok_assignments.append(a)

    obj["assignments"] = ok_assignments
    return obj


def load_names_registry(s3, cfg: Config) -> Dict[str, Dict[str, Any]]:
    if not s3_exists(s3, cfg.s3_bucket, cfg.s3_names_registry_key):
        return {}
    tmp = cfg.local_temp_dir / "names.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(cfg.s3_bucket, cfg.s3_names_registry_key, str(tmp))

    registry: Dict[str, Dict[str, Any]] = {}
    for line in tmp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gid = r["global_speaker_id"]
        registry[gid] = r
    return registry


def save_names_registry(s3, cfg: Config, registry: Dict[str, Dict[str, Any]]) -> None:
    tmp = cfg.local_temp_dir / "names.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        for gid in sorted(registry.keys()):
            json.dump(registry[gid], f, ensure_ascii=False)
            f.write("\n")
    s3.upload_file(str(tmp), cfg.s3_bucket, cfg.s3_names_registry_key)


def update_registry_with_episode(registry: Dict[str, Dict[str, Any]], episode_result: Dict[str, Any]) -> None:
    ep = str(episode_result.get("episode", ""))
    for a in episode_result.get("assignments", []):
        gid = str(a.get("global_speaker_id", "")).strip()
        name = str(a.get("name", "")).strip()
        conf = float(a.get("confidence", 0.0))
        ev = a.get("evidence", {}) or {}

        if not gid or not name:
            continue

        rec = registry.get(gid) or {"global_speaker_id": gid, "candidates": [], "canonical_name": None, "canonical_confidence": 0.0}
        rec["candidates"].append(
            {
                "episode": ep,
                "name": name,
                "confidence": conf,
                "evidence": ev,
            }
        )

        # oppdater canonical hvis høyere confidence
        if conf >= float(rec.get("canonical_confidence", 0.0)):
            rec["canonical_name"] = name
            rec["canonical_confidence"] = conf

        registry[gid] = rec


def process_episode_file(s3, cfg: Config, key: str, names_registry: Dict[str, Dict[str, Any]]) -> None:
    relative = key.replace(cfg.s3_processed_global_prefix, "", 1)  # podcast/episode.jsonl
    cache_key = f"{cfg.s3_global_prefix}name_cache/{relative[:-5]}.json"  # drop .jsonl

    if s3_exists(s3, cfg.s3_bucket, cache_key):
        print(f"SKIP (name cache finnes): {relative}")
        return

    obj = s3.get_object(Bucket=cfg.s3_bucket, Key=key)
    text = obj["Body"].read().decode("utf-8", errors="replace")
    rows = read_jsonl_text(text)
    if not rows:
        print(f"ADVARSEL: tom {relative}")
        return

    audio_file = str(rows[0].get("audio_file", ""))
    res = resolve_names_for_episode(cfg, audio_file, rows)

    # lagre cache
    tmp = cfg.local_temp_dir / "cache" / f"{relative[:-5]}.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    s3.upload_file(str(tmp), cfg.s3_bucket, cache_key)

    # oppdater global names registry
    update_registry_with_episode(names_registry, res)
    print(f"OK: {relative} (assignments={len(res.get('assignments', []))})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve human names for global speakers using LLM + evidence.")
    parser.add_argument("--max-files", type=int, default=0, help="Begrens antall episoder (0=ingen grense).")
    args = parser.parse_args()

    cfg = load_config()
    s3 = get_s3_client(cfg)
    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)

    keys = list_s3_keys(s3, cfg.s3_bucket, cfg.s3_processed_global_prefix, suffix=".jsonl")
    keys.sort()

    if args.max_files and args.max_files > 0:
        keys = keys[: args.max_files]

    names_registry = load_names_registry(s3, cfg)

    for i, k in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] {k}")
        process_episode_file(s3, cfg, k, names_registry)
        # lagre ofte
        save_names_registry(s3, cfg, names_registry)

    print("Ferdig.")


if __name__ == "__main__":
    main()
