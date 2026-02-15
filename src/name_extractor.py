from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
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
    Robust ekstraksjon av JSON fra LLM-svar.
    Håndterer markdown, tekst før/etter, og lister/objekter.
    """
    # 1. Fjern markdown code blocks
    s = s.replace("```json", "").replace("```", "")
    
    # 2. Finn første { eller [
    start_obj = s.find("{")
    start_list = s.find("[")
    
    # Hvis ingen finnes
    if start_obj == -1 and start_list == -1:
        raise ValueError("Fant ikke JSON-struktur (verken { eller [) i svar.")
    
    # 3. Bestem hva som kommer først
    is_list = False
    if start_obj == -1:
        start = start_list
        is_list = True
    elif start_list == -1:
        start = start_obj
    else:
        # Begge finnes, ta den første
        if start_list < start_obj:
            start = start_list
            is_list = True
        else:
            start = start_obj

    # 4. Finn matchende slutt-tegn
    end_char = "]" if is_list else "}"
    end = s.rfind(end_char)
    
    if end == -1 or end <= start:
        raise ValueError(f"Fant start '{s[start]}', men ingen slutt '{end_char}'.")
        
    return s[start : end + 1]


def normalize_line_id(line_id: str) -> str:
    """Gjør om 'L1', 'L01' -> 'L0001' for sikker matching."""
    line_id = line_id.strip()
    # Matcher L fulgt av tall
    m = re.fullmatch(r"[lL]?(\d{1,4})", line_id)
    if m:
        num = int(m.group(1))
        return f"L{num:04d}"
    return line_id


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def is_plausible_person_name(name: str) -> bool:
    n = name.strip()
    if not n: return False
    # Fjern tegn som ikke hører hjemme i navn
    n2 = re.sub(r"[^a-zA-ZÆØÅæøå\-\s']", " ", n)
    n2 = re.sub(r"\s+", " ", n2).strip()
    if not n2: return False

    tokens = [t for t in n2.split(" ") if len(t) >= 2]
    # Minst to navn (Fornavn Etternavn) ELLER ett langt navn (minst 4 tegn)
    if len(tokens) >= 2: return True
    if len(tokens) == 1 and len(tokens[0]) >= 4: return True
    return False


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

    dominant_episode_frac: float
    after_intro_window_s: float

    local_temp_dir: Path

    @property
    def s3_processed_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.processed_global_prefix}/"

    @property
    def s3_global_prefix(self) -> str:
        return f"{self.s3_base_path}/{self.global_prefix}/"


def load_config() -> Config:
    load_dotenv()
    s3_bucket = env_str("S3_BUCKET")
    if not s3_bucket: raise ValueError("Mangler S3_BUCKET")

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
        temperature=env_float("NAME_EXTRACTOR_TEMPERATURE", 0.0),
        max_episode_lines=env_int("NAME_EXTRACTOR_MAX_EPISODE_LINES", 240),
        max_speaker_lines=env_int("NAME_EXTRACTOR_MAX_SPEAKER_LINES", 40),
        dominant_episode_frac=env_float("NAME_DOMINANT_EPISODE_FRAC", 0.35),
        after_intro_window_s=env_float("NAME_AFTER_INTRO_WINDOW_S", 180.0),
        local_temp_dir=Path(env_str("LOCAL_TEMP_DIR", "temp_name_extractor")),
    )


def get_s3_client(cfg: Config):
    kwargs: Dict[str, Any] = {}
    if cfg.s3_endpoint: kwargs["endpoint_url"] = cfg.s3_endpoint
    if cfg.aws_access_key and cfg.aws_secret_key:
        kwargs["aws_access_key_id"] = cfg.aws_access_key
        kwargs["aws_secret_access_key"] = cfg.aws_secret_key
    return boto3.client("s3", **kwargs)


# -----------------------------
# LLM client (Robust Retry)
# -----------------------------
def llm_chat(cfg: Config, messages: List[Dict[str, str]]) -> str:
    if not cfg.llm_base_url or not cfg.llm_api_key:
        raise ValueError("LLM_BASE_URL/LLM_API_KEY mangler.")

    url = cfg.llm_base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.llm_api_key}"}
    payload = {
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": cfg.temperature,
    }

    # Retry logic (viktig for prod)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # Hvis vi får 429 (Rate Limit) eller 5xx (Server Error), vent og prøv igjen
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt # 1s, 2s, 4s...
                    print(f"⚠️ LLM Error {resp.status_code}, prøver igjen om {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
            
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Network error: {e}, prøver igjen...")
                time.sleep(2)
                continue
            raise e # Gi opp etter siste forsøk

    raise Exception("Max retries exceeded")


# -----------------------------
# Prompt building
# -----------------------------
INTRO_PAT = re.compile(
    r"\b(velkommen|med oss|gjest|i studio|i dag har vi|du hører på|mitt navn|jeg heter)\b",
    re.IGNORECASE,
)


def episode_speaker_durations(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for r in rows:
        gid = r.get("global_speaker_id")
        if not gid: continue
        try:
            st = float(r.get("start", 0.0))
            en = float(r.get("end", 0.0))
            dur = max(0.0, en - st)
        except Exception: continue
        d[str(gid)] = d.get(str(gid), 0.0) + dur
    return d


def dominant_speaker(rows: List[Dict[str, Any]], dominant_frac: float) -> Tuple[Optional[str], float]:
    d = episode_speaker_durations(rows)
    if not d: return None, 0.0
    total = sum(d.values())
    if total <= 1e-6: return None, 0.0
    gid = max(d, key=lambda k: d[k])
    frac = d[gid] / total
    if frac >= dominant_frac:
        return gid, frac
    return None, frac


def build_lines_with_ids(rows: List[Dict[str, Any]], cfg: Config) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    # 1) Startlinjer
    first = rows[: cfg.max_episode_lines]

    # 2) Intro-jakt med kontekst
    intro: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        txt = str(r.get("text", "")).strip()
        if txt and INTRO_PAT.search(txt):
            intro.append(r)
            # Ta med de 2 neste for kontekst (svar)
            if i + 1 < len(rows): intro.append(rows[i+1])
            if i + 2 < len(rows): intro.append(rows[i+2])
        
        if len(intro) >= cfg.max_episode_lines:
            break

    # 3) Per speaker
    per: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        gid = r.get("global_speaker_id")
        if not gid: continue
        gid = str(gid)
        per.setdefault(gid, [])
        if len(per[gid]) < cfg.max_speaker_lines:
            per[gid].append(r)

    merged: List[Dict[str, Any]] = []
    seen = set()
    def add_many(items):
        for r in items:
            k = (r.get("start"), r.get("text")) # Dedup key
            if k in seen: continue
            seen.add(k)
            merged.append(r)

    add_many(first)
    add_many(intro)
    for gid, items in per.items():
        sorted_items = sorted(items, key=lambda x: float(x.get("start", 0.0)))
        add_many(sorted_items[: cfg.max_speaker_lines])

    merged.sort(key=lambda x: float(x.get("start", 0.0)))

    line_index: Dict[str, Dict[str, Any]] = {}
    out_lines: List[str] = []
    for i, r in enumerate(merged, start=1):
        lid = f"L{i:04d}"
        gid = r.get("global_speaker_id") or "NO_GID"
        st = float(r.get("start", 0.0))
        en = float(r.get("end", 0.0))
        txt = str(r.get("text", "")).strip()
        out_lines.append(f"{lid}|[{st:.2f}-{en:.2f}]|{gid}|{txt}")
        line_index[lid] = {"gid": str(gid), "start": st, "end": en, "text": txt}

    snippet = "\n".join(out_lines)
    return snippet, line_index


# -----------------------------
# Validation
# -----------------------------
def speaker_speaks_soon_after(rows: List[Dict[str, Any]], gid: str, t: float, window_s: float) -> bool:
    t1 = t + window_s
    for r in rows:
        if str(r.get("global_speaker_id")) != gid: continue
        try: st = float(r.get("start", 0.0))
        except: continue
        if st >= t and st <= t1: return True
    return False


def validate_assignment(
    rows: List[Dict[str, Any]],
    line_index: Dict[str, Dict[str, Any]],
    dominant_gid: Optional[str],
    cfg: Config,
    a: Dict[str, Any],
) -> Tuple[bool, str]:
    gid = str(a.get("global_speaker_id", "")).strip()
    name = str(a.get("name", "")).strip()
    ev = a.get("evidence", {}) or {}
    
    # 1. Normaliser line_id for å unngå mismatch (L1 vs L0001)
    raw_line_id = str(ev.get("line_id", "")).strip()
    line_id = normalize_line_id(raw_line_id)
    ev_type = str(ev.get("type", "")).strip().lower()

    if not gid or not name or not line_id:
        return False, "missing_fields"

    if not is_plausible_person_name(name):
        return False, "name_not_plausible"

    li = line_index.get(line_id)
    if not li:
        return False, f"line_id_not_found_raw_{raw_line_id}"

    line_gid = str(li["gid"])
    line_text = str(li["text"])

    quote = str(ev.get("quote", "")).strip()
    if not quote or quote not in line_text:
        return False, "quote_not_in_line_text"

    if name.lower() not in line_text.lower():
        return False, "name_not_in_evidence_line"

    if ev_type not in {"self_intro", "host_intro"}:
        return False, "bad_evidence_type"

    # --- Regler ---
    
    # Self Intro
    if ev_type == "self_intro":
        if gid != line_gid:
            return False, "self_intro_gid_mismatch"
        if not re.search(r"\b(jeg heter|mitt navn)\b", line_text, re.IGNORECASE):
            return False, "self_intro_missing_phrase"
        return True, "ok"

    # Host Intro
    if gid == line_gid:
        return False, "host_intro_same_speaker"

    # OBS: Vi har fjernet "host_intro_targets_dominant" regelen her!
    # Gjesten KAN være dominant speaker i et intervju.
    # Vi stoler på "host_intro_same_speaker" og "speaker_speaks_soon_after" i stedet.

    t_after = float(li["end"])
    if not speaker_speaks_soon_after(rows, gid, t_after, cfg.after_intro_window_s):
        return False, "subject_not_speaking_soon_after_intro"

    return True, "ok"


# -----------------------------
# LLM Logic
# -----------------------------
def resolve_names_for_episode(cfg: Config, audio_file: str, snippet: str) -> Dict[str, Any]:
    # Generisk, robust prompt som tvinger logisk tenkning
    system = (
        "Du analyserer transkripsjoner for å finne personnavn.\n"
        "Bruk logisk deduksjon. Koble introduksjoner til riktig Speaker ID.\n\n"
        
        "VIKTIGSTE REGEL:\n"
        "Hvis Speaker A sier 'Velkommen, [NAVN]', så er det NESTE NYE speaker (Speaker B) som heter [NAVN].\n"
        "Aldri gi navnet til Speaker A (den som introduserer).\n\n"

        "EKSEMPLER (Formatet er L0000):\n"
        "Input:\n"
        "L0001|spk_A|Hei og velkommen, Per Hansen.\n"
        "L0002|spk_A|Hyggelig å se deg.\n"
        "L0003|spk_B|Takk for det.\n"
        "Output JSON:\n"
        "{\n"
        '  "assignments": [\n'
        '    { "global_speaker_id": "spk_B", "name": "Per Hansen", "confidence": 0.95, "evidence": { "type": "host_intro", "line_id": "L0001", "quote": "velkommen, Per Hansen" } }\n'
        '  ]\n'
        "}\n\n"
        
        "Svar KUN med gyldig JSON."
    )

    user = (
        f"EPISODE: {audio_file}\n\n"
        "TRANSKRIPSJON:\n"
        f"{snippet}\n\n"
        "Finn assignments nå."
    )

    try:
        content = llm_chat(cfg, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        
        # Robust parsing (håndterer både objekt og liste)
        json_str = extract_first_json_object(content)
        parsed = json.loads(json_str)
        
        # Normaliser strukturen
        if isinstance(parsed, list):
            obj = {"assignments": parsed, "unknown": []}
        else:
            obj = parsed

    except Exception as e:
        print(f"⚠️  LLM Error i {audio_file}: {e}")
        return {"assignments": [], "unknown": []}

    if "assignments" not in obj: obj["assignments"] = []
    if "unknown" not in obj: obj["unknown"] = []
    return obj


def process_one_episode(s3, cfg: Config, key: str) -> None:
    relative = key.replace(cfg.s3_processed_global_prefix, "", 1)
    cache_key = f"{cfg.s3_global_prefix}name_cache/{relative[:-5]}.json"

    if s3_exists(s3, cfg.s3_bucket, cache_key):
        print(f"SKIP (cache finnes): {relative}")
        return

    obj = s3.get_object(Bucket=cfg.s3_bucket, Key=key)
    text = obj["Body"].read().decode("utf-8", errors="replace")
    rows = read_jsonl_text(text)
    if not rows:
        print(f"ADVARSEL: tom {relative}")
        return

    audio_file = str(rows[0].get("audio_file", "")).strip()
    snippet, line_index = build_lines_with_ids(rows, cfg)
    dom_gid, dom_frac = dominant_speaker(rows, cfg.dominant_episode_frac)
    llm_res = resolve_names_for_episode(cfg, audio_file, snippet)

    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for a in llm_res.get("assignments", []):
        ok, reason = validate_assignment(rows, line_index, dom_gid, cfg, a)
        conf = clamp01(float(a.get("confidence", 0.0)))
        a["confidence"] = conf
        a["_validation"] = {"ok": ok, "reason": reason}
        if ok: valid.append(a)
        else: rejected.append(a)

    out = {
        "episode_key": key,
        "audio_file": audio_file,
        "dominant_gid": dom_gid,
        "valid_assignments": valid,
        "rejected_assignments": rejected,
    }

    tmp = cfg.local_temp_dir / "name_cache" / f"{relative[:-5]}.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    s3.upload_file(str(tmp), cfg.s3_bucket, cache_key)

    # Logging av rejects for debugging
    log_msg = f"OK: {relative} valid={len(valid)} rejected={len(rejected)} dom={dom_gid}({dom_frac:.2f})"
    if rejected:
        reasons = Counter([r["_validation"]["reason"] for r in rejected])
        top_reason = reasons.most_common(1)[0]
        log_msg += f" | Top Reject: {top_reason[0]} ({top_reason[1]}x)"
    print(log_msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ekstraher navn til S3.")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    s3 = get_s3_client(cfg)
    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)

    keys = list_s3_keys(s3, cfg.s3_bucket, cfg.s3_processed_global_prefix, suffix=".jsonl")
    keys.sort()

    if args.max_files > 0: keys = keys[: args.max_files]

    for i, k in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] {k}")
        process_one_episode(s3, cfg, k)

    print("Ferdig.")


if __name__ == "__main__":
    main()
