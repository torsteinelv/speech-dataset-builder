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


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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

    # VIKTIG: sorter på tid så "neste linje"/turn-taking blir riktig
    out.sort(key=lambda r: safe_float(r.get("start", 0.0), 0.0))
    return out


def extract_json_balanced(s: str) -> str:
    """
    Robust JSON-ekstraksjon som finner første '{' eller '[' og returnerer
    den første balanserte JSON-strukturen, selv om modellen skriver tekst før/etter.
    Tåler markdown ```json ... ```.
    """
    s = s.replace("```json", "").replace("```", "").strip()

    # finn første start-tegn
    start = None
    start_char = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            start_char = ch
            break
    if start is None:
        raise ValueError("Fant ikke { eller [ i LLM-svar")

    end_char = "}" if start_char == "{" else "]"

    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
            if depth == 0:
                return s[start : j + 1]

    raise ValueError("Fant start på JSON, men ikke balansert slutt (uferdig svar?)")


def normalize_line_id(line_id: str) -> str:
    """Gjør om 'L1', 'L01', '0001' -> 'L0001' for sikker matching."""
    line_id = line_id.strip()
    m = re.fullmatch(r"[lL]?(\d{1,4})", line_id)
    if m:
        num = int(m.group(1))
        return f"L{num:04d}"
    return line_id


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


_BAD_NAMES = {
    # roller / generiske labels
    "host",
    "guest",
    "vert",
    "gjest",
    "programleder",
    "speaker",
    "deltaker",
    "deltakere",
    "participant",
    # "unknown"
    "unknown",
    "ukjent",
    # pronomen/vanlige generiske ord (hindrer kortnavn-feil)
    "jeg",
    "du",
    "vi",
    "han",
    "hun",
    "dere",
    "alle",
    "ingen",
    "noen",
    "takk",
    "hei",
    "hallo",
}


def _normalize_name_for_check(name: str) -> str:
    n = re.sub(r"[^a-zA-ZÆØÅæøå\-\s']", " ", name).lower().strip()
    n = re.sub(r"\s+", " ", n)
    return n


def is_bad_name(name: str) -> bool:
    n = _normalize_name_for_check(name)
    if not n:
        return True
    if n in _BAD_NAMES:
        return True
    # hvis det er ett token og det tokenet er "bad"
    toks = n.split()
    return len(toks) == 1 and toks[0] in _BAD_NAMES


def is_plausible_person_name(name: str) -> bool:
    """
    Mild plausibilitetssjekk:
    - Tillat norske bokstaver, bindestrek, apostrof
    - Tillat både "Fornavn Etternavn" og korte fornavn som "Per"/"Ola"
    - Blokker roller/pronomen via stoppliste
    """
    n = name.strip()
    if not n:
        return False

    if is_bad_name(n):
        return False

    n2 = re.sub(r"[^a-zA-ZÆØÅæøå\-\s']", " ", n)
    n2 = re.sub(r"\s+", " ", n2).strip()
    if not n2:
        return False

    tokens = [t for t in n2.split(" ") if len(t) >= 2]
    if len(tokens) >= 2:
        return True

    # tillat ett token >= 3 (Per/Ola/Jan)
    if len(tokens) == 1 and len(tokens[0]) >= 3:
        return True

    return False


def _loose_text(s: str) -> str:
    """
    Normaliser tekst for robuste substring-sjekker:
    - lowercase
    - fjern tegnsetting
    - kollaps whitespace
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9ÆØÅæøå\s'\-]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_loose(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    h = _loose_text(haystack)
    n = _loose_text(needle)
    if not h or not n:
        return False
    return n in h


def name_appears_in_text(name: str, text: str) -> bool:
    """
    Sikker navnesjekk:
    - For korte enkeltnavn (<=3) bruk ordgrenser for å unngå "Per" i "person"
    - Ellers bruk loose substring
    """
    name_clean = re.sub(r"\s+", " ", name.strip())
    if not name_clean:
        return False

    tokens = [t for t in re.split(r"\s+", _normalize_name_for_check(name_clean)) if t]
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        pat = r"(?<!\w)" + re.escape(tokens[0]) + r"(?!\w)"
        return re.search(pat, text, flags=re.IGNORECASE) is not None

    return contains_loose(text, name_clean)


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
    cache_version: str

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    temperature: float

    max_episode_lines: int
    max_speaker_lines: int

    dominant_episode_frac: float
    after_intro_window_s: float

    save_llm_raw: bool

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
        cache_version=env_str("NAME_CACHE_VERSION", "v2"),
        llm_base_url=env_str("LLM_BASE_URL"),
        llm_api_key=env_str("LLM_API_KEY"),
        llm_model=env_str("LLM_MODEL", "gpt-4.1-mini"),
        temperature=env_float("NAME_EXTRACTOR_TEMPERATURE", 0.0),
        max_episode_lines=env_int("NAME_EXTRACTOR_MAX_EPISODE_LINES", 240),
        max_speaker_lines=env_int("NAME_EXTRACTOR_MAX_SPEAKER_LINES", 40),
        dominant_episode_frac=env_float("NAME_DOMINANT_EPISODE_FRAC", 0.35),
        after_intro_window_s=env_float("NAME_AFTER_INTRO_WINDOW_S", 180.0),
        save_llm_raw=env_bool("NAME_SAVE_LLM_RAW", False),
        local_temp_dir=Path(env_str("LOCAL_TEMP_DIR", "temp_name_extractor")),
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
# LLM client (Robust Retry)
# -----------------------------
_SESSION = requests.Session()


def llm_chat(cfg: Config, messages: List[Dict[str, str]]) -> str:
    if not cfg.llm_base_url or not cfg.llm_api_key:
        raise ValueError("LLM_BASE_URL/LLM_API_KEY mangler.")

    url = cfg.llm_base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": cfg.temperature,
    }

    max_retries = 4
    for attempt in range(max_retries):
        try:
            resp = _SESSION.post(url, headers=headers, json=payload, timeout=120)

            # Retry på rate limit og serverfeil
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + (0.1 * attempt)  # litt jitter
                    print(f"⚠️ LLM Error {resp.status_code}, prøver igjen om {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt)
                print(f"⚠️ Network error: {e}, prøver igjen om {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue
            raise

    raise RuntimeError("Max retries exceeded (LLM)")


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
        if not gid:
            continue
        st = safe_float(r.get("start", 0.0))
        en = safe_float(r.get("end", 0.0))
        dur = max(0.0, en - st)
        d[str(gid)] = d.get(str(gid), 0.0) + dur
    return d


def dominant_speaker(rows: List[Dict[str, Any]], dominant_frac: float) -> Tuple[Optional[str], float]:
    d = episode_speaker_durations(rows)
    if not d:
        return None, 0.0
    total = sum(d.values())
    if total <= 1e-6:
        return None, 0.0
    gid = max(d, key=lambda k: d[k])
    frac = d[gid] / total
    if frac >= dominant_frac:
        return gid, frac
    return None, frac


def build_lines_with_ids(rows: List[Dict[str, Any]], cfg: Config) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Lager tekst til LLM med line-id vi kan validere hardt mot.
    Vi tar:
      1) første N linjer (start av episode)
      2) intro-linjer (regex) + 2 linjer etter for turn-taking
      3) per speaker: første M linjer
    """
    # 1) startlinjer
    first = rows[: cfg.max_episode_lines]

    # 2) intro-linjer med kontekst (2 linjer etter)
    intro: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        txt = str(r.get("text", "")).strip()
        if txt and INTRO_PAT.search(txt):
            intro.append(r)
            if i + 1 < len(rows):
                intro.append(rows[i + 1])
            if i + 2 < len(rows):
                intro.append(rows[i + 2])
        if len(intro) >= cfg.max_episode_lines:
            break

    # 3) per speaker: første M linjer
    per: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        gid = r.get("global_speaker_id")
        if not gid:
            continue
        gid = str(gid)
        per.setdefault(gid, [])
        if len(per[gid]) < cfg.max_speaker_lines:
            per[gid].append(r)

    merged: List[Dict[str, Any]] = []
    seen = set()

    def add_many(items: List[Dict[str, Any]]) -> None:
        for rr in items:
            k = (rr.get("start"), rr.get("end"), rr.get("global_speaker_id"), rr.get("text"))
            if k in seen:
                continue
            seen.add(k)
            merged.append(rr)

    add_many(first)
    add_many(intro)
    for _, items in per.items():
        items_sorted = sorted(items, key=lambda x: safe_float(x.get("start", 0.0)))
        add_many(items_sorted[: cfg.max_speaker_lines])

    merged.sort(key=lambda x: safe_float(x.get("start", 0.0)))

    line_index: Dict[str, Dict[str, Any]] = {}
    out_lines: List[str] = []

    for i, r in enumerate(merged, start=1):
        lid = f"L{i:04d}"
        gid = r.get("global_speaker_id") or "NO_GID"
        st = safe_float(r.get("start", 0.0))
        en = safe_float(r.get("end", 0.0))
        txt = str(r.get("text", "")).strip()
        out_lines.append(f"{lid}|[{st:.2f}-{en:.2f}]|{gid}|{txt}")
        line_index[lid] = {"gid": str(gid), "start": st, "end": en, "text": txt}

    return "\n".join(out_lines), line_index


# -----------------------------
# Validation
# -----------------------------
def speaker_speaks_soon_after(rows: List[Dict[str, Any]], gid: str, t: float, window_s: float) -> bool:
    t1 = t + window_s
    for r in rows:
        if str(r.get("global_speaker_id")) != gid:
            continue
        st = safe_float(r.get("start", 0.0))
        if t <= st <= t1:
            return True
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

    raw_line_id = str(ev.get("line_id", "")).strip()
    line_id = normalize_line_id(raw_line_id)
    ev_type = str(ev.get("type", "")).strip().lower()

    if not gid or not name or not line_id:
        return False, "missing_fields"

    if is_bad_name(name):
        return False, "name_is_role_not_person"

    if not is_plausible_person_name(name):
        return False, "name_not_plausible"

    li = line_index.get(line_id)
    if not li:
        return False, f"line_id_not_found_raw_{raw_line_id}"

    line_gid = str(li["gid"])
    line_text = str(li["text"])

    quote = str(ev.get("quote", "")).strip()
    # Quote-match: case-insensitiv + loose fallback
    if not quote:
        return False, "quote_missing"
    if (quote.lower() not in line_text.lower()) and (not contains_loose(line_text, quote)):
        return False, "quote_not_in_line_text"

    # Navn må faktisk finnes på evidence-linja (robust)
    if not name_appears_in_text(name, line_text):
        return False, "name_not_in_evidence_line"

    if ev_type not in {"self_intro", "host_intro"}:
        return False, "bad_evidence_type"

    # Self intro: samme speaker og må inneholde "jeg heter"/"mitt navn"
    if ev_type == "self_intro":
        if gid != line_gid:
            return False, "self_intro_gid_mismatch"
        if not re.search(r"\b(jeg heter|mitt navn)\b", line_text, re.IGNORECASE):
            return False, "self_intro_missing_phrase"
        return True, "ok"

    # host intro: må ikke tilordnes intro-speaker
    if gid == line_gid:
        return False, "host_intro_same_speaker"

    # (Valgfritt å bruke dominant_gid igjen senere; nå kun logget)
    t_after = float(li["end"])
    if not speaker_speaks_soon_after(rows, gid, t_after, cfg.after_intro_window_s):
        return False, "subject_not_speaking_soon_after_intro"

    return True, "ok"


# -----------------------------
# LLM Logic
# -----------------------------
def resolve_names_for_episode(cfg: Config, audio_file: str, snippet: str) -> Tuple[Dict[str, Any], str]:
    """
    Returnerer (obj, raw_content)
    obj er normalisert til:
      {"assignments":[...], "unknown":[...]} der assignments kan komme fra liste eller objekt.
    """
    system = (
        "Du analyserer transkripsjoner for å finne PERSONNAVN.\n"
        "Du får linjer på format:\n"
        "  L0001|[start-end]|global_speaker_id|tekst\n\n"
        "KRAV:\n"
        "- Returner KUN gyldig JSON (ingen ekstra tekst).\n"
        "- Ikke bruk roller som 'Host', 'Guest', 'Vert', 'Gjest' som navn.\n"
        "- Hver assignment MÅ ha: global_speaker_id, name, confidence, evidence{type,line_id,quote}.\n"
        "- evidence.type må være 'host_intro' eller 'self_intro'.\n"
        "- evidence.line_id må være en av Lxxxx-linjene i input.\n"
        "- evidence.quote må være en eksakt substring fra tekstfeltet på evidence.line_id (helst copy/paste).\n\n"
        "VIKTIG LOGIKK:\n"
        "Hvis Speaker A introduserer noen med navn (f.eks 'Velkommen, Per Hansen'),\n"
        "så skal navnet knyttes til personen som BLIR introdusert (ofte en annen speaker som svarer etterpå),\n"
        "ikke til speaker A.\n\n"
        "EKSEMPEL:\n"
        "Input:\n"
        "L0001|[10.00-10.50]|spk_A|Hei og velkommen, Per Hansen.\n"
        "L0002|[10.50-12.00]|spk_A|Hyggelig å se deg.\n"
        "L0003|[12.00-12.50]|spk_B|Takk for det.\n"
        "Output:\n"
        "{\n"
        '  "assignments": [\n'
        '    {\n'
        '      "global_speaker_id": "spk_B",\n'
        '      "name": "Per Hansen",\n'
        '      "confidence": 0.95,\n'
        '      "evidence": { "type": "host_intro", "line_id": "L0001", "quote": "velkommen, Per Hansen" }\n'
        "    }\n"
        "  ],\n"
        '  "unknown": []\n'
        "}\n"
    )

    user = (
        f"EPISODE: {audio_file}\n\n"
        "TRANSKRIPSJON:\n"
        f"{snippet}\n\n"
        "Finn assignments nå."
    )

    content = llm_chat(cfg, [{"role": "system", "content": system}, {"role": "user", "content": user}])

    json_str = extract_json_balanced(content)
    parsed = json.loads(json_str)

    if isinstance(parsed, list):
        obj = {"assignments": parsed, "unknown": []}
    else:
        obj = dict(parsed)

    if "assignments" not in obj:
        obj["assignments"] = []
    if "unknown" not in obj:
        obj["unknown"] = []

    return obj, content


def process_one_episode(s3, cfg: Config, key: str) -> None:
    relative = key.replace(cfg.s3_processed_global_prefix, "", 1)
    cache_key = f"{cfg.s3_global_prefix}name_cache_{cfg.cache_version}/{relative[:-5]}.json"

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

    try:
        llm_res, raw_content = resolve_names_for_episode(cfg, audio_file, snippet)
    except Exception as e:
        print(f"⚠️  LLM/parse error i {audio_file}: {e}")
        llm_res, raw_content = {"assignments": [], "unknown": []}, ""

    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for a in llm_res.get("assignments", []):
        try:
            ok, reason = validate_assignment(rows, line_index, dom_gid, cfg, a)
        except Exception as e:
            ok, reason = False, f"validation_exception_{type(e).__name__}"

        conf = clamp01(safe_float(a.get("confidence", 0.0), 0.0))
        a["confidence"] = conf
        a["_validation"] = {"ok": ok, "reason": reason}

        if ok:
            valid.append(a)
        else:
            rejected.append(a)

    out: Dict[str, Any] = {
        "episode_key": key,
        "audio_file": audio_file,
        "dominant_gid": dom_gid,
        "dominant_frac": dom_frac,
        "valid_assignments": valid,
        "rejected_assignments": rejected,
        "unknown": llm_res.get("unknown", []),
        "cache_version": cfg.cache_version,
    }

    if cfg.save_llm_raw:
        out["llm_raw"] = raw_content

    tmp = cfg.local_temp_dir / f"name_cache_{cfg.cache_version}" / f"{relative[:-5]}.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    s3.upload_file(str(tmp), cfg.s3_bucket, cache_key)

    log_msg = f"OK: {relative} valid={len(valid)} rejected={len(rejected)} dom={dom_gid}({dom_frac:.2f})"
    if rejected:
        reasons = Counter([r["_validation"]["reason"] for r in rejected if "_validation" in r])
        top = reasons.most_common(3)
        if top:
            log_msg += " | Top Rejects: " + ", ".join([f"{k}({v}x)" for k, v in top])
    print(log_msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1: Ekstraher navnebevis per episode til S3 cache.")
    parser.add_argument("--max-files", type=int, default=0, help="Begrens antall episoder (0=ingen grense).")
    args = parser.parse_args()

    cfg = load_config()
    s3 = get_s3_client(cfg)
    cfg.local_temp_dir.mkdir(parents=True, exist_ok=True)

    keys = list_s3_keys(s3, cfg.s3_bucket, cfg.s3_processed_global_prefix, suffix=".jsonl")
    keys.sort()

    if args.max_files > 0:
        keys = keys[: args.max_files]

    for i, k in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] {k}")
        process_one_episode(s3, cfg, k)

    print("Ferdig.")


if __name__ == "__main__":
    main()
