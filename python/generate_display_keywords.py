#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


PROMPT_VERSION = "v1"
DEFAULT_MODEL = "gemini-2.5-flash"
STREET_TOKENS = {
    "street",
    "st",
    "avenue",
    "ave",
    "road",
    "rd",
    "lane",
    "ln",
    "drive",
    "dr",
    "boulevard",
    "blvd",
    "north",
    "south",
    "east",
    "west",
}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_event_gather_config(repo_root: Path) -> Optional[Dict[str, Any]]:
    cfg_path = repo_root / "python" / "event-gather-config.json"
    if not cfg_path.is_file():
        return None
    try:
        return _load_json(cfg_path)
    except Exception:
        return None


def _resolve_credentials_file(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Credentials file not found: {p}")
        return p

    env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {p}"
            )
        return p

    cfg = _resolve_event_gather_config(repo_root)
    if cfg and cfg.get("google_credentials_file"):
        p = Path(str(cfg["google_credentials_file"]))
        if not p.is_absolute():
            p = (repo_root / "python" / p).resolve()
        if p.is_file():
            return p

    keys_dir = repo_root / ".keys"
    if keys_dir.is_dir():
        json_files = list(keys_dir.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0]
        if len(json_files) > 1:
            cfg = _resolve_event_gather_config(repo_root) or {}
            wanted_project_id = None
            if cfg.get("gcs_bucket_name"):
                bucket_name = str(cfg["gcs_bucket_name"]).replace("gs://", "")
                if bucket_name.endswith(".appspot.com"):
                    wanted_project_id = bucket_name[: -len(".appspot.com")]

            def _project_id_from_json(path: Path) -> Optional[str]:
                try:
                    data = _load_json(path)
                    pid = data.get("project_id")
                    return str(pid) if pid else None
                except Exception:
                    return None

            if wanted_project_id:
                matches = [
                    p for p in json_files if _project_id_from_json(p) == wanted_project_id
                ]
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    by_name = [p for p in matches if wanted_project_id in p.name]
                    if len(by_name) == 1:
                        return by_name[0]

                    non_dev = [p for p in matches if "dev" not in p.name.lower()]
                    if len(non_dev) == 1:
                        return non_dev[0]

    raise FileNotFoundError(
        "Could not resolve credentials file. Provide --credentials-file, set "
        "GOOGLE_APPLICATION_CREDENTIALS, or ensure a matching .keys/*.json is present."
    )


def _project_id_from_creds(creds_file: Path) -> str:
    creds = _load_json(creds_file)
    pid = creds.get("project_id")
    if not pid:
        raise ValueError(f"Missing project_id in credentials: {creds_file}")
    return str(pid)


@dataclass(frozen=True)
class Collections:
    event: str
    session: str
    transcript: str
    file: str
    indexed_event_gram: str


def _normalize_collection_id(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _detect_collections(db: Any) -> Collections:
    ids = [c.id for c in db.collections()]
    norm_to_id = {_normalize_collection_id(i): i for i in ids}

    def pick(candidates: Sequence[str]) -> str:
        for cand in candidates:
            n = _normalize_collection_id(cand)
            if n in norm_to_id:
                return norm_to_id[n]
        raise KeyError(f"Could not match collection from candidates={candidates}. Have={ids}")

    return Collections(
        event=pick(["event", "events", "Event"]),
        session=pick(["session", "sessions", "Session"]),
        transcript=pick(["transcript", "transcripts", "Transcript"]),
        file=pick(["file", "files", "File"]),
        indexed_event_gram=pick(
            ["indexed_event_gram", "indexedeventgram", "indexed_event_grams", "IndexedEventGram"]
        ),
    )


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if uri.startswith("gs://"):
        rest = uri[len("gs://") :]
        bucket, obj = rest.split("/", 1)
        return bucket, obj

    if uri.startswith("https://storage.googleapis.com/"):
        rest = uri[len("https://storage.googleapis.com/") :]
        bucket, obj = rest.split("/", 1)
        return bucket, obj

    if uri.startswith("https://firebasestorage.googleapis.com/"):
        parsed = urlparse.urlparse(uri)
        parts = parsed.path.split("/")
        b_idx = parts.index("b")
        bucket = parts[b_idx + 1]
        o_idx = parts.index("o")
        obj_enc = "/".join(parts[o_idx + 1 :])
        obj = urlparse.unquote(obj_enc)
        return bucket, obj

    raise ValueError(f"Unsupported storage uri format: {uri}")


def _fetch_sessions_for_event(db: Any, cols: Collections, event_ref: Any) -> List[Any]:
    q = db.collection(cols.session).where("event_ref", "==", event_ref).order_by("session_index")
    return list(q.stream())


def _fetch_transcripts_for_session(db: Any, cols: Collections, session_ref: Any) -> List[Any]:
    q = db.collection(cols.transcript).where("session_ref", "==", session_ref)
    return list(q.stream())


def _fetch_file_uri(transcript_doc: Any) -> Optional[str]:
    tdata = transcript_doc.to_dict() or {}
    fref = tdata.get("file_ref")
    if not fref:
        return None
    fdoc = fref.get()
    if not fdoc or not fdoc.exists:
        return None
    return (fdoc.to_dict() or {}).get("uri")


def _read_transcript_json(
    *, storage_client: Any, uri: str, expected_bucket: Optional[str]
) -> Optional[Dict[str, Any]]:
    try:
        bucket, obj = _parse_gcs_uri(uri)
        if expected_bucket and bucket != expected_bucket:
            _eprint(
                "Warning: transcript file bucket mismatch "
                f"(expected={expected_bucket}, got={bucket})"
            )
        blob = storage_client.bucket(bucket).blob(obj)
        raw = blob.download_as_text(encoding="utf-8")
        return json.loads(raw)
    except Exception as e:
        _eprint(f"Warning: failed to read transcript uri={uri}: {e}")
        return None


def _extract_sentences(transcript_json: Dict[str, Any]) -> List[str]:
    sentences = transcript_json.get("sentences")
    if not isinstance(sentences, list):
        return []

    texts: List[str] = []
    for s in sentences:
        if isinstance(s, dict):
            text = s.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def _top_index_grams(db: Any, cols: Collections, event_ref: Any, limit: int) -> List[Dict[str, Any]]:
    from google.cloud import firestore

    q = (
        db.collection(cols.indexed_event_gram)
        .where("event_ref", "==", event_ref)
        .order_by("value", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    grams: List[Dict[str, Any]] = []
    for doc in q.stream():
        d = doc.to_dict() or {}
        grams.append(
            {
                "unstemmed_gram": d.get("unstemmed_gram") or "",
                "stemmed_gram": d.get("stemmed_gram") or "",
                "value": float(d.get("value") or 0),
                "context_span": d.get("context_span") or "",
            }
        )
    return grams


def _normalize_keyword(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.strip("'\"`.,;:!?()[]{}")
    return text


def _is_noise_keyword(text: str) -> bool:
    if not text:
        return True

    if len(text) <= 1:
        return True

    # mostly numeric or numeric punctuation
    if re.fullmatch(r"[\d\s\-_/.,]+", text):
        return True

    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not tokens:
        return True

    # address-ish phrase
    has_digit = any(any(ch.isdigit() for ch in t) for t in tokens)
    has_street = any(t in STREET_TOKENS for t in tokens)
    if has_digit and has_street:
        return True

    # long phrases tend to be poor card keywords
    if len(tokens) > 5:
        return True

    return False


def _dedupe_keywords(candidates: Iterable[str], max_count: int) -> List[str]:
    chosen: List[str] = []
    seen = set()

    for raw in candidates:
        kw = _normalize_keyword(raw)
        if _is_noise_keyword(kw):
            continue
        key = kw.casefold()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(kw)
        if len(chosen) >= max_count:
            break

    return chosen


def _keywords_from_index_fallback(index_grams: Sequence[Dict[str, Any]], max_count: int) -> List[str]:
    raw = [str(g.get("unstemmed_gram") or "").strip() for g in index_grams]
    return _dedupe_keywords(raw, max_count=max_count)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        maybe = text[start : end + 1]
        obj = json.loads(maybe)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from model response")


def _build_prompt(
    *,
    event_id: str,
    transcript_excerpt: str,
    index_grams: Sequence[Dict[str, Any]],
    keywords_count: int,
) -> str:
    grams_block = "\n".join(
        [
            f"- gram: {g.get('unstemmed_gram')} | score: {g.get('value'):.4f}"
            for g in index_grams
            if g.get("unstemmed_gram")
        ]
    )

    return (
        "You generate display keywords for a city meeting transcript.\n"
        "Return ONLY valid JSON with this schema: {\"keywords\": [string, ...]}.\n"
        f"Return exactly {keywords_count} keywords if possible.\n"
        "Rules:\n"
        "- Prefer topic phrases and policy concepts.\n"
        "- Avoid person names unless core to meeting subject.\n"
        "- Avoid street addresses, line-item numbers, and numeric-only tokens.\n"
        "- Keep each keyword concise (1-3 words preferred).\n"
        "- Use transcript-grounded terms only.\n"
        "- Do not include explanations.\n\n"
        f"Event ID: {event_id}\n\n"
        "Top current index grams (for additional signal):\n"
        f"{grams_block if grams_block else '- (none)'}\n\n"
        "Transcript excerpt:\n"
        f"{transcript_excerpt}"
    )


def _call_gemini_json(*, api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:"
        f"generateContent?key={urlparse.quote(api_key)}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urlerror.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {e.code}: {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}") from e

    data = json.loads(body)
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")

    parts = (((candidates[0] or {}).get("content") or {}).get("parts"))
    if not isinstance(parts, list) or not parts:
        raise RuntimeError(f"Gemini response missing content parts: {data}")

    text = parts[0].get("text") if isinstance(parts[0], dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Gemini response missing text payload: {data}")

    return _extract_json_object(text)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_csv_keywords(keywords: Sequence[str]) -> str:
    return ", ".join(keywords)


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(
        description=(
            "Generate and store event display keywords using Gemini, grounded in "
            "event transcripts. Dry-run by default."
        )
    )
    p.add_argument("--event-id", action="append", dest="event_ids", help="Event document ID (repeatable)")
    p.add_argument("--all", action="store_true", help="Process all events (requires --apply)")
    p.add_argument("--changed-only", action="store_true", help="Skip events whose transcript_hash already matches metadata")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of events to process after filtering")

    p.add_argument("--credentials-file", default=None, help="Path to Google service account JSON")
    p.add_argument("--bucket", default=None, help="Optional expected transcript bucket override")

    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    p.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Env var name containing Gemini API key")
    p.add_argument("--max-transcript-chars", type=int, default=16000)
    p.add_argument("--index-gram-limit", type=int, default=40)
    p.add_argument("--keywords-count", type=int, default=5)

    p.add_argument("--apply", action="store_true", help="Persist changes (default is dry-run)")
    args = p.parse_args(argv)

    if args.all and not args.apply:
        raise SystemExit("--all is potentially destructive; rerun with --apply")
    if not args.all and not args.event_ids:
        raise SystemExit("Provide --event-id (repeatable) or --all")

    dry_run = not bool(args.apply)

    creds_file = _resolve_credentials_file(repo_root, args.credentials_file)
    project_id = _project_id_from_creds(creds_file)

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing Gemini API key env var: {args.api_key_env}")

    try:
        from google.cloud import firestore
        from google.cloud import storage
        from google.oauth2 import service_account
    except Exception as e:
        raise SystemExit(
            "Missing deps. Install google-cloud-firestore and google-cloud-storage "
            "in your Python environment."
        ) from e

    creds = service_account.Credentials.from_service_account_file(str(creds_file))
    db = firestore.Client(project=project_id, credentials=creds)
    storage_client = storage.Client(project=project_id, credentials=creds)

    cols = _detect_collections(db)

    expected_bucket = None
    if args.bucket:
        expected_bucket = str(args.bucket).replace("gs://", "")

    print(f"Project: {project_id}")
    print(f"Credentials: {creds_file}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}")
    print(f"Collections: event={cols.event}, session={cols.session}, transcript={cols.transcript}, file={cols.file}, indexed={cols.indexed_event_gram}")
    print(f"Model: {args.model}")
    print(f"Changed-only: {bool(args.changed_only)}")

    event_col = db.collection(cols.event)
    if args.all:
        event_docs = list(event_col.stream())
    else:
        event_docs = []
        for eid in args.event_ids:
            doc = event_col.document(str(eid)).get()
            if not doc.exists:
                raise RuntimeError(f"Event not found: {eid} (collection {cols.event})")
            event_docs.append(doc)

    if args.limit is not None:
        event_docs = event_docs[: max(0, int(args.limit))]

    processed = 0
    skipped_no_transcript = 0
    skipped_unchanged = 0
    failed = 0

    for ev in event_docs:
        print("")
        print(f"== Event {ev.id} ==")

        sessions = _fetch_sessions_for_event(db, cols, ev.reference)
        if not sessions:
            print("skip: no sessions")
            skipped_no_transcript += 1
            continue

        transcript_texts: List[str] = []
        transcript_source_count = 0

        for s in sessions:
            tdocs = _fetch_transcripts_for_session(db, cols, s.reference)
            if not tdocs:
                continue

            # Keep highest confidence transcript per session.
            def _confidence(doc: Any) -> float:
                return float((doc.to_dict() or {}).get("confidence") or -1)

            chosen = sorted(tdocs, key=_confidence, reverse=True)[0]
            uri = _fetch_file_uri(chosen)
            if not uri:
                continue

            transcript_json = _read_transcript_json(
                storage_client=storage_client,
                uri=str(uri),
                expected_bucket=expected_bucket,
            )
            if not transcript_json:
                continue

            sentences = _extract_sentences(transcript_json)
            if not sentences:
                continue

            transcript_texts.extend(sentences)
            transcript_source_count += 1

        if not transcript_texts:
            print("skip: no transcript text")
            skipped_no_transcript += 1
            continue

        full_text = "\n".join(transcript_texts)
        transcript_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        ev_data = ev.to_dict() or {}
        meta = ev_data.get("display_keywords_meta")
        if args.changed_only and isinstance(meta, dict):
            prior_hash = meta.get("transcript_hash")
            if prior_hash and str(prior_hash) == transcript_hash:
                print("skip: unchanged transcript hash")
                skipped_unchanged += 1
                continue

        excerpt = full_text[: max(0, int(args.max_transcript_chars))]
        index_grams = _top_index_grams(db, cols, ev.reference, limit=int(args.index_gram_limit))

        prompt = _build_prompt(
            event_id=ev.id,
            transcript_excerpt=excerpt,
            index_grams=index_grams,
            keywords_count=int(args.keywords_count),
        )

        try:
            llm_obj = _call_gemini_json(api_key=api_key, model=str(args.model), prompt=prompt)
            llm_keywords = llm_obj.get("keywords") if isinstance(llm_obj, dict) else None
            if not isinstance(llm_keywords, list):
                llm_keywords = []
            llm_keywords = [str(x) for x in llm_keywords if isinstance(x, (str, int, float))]
        except Exception as e:
            _eprint(f"Gemini failed for event {ev.id}: {e}")
            llm_keywords = []

        keywords = _dedupe_keywords(llm_keywords, max_count=int(args.keywords_count))
        source = "gemini"

        if len(keywords) < int(args.keywords_count):
            fallback = _keywords_from_index_fallback(index_grams, max_count=int(args.keywords_count))
            merged = _dedupe_keywords([*keywords, *fallback], max_count=int(args.keywords_count))
            if merged != keywords:
                source = "gemini+index_fallback" if keywords else "index_fallback"
            keywords = merged

        if not keywords:
            _eprint(f"No usable keywords generated for event {ev.id}; skipping write")
            failed += 1
            continue

        payload = {
            "display_keywords": keywords,
            "display_keywords_meta": {
                "model": str(args.model),
                "prompt_version": PROMPT_VERSION,
                "transcript_hash": transcript_hash,
                "generated_at": _utc_now_iso(),
                "source": source,
                "transcript_sources": transcript_source_count,
            },
        }

        print(f"keywords: {_as_csv_keywords(keywords)}")
        print(f"source: {source}")

        if dry_run:
            print("[dry-run] would write display_keywords + display_keywords_meta")
        else:
            ev.reference.set(payload, merge=True)
            print("wrote display_keywords + display_keywords_meta")

        processed += 1

    print("")
    print("== Summary ==")
    print(f"processed: {processed}")
    print(f"skipped_no_transcript: {skipped_no_transcript}")
    print(f"skipped_unchanged: {skipped_unchanged}")
    print(f"failed: {failed}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
