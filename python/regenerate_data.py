#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _as_set_csv(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    parts = [p.strip().lower() for p in value.split(",")]
    return {p for p in parts if p}


@contextmanager
def _chdir(path: Path) -> Iterable[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


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

    raise FileNotFoundError(
        "Could not resolve credentials file. Provide --credentials-file or set "
        "GOOGLE_APPLICATION_CREDENTIALS, or place a single *.json in .keys/."
    )


def _project_id_from_creds(creds_file: Path) -> str:
    creds = _load_json(creds_file)
    pid = creds.get("project_id")
    if not pid:
        raise ValueError(f"Missing project_id in credentials: {creds_file}")
    return str(pid)


def _default_bucket_from_config_or_creds(repo_root: Path, creds_file: Path) -> str:
    cfg = _resolve_event_gather_config(repo_root)
    if cfg and cfg.get("gcs_bucket_name"):
        return str(cfg["gcs_bucket_name"]).replace("gs://", "")
    return f"{_project_id_from_creds(creds_file)}.appspot.com"


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """
    Return (bucket, object_name) from one of:
    - gs://bucket/object
    - https://storage.googleapis.com/bucket/object
    - https://firebasestorage.googleapis.com/v0/b/<bucket>/o/<urlencoded object>
    """
    if uri.startswith("gs://"):
        rest = uri[len("gs://") :]
        bucket, obj = rest.split("/", 1)
        return bucket, obj

    if uri.startswith("https://storage.googleapis.com/"):
        rest = uri[len("https://storage.googleapis.com/") :]
        bucket, obj = rest.split("/", 1)
        return bucket, obj

    if uri.startswith("https://firebasestorage.googleapis.com/"):
        # Example:
        # https://firebasestorage.googleapis.com/v0/b/<bucket>/o/<obj>?alt=media...
        # We only need bucket + decoded obj path.
        from urllib.parse import unquote, urlparse

        parsed = urlparse(uri)
        parts = parsed.path.split("/")
        # ["", "v0", "b", "<bucket>", "o", "<obj>"]
        try:
            b_idx = parts.index("b")
            bucket = parts[b_idx + 1]
            o_idx = parts.index("o")
            obj_enc = "/".join(parts[o_idx + 1 :])
            obj = unquote(obj_enc)
            return bucket, obj
        except Exception as e:
            raise ValueError(f"Unrecognized Firebase Storage URL: {uri}") from e

    raise ValueError(f"Unrecognized GCS URI format: {uri}")


def _safe_label(ref: Any) -> str:
    try:
        return str(ref.path)
    except Exception:
        return str(ref)


def _ensure_deps() -> None:
    try:
        import google.cloud.firestore  # noqa: F401
        import gcsfs  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install the project python package:\n"
            "  cd python && pip install .\n"
            "Or ensure google-cloud-firestore and gcsfs are installed."
        ) from e


def _patch_faster_whisper_language_en() -> None:
    # Match the GitHub Actions workflows: force language to English.
    from functools import wraps

    from faster_whisper import WhisperModel as FasterWhisperModel

    original_transcribe = FasterWhisperModel.transcribe

    @wraps(original_transcribe)
    def transcribe_with_en(self, *args, **kwargs):
        kwargs.setdefault("language", "en")
        return original_transcribe(self, *args, **kwargs)

    FasterWhisperModel.transcribe = transcribe_with_en


@dataclass(frozen=True)
class Collections:
    event: str
    session: str
    transcript: str
    file: str
    indexed_event_gram: str


def _normalize_collection_id(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _detect_collections(db: Any, overrides: Dict[str, Optional[str]]) -> Collections:
    # Firestore: list top-level collections. Pick best matches by normalization.
    ids = [c.id for c in db.collections()]
    norm_to_id: Dict[str, str] = {_normalize_collection_id(i): i for i in ids}

    def pick(key: str, candidates: Sequence[str]) -> str:
        if overrides.get(key):
            return str(overrides[key])
        for cand in candidates:
            n = _normalize_collection_id(cand)
            if n in norm_to_id:
                return norm_to_id[n]
        raise KeyError(
            f"Could not find Firestore collection for {key}. "
            f"Available collections: {ids}"
        )

    return Collections(
        event=pick("event", ["event", "events", "Event"]),
        session=pick("session", ["session", "sessions", "Session"]),
        transcript=pick("transcript", ["transcript", "transcripts", "Transcript"]),
        file=pick("file", ["file", "files", "File"]),
        indexed_event_gram=pick(
            "indexed_event_gram",
            ["indexed_event_gram", "indexedeventgram", "indexed_event_grams", "IndexedEventGram"],
        ),
    )


def _iter_query_docs(query: Any, page_size: int = 500) -> Iterable[Any]:
    # Firestore query streams; we keep this helper in case we need paging later.
    yield from query.stream()


def _gcs_delete(
    *,
    fs: Any,
    bucket: str,
    obj: str,
    dry_run: bool,
) -> bool:
    path = f"{bucket}/{obj}"
    try:
        exists = bool(fs.exists(path))
    except Exception:
        exists = True  # treat as existing; we'll try delete and surface errors
    if not exists:
        return False
    if dry_run:
        print(f"[dry-run] delete gs://{bucket}/{obj}")
        return True
    fs.rm(path)
    print(f"deleted gs://{bucket}/{obj}")
    return True


def _delete_indexed_event_grams(
    *,
    db: Any,
    cols: Collections,
    event_ref: Any,
    dry_run: bool,
) -> int:
    grams_ref = db.collection(cols.indexed_event_gram)
    q = grams_ref.where("event_ref", "==", event_ref)
    n = 0
    batch = db.batch()
    batch_ops = 0
    for doc in _iter_query_docs(q):
        n += 1
        if dry_run:
            continue
        batch.delete(doc.reference)
        batch_ops += 1
        if batch_ops >= 450:
            batch.commit()
            batch = db.batch()
            batch_ops = 0
    if not dry_run and batch_ops:
        batch.commit()
    if dry_run:
        print(f"[dry-run] would delete {n} indexed_event_gram docs for {_safe_label(event_ref)}")
    else:
        print(f"deleted {n} indexed_event_gram docs for {_safe_label(event_ref)}")
    return n


def _fetch_sessions_for_event(db: Any, cols: Collections, event_ref: Any) -> List[Any]:
    sessions_ref = db.collection(cols.session)
    q = sessions_ref.where("event_ref", "==", event_ref).order_by("session_index")
    return list(_iter_query_docs(q))


def _fetch_transcripts_for_session(db: Any, cols: Collections, session_ref: Any) -> List[Any]:
    transcripts_ref = db.collection(cols.transcript)
    q = transcripts_ref.where("session_ref", "==", session_ref)
    return list(_iter_query_docs(q))


def _fetch_file_doc(db: Any, cols: Collections, file_ref: Any) -> Any:
    if file_ref is None:
        return None
    return file_ref.get()


def _upload_overwrite(
    *,
    creds_file: Path,
    bucket: str,
    local_path: Path,
    remote_object_name: str,
    dry_run: bool,
) -> str:
    from cdp_backend.file_store import functions as fs_functions

    if dry_run:
        print(f"[dry-run] upload {local_path} -> gs://{bucket}/{remote_object_name} (overwrite)")
        return f"gs://{bucket}/{remote_object_name}"

    uri = fs_functions.upload_file(
        credentials_file=str(creds_file),
        bucket=bucket,
        filepath=str(local_path),
        save_name=remote_object_name,
        remove_local=False,
        overwrite=True,
    )
    print(f"uploaded -> {uri} (overwrite)")
    return uri


def _regenerate_transcripts_for_sessions(
    *,
    repo_root: Path,
    creds_file: Path,
    bucket: str,
    db: Any,
    cols: Collections,
    sessions: Sequence[Any],
    dry_run: bool,
    whisper_model_name: str,
    whisper_model_confidence: Optional[float],
) -> None:
    # Heavy path. Only do this if user explicitly requested regeneration.
    from cdp_backend.sr_models.whisper import WhisperModel
    from cdp_backend.utils import file_utils as cdp_file_utils

    _patch_faster_whisper_language_en()

    model = WhisperModel(model_name=whisper_model_name, confidence=whisper_model_confidence)

    for s in sessions:
        s_ref = s.reference
        s_data = s.to_dict() or {}
        video_url = s_data.get("video_uri")
        if not video_url:
            _eprint(f"Skipping session {_safe_label(s_ref)}: missing video_uri")
            continue

        # Find transcript docs (if any) so we can overwrite the exact object name(s).
        transcript_docs = _fetch_transcripts_for_session(db, cols, s_ref)
        file_targets: List[Tuple[Any, str, str]] = []
        for tdoc in transcript_docs:
            tdata = tdoc.to_dict() or {}
            fref = tdata.get("file_ref")
            if not fref:
                continue
            fdoc = _fetch_file_doc(db, cols, fref)
            if not fdoc or not fdoc.exists:
                continue
            furi = (fdoc.to_dict() or {}).get("uri")
            if not furi:
                continue
            b, obj = _parse_gcs_uri(str(furi))
            file_targets.append((tdoc, b, obj))

        if not file_targets:
            raise RuntimeError(
                f"No existing transcript file targets found for session {_safe_label(s_ref)}. "
                "This script currently requires an existing transcript to overwrite."
            )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Download hosted video
            local_video = td_path / "session_video"
            try:
                local_video_path = Path(
                    cdp_file_utils.resource_copy(
                        uri=str(video_url),
                        dst=local_video,
                        copy_suffix=True,
                        overwrite=True,
                    )
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download session video: {video_url}") from e

            # Split audio
            local_audio = td_path / "audio.wav"
            try:
                cdp_file_utils.split_audio(
                    video_read_path=str(local_video_path),
                    audio_save_path=str(local_audio),
                    overwrite=True,
                )
            except Exception as e:
                raise RuntimeError("Failed to split audio. Is ffmpeg installed?") from e

            # Transcribe
            transcript = model.transcribe(file_uri=str(local_audio))
            # Attach session datetime if present
            sdt = s_data.get("session_datetime")
            if sdt is not None:
                # Firestore returns datetime objects; transcript expects ISO string.
                if isinstance(sdt, datetime):
                    transcript.session_datetime = sdt.isoformat()
                else:
                    transcript.session_datetime = str(sdt)

            # Write transcript JSON once, then upload to each target object.
            local_json = td_path / "transcript.json"
            local_json.write_text(transcript.to_json(), encoding="utf-8")

            for tdoc, b, obj in file_targets:
                if b != bucket:
                    # This shouldn't happen, but we won't silently upload to a different bucket.
                    raise RuntimeError(
                        f"Transcript target bucket mismatch for session {_safe_label(s_ref)}: "
                        f"expected {bucket}, found {b} (uri object {obj})"
                    )

                _upload_overwrite(
                    creds_file=creds_file,
                    bucket=bucket,
                    local_path=local_json,
                    remote_object_name=obj,
                    dry_run=dry_run,
                )

                # Update transcript metadata doc to match the regenerated transcript.
                update = {
                    "generator": transcript.generator,
                    "confidence": float(transcript.confidence),
                    "created": datetime.fromisoformat(transcript.created_datetime),
                }
                if dry_run:
                    print(f"[dry-run] update transcript doc {_safe_label(tdoc.reference)} metadata")
                else:
                    tdoc.reference.update(update)


def _regenerate_event_thumbnails(
    *,
    repo_root: Path,
    creds_file: Path,
    bucket: str,
    db: Any,
    cols: Collections,
    event_doc: Any,
    sessions: Sequence[Any],
    dry_run: bool,
) -> None:
    # Reuse the same patch used in pipelines.
    from cdp_patches import patch_thumbnails

    from cdp_backend.utils import file_utils as cdp_file_utils

    patch_thumbnails(num_frames=10, gif_duration_seconds=10.0)

    event_data = event_doc.to_dict() or {}
    static_ref = event_data.get("static_thumbnail_ref")
    hover_ref = event_data.get("hover_thumbnail_ref")
    if not static_ref or not hover_ref:
        raise RuntimeError(
            f"Event {_safe_label(event_doc.reference)} is missing thumbnail refs; "
            "cannot regenerate in-place without creating new File docs."
        )

    static_file_doc = _fetch_file_doc(db, cols, static_ref)
    hover_file_doc = _fetch_file_doc(db, cols, hover_ref)
    if not static_file_doc or not static_file_doc.exists or not hover_file_doc or not hover_file_doc.exists:
        raise RuntimeError("Thumbnail File docs not found; cannot regenerate.")

    static_uri = (static_file_doc.to_dict() or {}).get("uri")
    hover_uri = (hover_file_doc.to_dict() or {}).get("uri")
    if not static_uri or not hover_uri:
        raise RuntimeError("Thumbnail File docs missing uri field; cannot regenerate.")

    static_bucket, static_obj = _parse_gcs_uri(str(static_uri))
    hover_bucket, hover_obj = _parse_gcs_uri(str(hover_uri))
    if static_bucket != bucket or hover_bucket != bucket:
        raise RuntimeError(
            f"Thumbnail bucket mismatch: expected {bucket}, found {static_bucket} / {hover_bucket}"
        )

    # Choose a session video for thumbnail generation.
    # Prefer a session whose content hash matches the thumbnail filename prefix.
    def prefix_hash(obj_name: str) -> Optional[str]:
        base = obj_name.split("/")[-1]
        if "-static-thumbnail" in base:
            return base.split("-static-thumbnail", 1)[0]
        if "-hover-thumbnail" in base:
            return base.split("-hover-thumbnail", 1)[0]
        return None

    target_hash = prefix_hash(static_obj) or prefix_hash(hover_obj)

    chosen = None
    for s in sessions:
        h = (s.to_dict() or {}).get("session_content_hash")
        if target_hash and h == target_hash:
            chosen = s
            break
    if chosen is None and sessions:
        chosen = sessions[0]

    if chosen is None:
        raise RuntimeError(f"No sessions found for event {_safe_label(event_doc.reference)}")

    s_data = chosen.to_dict() or {}
    video_url = s_data.get("video_uri")
    sess_hash = s_data.get("session_content_hash") or target_hash
    if not video_url or not sess_hash:
        raise RuntimeError("Chosen session missing video_uri or session_content_hash")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with _chdir(td_path):
            local_video = td_path / "session_video"
            try:
                local_video_path = Path(
                    cdp_file_utils.resource_copy(
                        uri=str(video_url),
                        dst=local_video,
                        copy_suffix=True,
                        overwrite=True,
                    )
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download session video: {video_url}") from e

            static_local = Path(cdp_file_utils.get_static_thumbnail(str(local_video_path), str(sess_hash)))
            hover_local = Path(cdp_file_utils.get_hover_thumbnail(str(local_video_path), str(sess_hash)))

            _upload_overwrite(
                creds_file=creds_file,
                bucket=bucket,
                local_path=static_local,
                remote_object_name=static_obj,
                dry_run=dry_run,
            )
            _upload_overwrite(
                creds_file=creds_file,
                bucket=bucket,
                local_path=hover_local,
                remote_object_name=hover_obj,
                dry_run=dry_run,
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(
        description=(
            "Delete and/or regenerate CDP data artifacts (transcripts, thumbnails, index grams) "
            "for a specific Event ID (or all). Dry-run by default."
        )
    )
    p.add_argument("--event-id", action="append", dest="event_ids", help="Event document ID (repeatable)")
    p.add_argument("--all", action="store_true", help="Target all events (requires --apply)")

    p.add_argument(
        "--delete",
        default="",
        help="Comma-separated: transcripts,thumbnails,index",
    )
    p.add_argument(
        "--regenerate",
        default="",
        help="Comma-separated: transcripts,thumbnails",
    )

    p.add_argument("--credentials-file", default=None, help="Path to Google service account JSON")
    p.add_argument("--bucket", default=None, help="Override GCS bucket (default from event-gather-config.json)")

    p.add_argument("--whisper-model", default=None, help="Override whisper model name (default from event-gather-config.json)")
    p.add_argument("--whisper-confidence", default=None, type=float, help="Override whisper confidence (float)")

    p.add_argument("--apply", action="store_true", help="Perform changes (otherwise dry-run)")

    # Collection overrides (rarely needed, but avoids guessing if schema differs).
    p.add_argument("--collection-event", default=None)
    p.add_argument("--collection-session", default=None)
    p.add_argument("--collection-transcript", default=None)
    p.add_argument("--collection-file", default=None)
    p.add_argument("--collection-indexed-event-gram", default=None)

    args = p.parse_args(argv)

    delete_set = _as_set_csv(args.delete)
    regen_set = _as_set_csv(args.regenerate)
    allowed_delete = {"transcripts", "thumbnails", "index"}
    allowed_regen = {"transcripts", "thumbnails"}
    if not delete_set.issubset(allowed_delete):
        raise SystemExit(f"--delete supports only: {sorted(allowed_delete)}")
    if not regen_set.issubset(allowed_regen):
        raise SystemExit(f"--regenerate supports only: {sorted(allowed_regen)}")

    dry_run = not bool(args.apply)
    if args.all and dry_run:
        raise SystemExit("--all is destructive; re-run with --apply (or target specific --event-id).")
    if not args.all and not args.event_ids:
        raise SystemExit("Provide --event-id (repeatable) or --all")
    if not delete_set and not regen_set:
        raise SystemExit("Nothing to do. Provide --delete and/or --regenerate")

    _ensure_deps()

    creds_file = _resolve_credentials_file(repo_root, args.credentials_file)
    project_id = _project_id_from_creds(creds_file)
    bucket = (args.bucket or _default_bucket_from_config_or_creds(repo_root, creds_file)).replace("gs://", "")

    cfg = _resolve_event_gather_config(repo_root) or {}
    whisper_model_name = args.whisper_model or str(cfg.get("whisper_model_name") or "medium")
    whisper_conf = args.whisper_confidence
    if whisper_conf is None:
        whisper_conf = cfg.get("whisper_model_confidence")
        whisper_conf = float(whisper_conf) if whisper_conf is not None else None

    print(f"Project: {project_id}")
    print(f"Bucket: {bucket}")
    print(f"Credentials: {creds_file}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}")
    print(f"Delete: {sorted(delete_set) if delete_set else '(none)'}")
    print(f"Regenerate: {sorted(regen_set) if regen_set else '(none)'}")
    if "transcripts" in regen_set:
        print(f"Whisper model: {whisper_model_name}")
        print(f"Whisper confidence: {whisper_conf}")

    from google.cloud import firestore
    from google.oauth2 import service_account
    from gcsfs import GCSFileSystem

    creds = service_account.Credentials.from_service_account_file(str(creds_file))
    db = firestore.Client(project=project_id, credentials=creds)
    fs = GCSFileSystem(token=str(creds_file))

    cols = _detect_collections(
        db,
        overrides={
            "event": args.collection_event,
            "session": args.collection_session,
            "transcript": args.collection_transcript,
            "file": args.collection_file,
            "indexed_event_gram": args.collection_indexed_event_gram,
        },
    )

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

    for ev in event_docs:
        ev_ref = ev.reference
        print("")
        print(f"== Event {ev.id} ==")

        sessions = _fetch_sessions_for_event(db, cols, ev_ref)
        print(f"Sessions: {len(sessions)}")

        if "index" in delete_set:
            _delete_indexed_event_grams(db=db, cols=cols, event_ref=ev_ref, dry_run=dry_run)

        if "thumbnails" in delete_set:
            ev_data = ev.to_dict() or {}
            for k in ["static_thumbnail_ref", "hover_thumbnail_ref"]:
                fref = ev_data.get(k)
                if not fref:
                    continue
                fdoc = _fetch_file_doc(db, cols, fref)
                if not fdoc or not fdoc.exists:
                    continue
                furi = (fdoc.to_dict() or {}).get("uri")
                if not furi:
                    continue
                b, obj = _parse_gcs_uri(str(furi))
                if b != bucket:
                    raise RuntimeError(f"{k} bucket mismatch: expected {bucket}, found {b}")
                _gcs_delete(fs=fs, bucket=b, obj=obj, dry_run=dry_run)

        if "transcripts" in delete_set:
            deleted = 0
            for s in sessions:
                tdocs = _fetch_transcripts_for_session(db, cols, s.reference)
                for tdoc in tdocs:
                    tdata = tdoc.to_dict() or {}
                    fref = tdata.get("file_ref")
                    if not fref:
                        continue
                    fdoc = _fetch_file_doc(db, cols, fref)
                    if not fdoc or not fdoc.exists:
                        continue
                    furi = (fdoc.to_dict() or {}).get("uri")
                    if not furi:
                        continue
                    b, obj = _parse_gcs_uri(str(furi))
                    if b != bucket:
                        raise RuntimeError(f"Transcript bucket mismatch: expected {bucket}, found {b}")
                    if _gcs_delete(fs=fs, bucket=b, obj=obj, dry_run=dry_run):
                        deleted += 1
            if dry_run:
                print(f"[dry-run] would delete {deleted} transcript JSON objects")
            else:
                print(f"deleted {deleted} transcript JSON objects")

        if "thumbnails" in regen_set:
            _regenerate_event_thumbnails(
                repo_root=repo_root,
                creds_file=creds_file,
                bucket=bucket,
                db=db,
                cols=cols,
                event_doc=ev,
                sessions=sessions,
                dry_run=dry_run,
            )

        if "transcripts" in regen_set:
            _regenerate_transcripts_for_sessions(
                repo_root=repo_root,
                creds_file=creds_file,
                bucket=bucket,
                db=db,
                cols=cols,
                sessions=sessions,
                dry_run=dry_run,
                whisper_model_name=whisper_model_name,
                whisper_model_confidence=whisper_conf,
            )

        if "index" in delete_set and dry_run:
            print(
                "Note: keywords/search won't update until you rerun the Event Index workflow "
                "(the index generation is global)."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
