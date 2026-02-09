#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internet Archive scraper for Richmond, IN public meeting videos."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import re
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cdp_backend.pipeline import ingestion_models
from cdp_backend.pipeline.ingestion_models import EventIngestionModel

IA_ADVANCED_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_METADATA_URL = "https://archive.org/metadata/{identifier}"
IA_DETAILS_URL = "https://archive.org/details/{identifier}"
IA_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

DEFAULT_SEARCH_QUERY = 'creator:"WCTV" richmond'
DEFAULT_BODY_NAME = "Wayne County (Unknown Governing Body)"
BODY_KEYWORDS = {
    # Richmond
    "advisory plat committee": "Richmond Advisory Plat Committee",
    "board of public works": "Richmond Board of Public Works and Safety",
    "board of zoning appeals": "Richmond Board of Zoning Appeals",
    "common council": "Richmond Common Council",
    "parks board": "Richmond Parks Board",
    "planning commission": "Richmond Planning Commission",
    "redevelopment commission": "Richmond Redevelopment Commission",
    "rp&l board of directors": "Richmond Power & Light Board of Directors",
    "sanitary district board of commissioners": (
        "Richmond Sanitary District Board of Commissioners"
    ),
    "unsafe building committee": "Richmond Unsafe Building Committee",
    # Wayne County
    "county council & commissioners workshop": (
        "Wayne County Council & Commissioners Workshop"
    ),
    "wayne county advisory plan commission": "Wayne County Advisory Plan Commission",
    "wayne county board of finance": "Wayne County Board of Finance",
    "wayne county commissioners": "Wayne County Board of Commissioners",
    "wayne county council": "Wayne County Council",
    "wayne county drainage board": "Wayne County Drainage Board",
    "wayne county finance meeting": "Wayne County Finance Committee",
    "wayne county health board": "Wayne County Health Board",
    "wayne county personnel committee": "Wayne County Personnel Committee",
}

# Bot guidance from IA asks for clear identity and contact details.
# Override IA_USER_AGENT in environment to set a custom user agent.
DEFAULT_USER_AGENT = (
    "cdp-richmond-indiana/1.0 (+https://github.com/jbax1899/cdp-richmond-indiana)"
)

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.25
PAGE_SIZE = 100
ALLOW_AUDIO_AS_PRIMARY = False
EPOCH_SENTINEL = datetime(1970, 1, 1, tzinfo=timezone.utc)

VIDEO_EXTENSIONS = {
    "mp4",
    "mov",
    "m4v",
    "webm",
    "mkv",
    "avi",
    "mpg",
    "mpeg",
    "ts",
}

# Prefer web-native containers first to avoid expensive conversion/re-hosting
# in the downstream CDP gather pipeline.
VIDEO_EXTENSION_PRIORITY = {
    "mp4": 0,
    "webm": 1,
    "m4v": 2,
    "mov": 3,
}

AUDIO_EXTENSIONS = {
    "mp3",
    "m4a",
    "wav",
    "aac",
    "ogg",
    "opus",
    "flac",
}

IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "gif",
}


@dataclass(frozen=True)
class EventParts:
    """Normalized fields used to build an EventIngestionModel."""

    body_name: str
    event_name: str
    source_uri: str
    event_datetime: datetime
    primary_video_uri: str
    video_uris: List[str]


@dataclass(frozen=True)
class MediaFile:
    """Normalized representation of an IA media file."""

    name: str
    ext: str
    uri: str
    size_bytes: int
    mtime: Optional[datetime]
    source: str
    media_kind: str
    resolution: int


@dataclass
class IngestionSummary:
    """Structured counters for run-level ingest reporting."""

    total_items_fetched: int = 0
    ingested: int = 0
    skipped_by_reason: Dict[str, int] = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        return sum(self.skipped_by_reason.values())

    def increment_skip(self, reason: str) -> None:
        key = reason or "unknown"
        self.skipped_by_reason[key] = self.skipped_by_reason.get(key, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items_fetched": self.total_items_fetched,
            "ingested": self.ingested,
            "skipped": self.skipped,
            "skipped_by_reason": dict(self.skipped_by_reason),
        }

    def as_stdout_lines(self) -> List[str]:
        lines = [
            "[CDP-Richmond-Ingest Summary]",
            f"Total items fetched: {self.total_items_fetched}",
            f"Ingested: {self.ingested}",
            f"Skipped: {self.skipped}",
        ]
        for reason, count in sorted(self.skipped_by_reason.items()):
            lines.append(f" - {count} {reason.replace('_', ' ')}")
        return lines

    def emit_stdout(self) -> None:
        print("\n".join(self.as_stdout_lines()))


class IAClient:
    """Interface for IA search + metadata retrieval."""

    def search_docs(
        self,
        from_dt: datetime,
        to_dt: datetime,
    ) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_metadata(self, identifier: str) -> Dict[str, Any]:
        raise NotImplementedError


class InternetArchiveClient(IAClient):
    """Concrete IA client using archive.org HTTP endpoints."""

    def __init__(self, session: requests.Session):
        self._session = session

    def search_docs(
        self,
        from_dt: datetime,
        to_dt: datetime,
    ) -> Iterable[Dict[str, Any]]:
        return _search_docs(self._session, from_dt, to_dt)

    def fetch_metadata(self, identifier: str) -> Dict[str, Any]:
        return _fetch_metadata(self._session, identifier)


class EventBuilder:
    """Build events from IA search docs + metadata payloads."""

    def __init__(self, allow_audio_as_primary: bool = ALLOW_AUDIO_AS_PRIMARY):
        self._allow_audio_as_primary = allow_audio_as_primary

    def build_event(
        self,
        doc: Dict[str, Any],
        metadata_payload: Dict[str, Any],
    ) -> Tuple[Optional[EventIngestionModel], Optional[str]]:
        identifier = doc.get("identifier")
        if not identifier:
            return None, "missing_identifier"

        metadata = metadata_payload.get("metadata", {})
        files = metadata_payload.get("files", [])

        title = str(metadata.get("title") or doc.get("title") or identifier).strip()
        if not title:
            return None, "missing_title"

        description = _coerce_description(
            metadata.get("description") or doc.get("description")
        )
        media_files = _media_inventory(
            identifier,
            files,
            download_base_uri=_metadata_download_base_uri(metadata_payload),
        )
        primary_media, ranked_videos = _select_primary_video(
            media_files,
            allow_audio_as_primary=self._allow_audio_as_primary,
        )
        if primary_media is None:
            return None, "missing_video"

        event_datetime = _extract_datetime(
            description,
            metadata.get("date") or doc.get("date"),
            media_files=media_files,
        )
        if event_datetime == EPOCH_SENTINEL:
            return None, "malformed_date"

        # IA /download URLs often 302; resolve ahead of CDP validator HEAD checks.
        primary_video_uri = _resolve_redirected_media_uri(primary_media.uri)
        video_uris = (
            [primary_video_uri] + [video.uri for video in ranked_videos[1:]]
            if ranked_videos
            else [primary_video_uri]
        )

        parts = EventParts(
            body_name=_normalize_body_name(title),
            event_name=title,
            source_uri=_event_source_uri(identifier),
            event_datetime=event_datetime,
            primary_video_uri=primary_video_uri,
            video_uris=video_uris,
        )
        return _build_event_model(parts), None


def _build_session(user_agent: str) -> requests.Session:
    """Create a resilient HTTP session for Internet Archive API requests."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
    )
    return session


def _coerce_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC, assuming naive values are already UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_ia_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse IA date strings into timezone-aware UTC datetimes."""
    if not value:
        return None

    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    # IA returns mixed formats: YYYY-MM-DD and full ISO timestamps.
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_datetime_from_description(description: str) -> Optional[datetime]:
    """Parse datetime text embedded in item descriptions when available."""
    text = description.strip()
    match = re.search(
        (
            r"([A-Za-z]+ \d{1,2}, \d{4})\s+at\s+(\d{1,2}:\d{2})"
            r"(?:\s*([AaPp]\.?\s*[Mm]\.?))?"
        ),
        text,
        flags=re.IGNORECASE,
    )

    if match:
        date_part, time_part, meridiem = match.groups()
        try:
            if meridiem:
                normalized = re.sub(r"[.\s]", "", meridiem).upper()
                return datetime.strptime(
                    f"{date_part} {time_part} {normalized}",
                    "%B %d, %Y %I:%M %p",
                ).replace(tzinfo=timezone.utc)

            return datetime.strptime(
                f"{date_part} {time_part}",
                "%B %d, %Y %H:%M",
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    return None


def _parse_datetime_fallback(fallback_date: Optional[str]) -> datetime:
    """Fallback parser for metadata date values."""
    parsed = _parse_ia_datetime(fallback_date)
    if parsed:
        # The IA date field may only provide date precision, not event time.
        return parsed
    return EPOCH_SENTINEL


def _date_only_utc(dt: datetime) -> datetime:
    """Normalize datetime to UTC date precision with midnight time."""
    normalized = _coerce_utc(dt)
    # Time unknown: retain only the date at 00:00 UTC.
    return datetime(
        normalized.year,
        normalized.month,
        normalized.day,
        tzinfo=timezone.utc,
    )


def _parse_mtime(value: Any) -> Optional[datetime]:
    """Parse IA file mtime values into UTC datetime."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        try:
            return datetime.fromtimestamp(int(text), tz=timezone.utc)
        except (ValueError, OverflowError):
            return None

    return _parse_ia_datetime(text)


def _parse_size_bytes(value: Any) -> int:
    """Parse IA file size into integer bytes with safe fallback."""
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return 0


def _is_artifact_file(name: str) -> bool:
    """Return True when filename matches IA system/artifact patterns."""
    lowered = name.lower()
    if lowered.endswith("/"):
        return True
    if lowered.startswith("__ia_"):
        return True
    if "_meta." in lowered:
        return True
    if lowered.endswith("_files.xml"):
        return True
    if lowered.endswith(".torrent"):
        return True
    if ".thumbs/" in lowered:
        return True
    return False


def _file_extension(name: str) -> str:
    """Extract lowercased filename extension without leading dot."""
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[1].lower()


def _classify_media_kind(ext: str) -> str:
    """Classify media kind from extension."""
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "other"


def _parse_resolution_from_name(name: str) -> int:
    """Extract resolution hint (e.g. 720p) from filename when present."""
    match = re.search(r"(\d{3,4})p", name, flags=re.IGNORECASE)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _build_media_file(
    identifier: str,
    file_record: Dict[str, Any],
    download_base_uri: Optional[str] = None,
) -> Optional[MediaFile]:
    """Build normalized MediaFile from an IA file record."""
    name = str(file_record.get("name", "")).strip()
    if not name or _is_artifact_file(name):
        return None

    ext = _file_extension(name)
    return MediaFile(
        name=name,
        ext=ext,
        uri=_build_download_uri(identifier, name, download_base_uri),
        size_bytes=_parse_size_bytes(file_record.get("size")),
        mtime=_parse_mtime(file_record.get("mtime")),
        source=str(file_record.get("source", "")).lower(),
        media_kind=_classify_media_kind(ext),
        resolution=_parse_resolution_from_name(name),
    )


def _metadata_download_base_uri(metadata_payload: Dict[str, Any]) -> Optional[str]:
    """Build a direct IA media host base URI from metadata payload when available."""
    host = str(
        metadata_payload.get("d1") or metadata_payload.get("server") or ""
    ).strip()
    directory = str(metadata_payload.get("dir") or "").strip()
    if not host or not directory:
        return None

    if not directory.startswith("/"):
        directory = f"/{directory}"
    return f"https://{host}{directory}"


def _build_download_uri(
    identifier: str,
    filename: str,
    download_base_uri: Optional[str] = None,
) -> str:
    """Build a downloadable URI for a media filename."""
    if download_base_uri:
        base = download_base_uri.rstrip("/")
        return requests.utils.requote_uri(f"{base}/{filename}")
    return requests.utils.requote_uri(
        IA_DOWNLOAD_URL.format(identifier=identifier, filename=filename)
    )


@lru_cache(maxsize=2048)
def _resolve_redirected_media_uri(uri: str) -> str:
    """Resolve HTTP redirects so downstream URL checks see final media URLs."""
    if not uri.startswith("http"):
        return uri

    try:
        response = requests.head(
            uri,
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            verify=False,
        )
    except requests.RequestException:
        return uri

    if response.status_code >= 400:
        return uri

    resolved = str(response.url or "").strip()
    if not resolved.startswith("http"):
        return uri
    return resolved


def _media_inventory(
    identifier: str,
    files: Iterable[Dict[str, Any]],
    download_base_uri: Optional[str] = None,
) -> List[MediaFile]:
    """Build a normalized media inventory from IA files metadata."""
    inventory: List[MediaFile] = []
    for file_record in files:
        media = _build_media_file(
            identifier,
            file_record,
            download_base_uri=download_base_uri,
        )
        if media is not None:
            inventory.append(media)
    return inventory


def _rank_media_candidates(candidates: Iterable[MediaFile]) -> List[MediaFile]:
    """Rank media candidates with deterministic quality-first ordering."""

    def _sort_key(media: MediaFile) -> tuple[int, int, int, int, int, str]:
        mtime_ts = int(media.mtime.timestamp()) if media.mtime else 0
        source_rank = 0 if media.source == "original" else 1
        video_ext_rank = VIDEO_EXTENSION_PRIORITY.get(
            media.ext, len(VIDEO_EXTENSION_PRIORITY)
        )
        # Keep audio ranking behavior unchanged.
        if media.media_kind != "video":
            video_ext_rank = 0
        return (
            video_ext_rank,
            -media.size_bytes,
            source_rank,
            -media.resolution,
            -mtime_ts,
            media.name,
        )

    return sorted(candidates, key=_sort_key)


def _coerce_bool(value: Any) -> bool:
    """Parse boolean-like values from kwargs or environment style strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _select_primary_video(
    media_files: Iterable[MediaFile],
    allow_audio_as_primary: bool = ALLOW_AUDIO_AS_PRIMARY,
) -> tuple[Optional[MediaFile], List[MediaFile]]:
    """Select primary ingest media and ranked list of video candidates."""
    videos = _rank_media_candidates(m for m in media_files if m.media_kind == "video")
    if videos:
        return videos[0], videos

    if allow_audio_as_primary:
        audios = _rank_media_candidates(
            m for m in media_files if m.media_kind == "audio"
        )
        if audios:
            return audios[0], []

    return None, []


def _fallback_datetime_from_media(
    media_files: Iterable[MediaFile],
) -> Optional[datetime]:
    """Fallback to newest media mtime date with unknown-time precision."""
    mtimes = [m.mtime for m in media_files if m.mtime is not None]
    if not mtimes:
        return None
    newest = max(mtimes)
    # Time unknown; keep only the date precision at UTC midnight.
    return _date_only_utc(newest)


def _extract_datetime(
    description: Any,
    fallback_date: Optional[str],
    media_files: Optional[Iterable[MediaFile]] = None,
) -> datetime:
    """Extract meeting datetime from description text, then fall back to date fields."""
    parsed = _parse_datetime_from_description(str(description or ""))
    if parsed:
        return parsed

    parsed_fallback = _parse_datetime_fallback(fallback_date)
    if parsed_fallback != EPOCH_SENTINEL:
        return parsed_fallback

    if media_files is not None:
        parsed_media = _fallback_datetime_from_media(media_files)
        if parsed_media:
            return parsed_media

    return EPOCH_SENTINEL


def _normalize_body_name(title: Any) -> str:
    """Map event title text to a known governing body name."""
    lowered = str(title or "").lower()
    for key, body_name in BODY_KEYWORDS.items():
        if key in lowered:
            return body_name
    return DEFAULT_BODY_NAME


def _coerce_description(value: Any) -> str:
    """Convert description payloads (string or list) to a single plain string."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value or "")


def _event_source_uri(identifier: str) -> str:
    """Build a canonical details page URL for an IA item."""
    return IA_DETAILS_URL.format(identifier=identifier)


def _build_event_model(parts: EventParts) -> EventIngestionModel:
    """Create an EventIngestionModel across CDP backend model variants."""
    # Support both modern (body/sessions) and legacy flat model variants.
    try:
        body = ingestion_models.Body(name=parts.body_name)
        sessions = [
            ingestion_models.Session(
                video_uri=parts.primary_video_uri,
                session_datetime=parts.event_datetime,
                session_index=0,
            )
        ]
        return EventIngestionModel(body=body, sessions=sessions)
    except (TypeError, ValueError, AttributeError):
        return EventIngestionModel(
            body_name=parts.body_name,
            event_name=parts.event_name,
            source_uri=parts.source_uri,
            event_datetime=parts.event_datetime,
            video_uris=parts.video_uris,
        )


def _throttled_get_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform a throttled GET request and return JSON payload."""
    time.sleep(REQUEST_DELAY_SECONDS)
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _search_docs(
    session: requests.Session,
    from_dt: datetime,
    to_dt: datetime,
) -> Iterable[Dict[str, Any]]:
    """Yield advanced search docs within the requested datetime window."""
    page = 1

    while True:
        params = {
            "q": DEFAULT_SEARCH_QUERY,
            "fl[]": ["identifier", "title", "date", "description"],
            "sort[]": "date desc",
            "rows": PAGE_SIZE,
            "page": page,
            "output": "json",
        }
        payload = _throttled_get_json(
            session,
            IA_ADVANCED_SEARCH_URL,
            params=params,
        ).get("response", {})
        docs = payload.get("docs", [])
        if not docs:
            break

        for doc in docs:
            event_dt = _parse_ia_datetime(doc.get("date"))
            if not event_dt:
                continue

            if from_dt <= event_dt <= to_dt:
                yield doc

            # Results are sorted newest->oldest; stop when entirely outside lower bound.
            if event_dt < from_dt:
                return

        page += 1


def _fetch_metadata(
    session: requests.Session,
    identifier: str,
) -> Dict[str, Any]:
    """Fetch full metadata payload for an IA item identifier."""
    return _throttled_get_json(
        session,
        IA_METADATA_URL.format(identifier=identifier),
    )


def _doc_to_event(
    doc: Dict[str, Any],
    metadata_payload: Dict[str, Any],
    allow_audio_as_primary: bool = ALLOW_AUDIO_AS_PRIMARY,
) -> Optional[EventIngestionModel]:
    """Compatibility helper for tests that returns only the event object."""
    builder = EventBuilder(allow_audio_as_primary=allow_audio_as_primary)
    event, _ = builder.build_event(doc, metadata_payload)
    return event


def get_events(
    from_dt: datetime,
    to_dt: datetime,
    **kwargs,
) -> List[EventIngestionModel]:
    """
    Get all events for the provided timespan.

    Parameters
    ----------
    from_dt: datetime
        Datetime to start event gather from.
    to_dt: datetime
        Datetime to end event gather at.

    Returns
    -------
    events: List[EventIngestionModel]
        All events gathered that occured in the provided time range.

    Notes
    -----
    As the implimenter of the get_events function, you can choose to ignore the from_dt
    and to_dt parameters. However, they are useful for manually kicking off pipelines
    from GitHub Actions UI.
    """

    # Allow runtime override for contact identity, then fall back to default.
    user_agent = (
        kwargs.get("user_agent") or os.getenv("IA_USER_AGENT") or DEFAULT_USER_AGENT
    )
    allow_audio_as_primary = _coerce_bool(
        kwargs.get("allow_audio_as_primary", ALLOW_AUDIO_AS_PRIMARY)
    )
    report_dict = kwargs.get("report")

    from_dt_utc = _coerce_utc(from_dt)
    to_dt_utc = _coerce_utc(to_dt)

    ia_client = kwargs.get("ia_client")
    if ia_client is None:
        session = _build_session(user_agent=user_agent)
        ia_client = InternetArchiveClient(session=session)

    event_builder = kwargs.get("event_builder") or EventBuilder(
        allow_audio_as_primary=allow_audio_as_primary
    )

    events: List[EventIngestionModel] = []
    summary = IngestionSummary()

    try:
        docs_iterator = iter(ia_client.search_docs(from_dt_utc, to_dt_utc))
    except Exception as exc:
        summary.increment_skip("search_error")
        print(f"[CDP-Richmond-Ingest] Failed to initialize search iteration: {exc}")
        summary.emit_stdout()
        if isinstance(report_dict, dict):
            report_dict.update(summary.to_dict())
        return events

    while True:
        try:
            doc = next(docs_iterator)
        except StopIteration:
            break
        except Exception as exc:
            summary.increment_skip("search_error")
            print(f"[CDP-Richmond-Ingest] Search iteration failed: {exc}")
            break

        summary.total_items_fetched += 1

        identifier = doc.get("identifier")
        if not identifier:
            summary.increment_skip("missing_identifier")
            print("[CDP-Richmond-Ingest] Skipping item: missing identifier")
            continue

        try:
            metadata_payload = ia_client.fetch_metadata(identifier)
        except Exception as exc:
            summary.increment_skip("metadata_fetch_error")
            print(
                "[CDP-Richmond-Ingest] "
                f"Skipping '{identifier}': metadata fetch error ({exc})"
            )
            continue

        try:
            event, skip_reason = event_builder.build_event(doc, metadata_payload)
        except Exception as exc:
            summary.increment_skip("transform_error")
            print(
                "[CDP-Richmond-Ingest] "
                f"Skipping '{identifier}': transform error ({exc})"
            )
            continue

        if event is None:
            summary.increment_skip(skip_reason or "skipped")
            print(
                f"[CDP-Richmond-Ingest] Skipping '{identifier}': "
                f"{(skip_reason or 'skipped').replace('_', ' ')}"
            )
            continue

        events.append(event)
        summary.ingested += 1

    summary.emit_stdout()
    if isinstance(report_dict, dict):
        report_dict.update(summary.to_dict())
    return events
