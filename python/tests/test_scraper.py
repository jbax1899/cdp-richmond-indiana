import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRAPER_PATH = ROOT / "cdp_richmond_(in)_backend" / "scraper.py"


def _install_cdp_backend_stubs() -> None:
    cdp_backend = types.ModuleType("cdp_backend")
    pipeline = types.ModuleType("cdp_backend.pipeline")
    ingestion_models = types.ModuleType("cdp_backend.pipeline.ingestion_models")

    class Body:
        def __init__(self, name):
            self.name = name

    class Session:
        def __init__(self, video_uri, session_datetime, session_index):
            self.video_uri = video_uri
            self.session_datetime = session_datetime
            self.session_index = session_index

    class EventIngestionModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ingestion_models.Body = Body
    ingestion_models.Session = Session
    ingestion_models.EventIngestionModel = EventIngestionModel

    pipeline.ingestion_models = ingestion_models
    cdp_backend.pipeline = pipeline

    sys.modules["cdp_backend"] = cdp_backend
    sys.modules["cdp_backend.pipeline"] = pipeline
    sys.modules["cdp_backend.pipeline.ingestion_models"] = ingestion_models


def _load_scraper_module():
    _install_cdp_backend_stubs()
    spec = importlib.util.spec_from_file_location("scraper_under_test", SCRAPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ScraperRefactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scraper = _load_scraper_module()

    def test_parse_datetime_from_description(self):
        parsed = self.scraper._parse_datetime_from_description(
            "Meeting held January 29, 2026 at 18:30 in council chambers."
        )
        self.assertEqual(parsed, datetime(2026, 1, 29, 18, 30, tzinfo=timezone.utc))

    def test_parse_datetime_from_description_with_meridiem(self):
        parsed = self.scraper._parse_datetime_from_description(
            "Meeting held January 29, 2026 at 6:30 p.m. in council chambers."
        )
        self.assertEqual(parsed, datetime(2026, 1, 29, 18, 30, tzinfo=timezone.utc))

    def test_parse_datetime_fallback_to_epoch(self):
        parsed = self.scraper._parse_datetime_fallback(None)
        self.assertEqual(parsed, datetime(1970, 1, 1, tzinfo=timezone.utc))

    def test_doc_to_event_returns_none_when_no_mp4(self):
        doc = {"identifier": "item-1", "title": "Common Council", "date": "2026-01-29"}
        metadata_payload = {
            "metadata": {"title": "Common Council"},
            "files": [{"name": "audio.mp3"}],
        }
        self.assertIsNone(self.scraper._doc_to_event(doc, metadata_payload))

    def test_build_event_model_falls_back_to_flat_fields(self):
        parts = self.scraper.EventParts(
            body_name="Richmond Common Council",
            event_name="Meeting",
            source_uri="https://archive.org/details/x",
            event_datetime=datetime(2026, 1, 29, tzinfo=timezone.utc),
            primary_video_uri="https://archive.org/download/x/a.mp4",
            video_uris=["https://archive.org/download/x/a.mp4"],
        )

        with patch.object(self.scraper.ingestion_models, "Body", side_effect=TypeError):
            event = self.scraper._build_event_model(parts)

        self.assertEqual(event.kwargs["body_name"], "Richmond Common Council")
        self.assertIn("video_uris", event.kwargs)

    def test_search_docs_stops_after_lower_bound(self):
        class FakeSession:
            def __init__(self):
                self.calls = 0

            def get(self, url, params=None, timeout=None):
                self.calls += 1
                docs = (
                    [
                        {"identifier": "a", "date": "2026-01-30"},
                        {"identifier": "b", "date": "2026-01-29"},
                        {"identifier": "c", "date": "2025-12-31"},
                    ]
                    if self.calls == 1
                    else []
                )
                payload = {"response": {"docs": docs}}

                class Resp:
                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return payload

                return Resp()

        from_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        to_dt = datetime(2026, 12, 31, tzinfo=timezone.utc)

        with patch.object(self.scraper.time, "sleep", return_value=None):
            docs = list(self.scraper._search_docs(FakeSession(), from_dt, to_dt))

        self.assertEqual([d["identifier"] for d in docs], ["a", "b"])

    def test_media_inventory_excludes_artifacts(self):
        files = [
            {"name": "__ia_thumb.jpg", "size": "100"},
            {"name": "item_meta.xml", "size": "100"},
            {"name": "item_files.xml", "size": "100"},
            {"name": "item_archive.torrent", "size": "100"},
            {"name": "item.thumbs/", "size": "100"},
            {"name": "valid_video.mp4", "size": "200"},
            {"name": "valid_audio.mp3", "size": "300"},
        ]

        media = self.scraper._media_inventory("item", files)
        self.assertEqual([m.name for m in media], ["valid_video.mp4", "valid_audio.mp3"])

    def test_select_primary_video_uses_largest_video(self):
        files = [
            {"name": "meeting_480p.mp4", "size": "1000"},
            {"name": "meeting_720p.mp4", "size": "2000"},
            {"name": "meeting_1080p.mov", "size": "1500"},
        ]
        media = self.scraper._media_inventory("item", files)
        primary, videos = self.scraper._select_primary_video(media)

        self.assertIsNotNone(primary)
        self.assertEqual(primary.name, "meeting_720p.mp4")
        self.assertEqual([v.name for v in videos], ["meeting_720p.mp4", "meeting_1080p.mov", "meeting_480p.mp4"])

    def test_select_primary_video_can_fallback_to_audio(self):
        files = [
            {"name": "meeting_audio_low.mp3", "size": "1000"},
            {"name": "meeting_audio_high.mp3", "size": "2000"},
        ]
        media = self.scraper._media_inventory("item", files)
        primary, videos = self.scraper._select_primary_video(
            media,
            allow_audio_as_primary=True,
        )

        self.assertIsNotNone(primary)
        self.assertEqual(primary.name, "meeting_audio_high.mp3")
        self.assertEqual(videos, [])

    def test_extract_datetime_fallbacks_to_media_mtime_date(self):
        media = self.scraper._media_inventory(
            "item",
            [
                {"name": "meeting.mp4", "size": "1000", "mtime": "1738195200"},
            ],
        )
        parsed = self.scraper._extract_datetime("", None, media_files=media)
        self.assertEqual(parsed, datetime(2025, 1, 30, 0, 0, tzinfo=timezone.utc))

    def test_doc_to_event_uses_primary_video_uri(self):
        doc = {"identifier": "item-1", "title": "Common Council", "date": "2026-01-29"}
        metadata_payload = {
            "metadata": {"title": "Common Council", "date": "2026-01-29"},
            "files": [
                {"name": "meeting_small.mp4", "size": "1000"},
                {"name": "meeting_large.mp4", "size": "2000"},
            ],
        }

        event = self.scraper._doc_to_event(doc, metadata_payload)
        self.assertIsNotNone(event)
        session = event.kwargs["sessions"][0]
        self.assertTrue(session.video_uri.endswith("meeting_large.mp4"))

    def test_get_events_end_to_end_with_fake_client(self):
        class FakeClient:
            def search_docs(self, from_dt, to_dt):
                return iter(
                    [
                        {"identifier": "item-1", "title": "Common Council", "date": "2026-01-29"},
                    ]
                )

            def fetch_metadata(self, identifier):
                return {
                    "metadata": {"title": "Common Council", "date": "2026-01-29"},
                    "files": [
                        {"name": "meeting_small.mp4", "size": "1000"},
                        {"name": "meeting_large.mp4", "size": "2000"},
                    ],
                }

        report = {}
        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            events = self.scraper.get_events(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 12, 31, tzinfo=timezone.utc),
                ia_client=FakeClient(),
                report=report,
            )

        self.assertEqual(len(events), 1)
        session = events[0].kwargs["sessions"][0]
        self.assertTrue(session.video_uri.endswith("meeting_large.mp4"))
        self.assertEqual(report["total_items_fetched"], 1)
        self.assertEqual(report["ingested"], 1)
        self.assertEqual(report["skipped"], 0)
        self.assertIn("[CDP-Richmond-Ingest Summary]", stdout_buffer.getvalue())

    def test_get_events_isolates_per_item_errors_and_counts_skips(self):
        class FakeClient:
            def __init__(self):
                self.docs = [
                    {"identifier": "ok-item", "title": "Common Council", "date": "2026-01-29"},
                    {"identifier": "bad-fetch", "title": "Common Council", "date": "2026-01-29"},
                    {"identifier": "no-video", "title": "Common Council", "date": "2026-01-29"},
                ]

            def search_docs(self, from_dt, to_dt):
                return iter(self.docs)

            def fetch_metadata(self, identifier):
                if identifier == "bad-fetch":
                    raise RuntimeError("metadata unavailable")
                if identifier == "no-video":
                    return {
                        "metadata": {"title": "Common Council", "date": "2026-01-29"},
                        "files": [{"name": "audio.mp3", "size": "1200"}],
                    }
                return {
                    "metadata": {"title": "Common Council", "date": "2026-01-29"},
                    "files": [{"name": "meeting.mp4", "size": "9999"}],
                }

        report = {}
        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            events = self.scraper.get_events(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 12, 31, tzinfo=timezone.utc),
                ia_client=FakeClient(),
                report=report,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(report["total_items_fetched"], 3)
        self.assertEqual(report["ingested"], 1)
        self.assertEqual(report["skipped"], 2)
        self.assertEqual(report["skipped_by_reason"]["metadata_fetch_error"], 1)
        self.assertEqual(report["skipped_by_reason"]["missing_video"], 1)
        summary_text = stdout_buffer.getvalue()
        self.assertIn("Skipped: 2", summary_text)
        self.assertIn("missing video", summary_text)

    def test_get_events_skips_epoch_dated_items(self):
        class FakeClient:
            def search_docs(self, from_dt, to_dt):
                return iter(
                    [
                        {"identifier": "epoch-item", "title": "Common Council"},
                    ]
                )

            def fetch_metadata(self, identifier):
                return {
                    "metadata": {"title": "Common Council"},
                    "files": [{"name": "meeting.mp4", "size": "1200"}],
                }

        report = {}
        events = self.scraper.get_events(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 12, 31, tzinfo=timezone.utc),
            ia_client=FakeClient(),
            report=report,
        )

        self.assertEqual(events, [])
        self.assertEqual(report["skipped_by_reason"]["malformed_date"], 1)


if __name__ == "__main__":
    unittest.main()
