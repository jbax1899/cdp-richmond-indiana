import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
REGENERATE_PATH = ROOT / "regenerate_data.py"


def _load_regenerate_module():
    module_name = "regenerate_data_under_test"
    spec = importlib.util.spec_from_file_location(module_name, REGENERATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRef:
    def __init__(self, path: str):
        self.path = path


class _FakeDoc:
    def __init__(self, data, path: str, exists: bool = True):
        self._data = dict(data)
        self.reference = _FakeRef(path)
        self.exists = exists

    def to_dict(self):
        return dict(self._data)


class RegenerateThumbnailTests(unittest.TestCase):
    def test_remote_first_succeeds_without_download(self):
        mod = _load_regenerate_module()
        static_ref = object()
        hover_ref = object()
        event_doc = _FakeDoc(
            {"static_thumbnail_ref": static_ref, "hover_thumbnail_ref": hover_ref},
            path="event/E1",
        )
        session_doc = _FakeDoc(
            {"video_uri": "https://example.com/video.mp4", "session_content_hash": "abc"},
            path="session/S1",
        )
        static_file_doc = _FakeDoc({"uri": "gs://bucket/thumbnails/static.png"}, path="file/F1")
        hover_file_doc = _FakeDoc({"uri": "gs://bucket/thumbnails/hover.mp4"}, path="file/F2")

        calls = {"resource_copy": 0, "static_inputs": [], "hover_inputs": [], "uploads": []}

        file_utils = types.ModuleType("cdp_backend.utils.file_utils")

        def _get_static_thumbnail(video_path, _sess_hash):
            calls["static_inputs"].append(video_path)
            return "abc-static-thumbnail.png"

        def _get_hover_thumbnail(video_path, _sess_hash):
            calls["hover_inputs"].append(video_path)
            return "abc-hover-preview.mp4"

        def _resource_copy(**_kwargs):
            calls["resource_copy"] += 1
            return "local-video.mp4"

        file_utils.get_static_thumbnail = _get_static_thumbnail
        file_utils.get_hover_thumbnail = _get_hover_thumbnail
        file_utils.resource_copy = _resource_copy

        cdp_backend = types.ModuleType("cdp_backend")
        cdp_backend_utils = types.ModuleType("cdp_backend.utils")
        cdp_backend_utils.file_utils = file_utils
        cdp_backend.utils = cdp_backend_utils

        cdp_patches = types.ModuleType("cdp_patches")
        cdp_patches.patch_thumbnails = lambda **_kwargs: None

        def _fetch_file_doc(_db, _cols, file_ref):
            return static_file_doc if file_ref is static_ref else hover_file_doc

        def _upload_overwrite(**kwargs):
            calls["uploads"].append((str(kwargs["local_path"]), kwargs["remote_object_name"]))
            return "ok"

        with patch.dict(
            sys.modules,
            {
                "cdp_patches": cdp_patches,
                "cdp_backend": cdp_backend,
                "cdp_backend.utils": cdp_backend_utils,
                "cdp_backend.utils.file_utils": file_utils,
            },
            clear=False,
        ), patch.object(mod, "_fetch_file_doc", side_effect=_fetch_file_doc), patch.object(
            mod, "_upload_overwrite", side_effect=_upload_overwrite
        ):
            mod._regenerate_event_thumbnails(
                repo_root=ROOT.parent,
                storage_client=object(),
                bucket="bucket",
                db=object(),
                cols=object(),
                event_doc=event_doc,
                sessions=[session_doc],
                dry_run=False,
            )

        self.assertEqual(calls["resource_copy"], 0)
        self.assertEqual(calls["static_inputs"], ["https://example.com/video.mp4"])
        self.assertEqual(calls["hover_inputs"], ["https://example.com/video.mp4"])
        self.assertEqual(len(calls["uploads"]), 2)
        self.assertEqual(calls["uploads"][0][1], "thumbnails/static.png")
        self.assertEqual(calls["uploads"][1][1], "thumbnails/hover.mp4")

    def test_remote_failure_falls_back_to_local_download_with_warning(self):
        mod = _load_regenerate_module()
        static_ref = object()
        hover_ref = object()
        event_doc = _FakeDoc(
            {"static_thumbnail_ref": static_ref, "hover_thumbnail_ref": hover_ref},
            path="event/E2",
        )
        session_doc = _FakeDoc(
            {"video_uri": "https://example.com/video.mp4", "session_content_hash": "def"},
            path="session/S2",
        )
        static_file_doc = _FakeDoc({"uri": "gs://bucket/thumbnails/static2.png"}, path="file/F3")
        hover_file_doc = _FakeDoc({"uri": "gs://bucket/thumbnails/hover2.mp4"}, path="file/F4")

        calls = {"resource_copy": 0, "static_inputs": [], "hover_inputs": [], "warnings": []}

        file_utils = types.ModuleType("cdp_backend.utils.file_utils")

        def _get_static_thumbnail(video_path, _sess_hash):
            calls["static_inputs"].append(video_path)
            if str(video_path).startswith("https://"):
                raise RuntimeError("remote static failed")
            return "def-static-thumbnail.png"

        def _get_hover_thumbnail(video_path, _sess_hash):
            calls["hover_inputs"].append(video_path)
            return "def-hover-preview.mp4"

        def _resource_copy(**_kwargs):
            calls["resource_copy"] += 1
            return "local-video.mp4"

        file_utils.get_static_thumbnail = _get_static_thumbnail
        file_utils.get_hover_thumbnail = _get_hover_thumbnail
        file_utils.resource_copy = _resource_copy

        cdp_backend = types.ModuleType("cdp_backend")
        cdp_backend_utils = types.ModuleType("cdp_backend.utils")
        cdp_backend_utils.file_utils = file_utils
        cdp_backend.utils = cdp_backend_utils

        cdp_patches = types.ModuleType("cdp_patches")
        cdp_patches.patch_thumbnails = lambda **_kwargs: None

        def _fetch_file_doc(_db, _cols, file_ref):
            return static_file_doc if file_ref is static_ref else hover_file_doc

        with patch.dict(
            sys.modules,
            {
                "cdp_patches": cdp_patches,
                "cdp_backend": cdp_backend,
                "cdp_backend.utils": cdp_backend_utils,
                "cdp_backend.utils.file_utils": file_utils,
            },
            clear=False,
        ), patch.object(mod, "_fetch_file_doc", side_effect=_fetch_file_doc), patch.object(
            mod, "_upload_overwrite", return_value="ok"
        ), patch.object(mod, "_eprint", side_effect=lambda msg: calls["warnings"].append(msg)):
            mod._regenerate_event_thumbnails(
                repo_root=ROOT.parent,
                storage_client=object(),
                bucket="bucket",
                db=object(),
                cols=object(),
                event_doc=event_doc,
                sessions=[session_doc],
                dry_run=False,
            )

        self.assertEqual(calls["resource_copy"], 1)
        self.assertEqual(
            calls["static_inputs"],
            ["https://example.com/video.mp4", "local-video.mp4"],
        )
        self.assertEqual(calls["hover_inputs"], ["local-video.mp4"])
        self.assertTrue(calls["warnings"])
        self.assertIn("Remote thumbnail generation failed", calls["warnings"][0])


if __name__ == "__main__":
    unittest.main()
