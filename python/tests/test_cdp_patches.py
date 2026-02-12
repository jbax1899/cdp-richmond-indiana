import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = ROOT / "cdp_patches.py"


class _Frame:
    def __init__(self, name: str, blank: bool = False, kind: str = None):
        self.name = name
        self.kind = kind or ("blank" if blank else "content")
        self.shape = (90, 160, 3)


class _FakeReader:
    def __init__(self, frames, fps: float, duration: float):
        self._frames = frames
        self._fps = fps
        self._duration = duration
        self.requests = []
        self.closed = False

    def get_meta_data(self):
        return {"fps": self._fps, "duration": self._duration}

    def get_length(self):
        return len(self._frames)

    def get_data(self, idx):
        idx = int(idx)
        self.requests.append(idx)
        if idx < 0 or idx >= len(self._frames):
            raise IndexError(idx)
        return self._frames[idx]

    def close(self):
        self.closed = True


class _FakeImageio(types.ModuleType):
    def __init__(self, reader: _FakeReader):
        super().__init__("imageio")
        self._reader = reader
        self.writes = []

    def get_reader(self, _video_path):
        return self._reader

    def imwrite(self, path, image):
        self.writes.append((path, image))


class _FakeGrayImage:
    def __init__(self, frame: _Frame, is_edge: bool = False):
        self.frame = frame
        self.is_edge = is_edge

    def convert(self, _mode):
        return self

    def resize(self, _size):
        return self

    def filter(self, _filter):
        return _FakeGrayImage(self.frame, is_edge=True)

    def getextrema(self):
        if self.frame.kind == "blank":
            return (0, 4)
        if self.frame.kind == "gray_slate":
            return (190, 239)
        return (10, 220)


class _FakeImageModule(types.ModuleType):
    def __init__(self):
        super().__init__("PIL.Image")

    @staticmethod
    def fromarray(frame):
        return _FakeGrayImage(frame)


class _FakeImageStatModule(types.ModuleType):
    def __init__(self):
        super().__init__("PIL.ImageStat")

    class Stat:
        def __init__(self, gray_image: _FakeGrayImage):
            kind = gray_image.frame.kind
            if gray_image.is_edge:
                if kind == "blank":
                    self.mean = [1.0]
                    self.var = [1.0]
                elif kind == "gray_slate":
                    self.mean = [19.0]
                    self.var = [16.0]
                else:
                    self.mean = [50.0]
                    self.var = [225.0]
                return

            if kind == "blank":
                self.mean = [5.0]
                self.var = [4.0]
            elif kind == "gray_slate":
                self.mean = [221.0]
                self.var = [153.0]
            else:
                self.mean = [80.0]
                self.var = [196.0]


class _FakeImageFilterModule(types.ModuleType):
    FIND_EDGES = object()


def _load_patches_module():
    module_name = "cdp_patches_under_test"
    spec = importlib.util.spec_from_file_location(module_name, PATCHES_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _install_cdp_backend_stubs():
    cdp_backend = types.ModuleType("cdp_backend")
    utils = types.ModuleType("cdp_backend.utils")
    file_utils = types.ModuleType("cdp_backend.utils.file_utils")
    file_utils.find_proper_resize_ratio = lambda _h, _w: 1.0
    utils.file_utils = file_utils
    cdp_backend.utils = utils
    return cdp_backend, utils, file_utils


def _install_ffmpeg_stub(duration: float, fps: float):
    ffmpeg = types.ModuleType("ffmpeg")
    fps_num = int(round(fps))

    def _probe(_video_path):
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": str(duration),
                    "avg_frame_rate": f"{fps_num}/1",
                }
            ],
            "format": {"duration": str(duration)},
        }

    ffmpeg.probe = _probe
    return ffmpeg


class CdpPatchesStaticThumbnailTests(unittest.TestCase):
    def _run_static_thumbnail(self, frames, fps=1.0, duration=None):
        duration = float(duration if duration is not None else len(frames))
        reader = _FakeReader(frames, fps=float(fps), duration=duration)
        imageio_mod = _FakeImageio(reader)
        image_mod = _FakeImageModule()
        imagestat_mod = _FakeImageStatModule()
        imagefilter_mod = _FakeImageFilterModule("PIL.ImageFilter")
        pil_pkg = types.ModuleType("PIL")
        pil_pkg.Image = image_mod
        pil_pkg.ImageStat = imagestat_mod
        pil_pkg.ImageFilter = imagefilter_mod
        cdp_backend, utils, file_utils = _install_cdp_backend_stubs()
        ffmpeg = _install_ffmpeg_stub(duration=duration, fps=float(fps))

        with patch.dict(
            sys.modules,
            {
                "cdp_backend": cdp_backend,
                "cdp_backend.utils": utils,
                "cdp_backend.utils.file_utils": file_utils,
                "imageio": imageio_mod,
                "PIL": pil_pkg,
                "PIL.Image": image_mod,
                "PIL.ImageStat": imagestat_mod,
                "PIL.ImageFilter": imagefilter_mod,
                "ffmpeg": ffmpeg,
            },
            clear=False,
        ):
            mod = _load_patches_module()
            mod.patch_thumbnails(num_frames=10, clip_duration_seconds=2.0)
            output_path = file_utils.get_static_thumbnail("fake-video.mp4", "session-hash")

        self.assertTrue(reader.closed)
        self.assertTrue(imageio_mod.writes)
        saved_path, saved_image = imageio_mod.writes[-1]
        self.assertEqual(saved_path, output_path)
        return saved_image, reader.requests

    def test_static_thumbnail_prefers_first_frame_when_not_blank(self):
        frames = [_Frame("f0", blank=False), _Frame("f1", blank=False), _Frame("f2", blank=False)]
        saved, requests = self._run_static_thumbnail(frames, fps=1.0, duration=30.0)
        self.assertEqual(saved.name, "f0")
        self.assertEqual(requests, [0])

    def test_static_thumbnail_skips_blank_lead_in(self):
        frames = [_Frame(f"f{i}", blank=(i < 10)) for i in range(30)]
        saved, requests = self._run_static_thumbnail(frames, fps=1.0, duration=30.0)
        self.assertEqual(saved.name, "f10")
        self.assertEqual(requests[0], 0)
        self.assertIn(10, requests)

    def test_static_thumbnail_falls_back_to_first_frame_when_all_blank(self):
        frames = [_Frame(f"f{i}", blank=True) for i in range(18)]
        saved, requests = self._run_static_thumbnail(frames, fps=1.0, duration=18.0)
        self.assertEqual(saved.name, "f0")
        self.assertEqual(requests[0], 0)

    def test_static_thumbnail_skips_gray_slate_lead_in(self):
        frames = [_Frame(f"f{i}", kind=("gray_slate" if i < 8 else "content")) for i in range(20)]
        saved, requests = self._run_static_thumbnail(frames, fps=1.0, duration=20.0)
        self.assertEqual(saved.name, "f8")
        self.assertEqual(requests[0], 0)
        self.assertIn(8, requests)


if __name__ == "__main__":
    unittest.main()
