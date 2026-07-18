import numpy as np
import pytest
from src.camera import MockCamera


class TestMockCamera:
    def test_open_close(self):
        cam = MockCamera(width=640, height=480)
        assert cam.open() is True
        cam.close()

    def test_capture_returns_bgr_and_depth(self):
        cam = MockCamera(width=640, height=480)
        cam.open()
        bgr, depth = cam.capture()
        assert isinstance(bgr, np.ndarray)
        assert isinstance(depth, np.ndarray)
        assert bgr.shape == (480, 640, 3)
        assert depth.shape == (480, 640)
        assert bgr.dtype == np.uint8
        cam.close()

    def test_capture_is_repeatable(self):
        cam = MockCamera(width=640, height=480)
        cam.open()
        bgr1, depth1 = cam.capture()
        bgr2, depth2 = cam.capture()
        assert bgr1.shape == bgr2.shape
        assert depth1.shape == depth2.shape
        cam.close()

    def test_close_is_idempotent(self):
        cam = MockCamera(width=640, height=480)
        cam.open()
        cam.close()
        cam.close()  # should not raise

    def test_capture_before_open_raises(self):
        cam = MockCamera(width=640, height=480)
        with pytest.raises(RuntimeError, match="Camera not opened"):
            cam.capture()
