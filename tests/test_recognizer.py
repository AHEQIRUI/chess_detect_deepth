from src.recognizer import ChessRecognizer
from src.camera import MockCamera
import numpy as np
import cv2


def _make_chessboard_image():
    """Create a synthetic full chessboard in initial position for integration test."""
    img = np.full((480, 640, 3), (80, 120, 80), dtype=np.uint8)
    depth = np.full((480, 640), 500.0, dtype=np.float32)

    x, y, s = 70, 40, 400
    cell_s = s // 8
    depth[y:y + s, x:x + s] = 480.0

    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                color = (240, 220, 190)
            else:
                color = (60, 40, 20)
            cv2.rectangle(img,
                          (x + col * cell_s, y + row * cell_s),
                          (x + (col + 1) * cell_s, y + (row + 1) * cell_s),
                          color, -1)

    return img, depth


class TestChessRecognizer:
    def test_run_returns_fen_string(self):
        img, depth = _make_chessboard_image()

        class FixedCamera(MockCamera):
            def capture(self):
                return img.copy(), depth.copy()

        recognizer = ChessRecognizer()
        recognizer.camera = FixedCamera(width=640, height=480)
        recognizer.camera.open()

        fen = recognizer.run()
        assert isinstance(fen, str)
        assert "/" in fen
        assert fen.count("/") == 7

    def test_run_on_empty_board_produces_all_empty(self):
        img, depth = _make_chessboard_image()

        class FixedCamera(MockCamera):
            def capture(self):
                return img.copy(), depth.copy()

        recognizer = ChessRecognizer()
        recognizer.camera = FixedCamera(width=640, height=480)
        recognizer.camera.open()
        fen = recognizer.run()
        parts = fen.split(" ")[0]
        assert all(ch in "12345678/" for ch in parts)
