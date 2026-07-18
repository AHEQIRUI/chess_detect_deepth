import numpy as np
import cv2
from src.board_detector import BoardDetector


def _make_synthetic_board(width=800, height=600, board_tl=(100, 80), board_size=400):
    """Create a synthetic top-down chessboard image on a table background."""
    img = np.full((height, width, 3), (80, 120, 80), dtype=np.uint8)  # green table

    x, y = board_tl
    s = board_size
    cell_s = s // 8

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

    depth = np.full((height, width), 500.0, dtype=np.float32)
    depth[y:y + s, x:x + s] = 480.0

    return img, depth, (x, y, s)


class TestBoardDetector:
    def test_detects_corners_on_synthetic_board(self):
        img, depth, (bx, by, bs) = _make_synthetic_board()
        detector = BoardDetector(output_size=800)
        result = detector.detect(img, depth)

        assert result.corners.shape == (4, 2)
        assert result.warped_rgb.shape == (800, 800, 3)
        assert result.warped_depth.shape == (800, 800)
        assert isinstance(result.plane_depth, float)

        # Corners should be near the actual board corners
        expected_corners = np.array([
            [bx, by], [bx + bs, by], [bx + bs, by + bs], [bx, by + bs]
        ], dtype=np.float32)
        for i, ec in enumerate(expected_corners):
            dist = np.linalg.norm(result.corners[i] - ec)
            assert dist < 20, f"Corner {i} off by {dist}px"

    def test_warped_output_is_square(self):
        img, depth, _ = _make_synthetic_board()
        detector = BoardDetector(output_size=800)
        result = detector.detect(img, depth)
        h, w = result.warped_rgb.shape[:2]
        assert h == w == 800

    def test_detect_on_rotated_board(self):
        """Board slightly rotated should still be detected."""
        width, height = 640, 480
        img = np.full((height, width, 3), (80, 120, 80), dtype=np.uint8)
        depth = np.full((height, width), 500.0, dtype=np.float32)

        s = 300
        x, y = 120, 70
        cell_s = s // 8
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
        depth[y:y + s, x:x + s] = 480.0

        # Rotate image by 5 degrees
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, 5, 1.0)
        img = cv2.warpAffine(img, M, (width, height))
        depth = cv2.warpAffine(depth, M, (width, height))

        detector = BoardDetector(output_size=800)
        result = detector.detect(img, depth)
        assert result.corners.shape == (4, 2)
        assert result.warped_rgb.shape == (800, 800, 3)
