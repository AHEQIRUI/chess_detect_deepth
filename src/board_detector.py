import cv2
import numpy as np
from src.types import BoardResult


class BoardDetector:
    def __init__(self, output_size: int = 800):
        self.output_size = output_size

    def detect(self, rgb: np.ndarray, depth: np.ndarray,
               corners: np.ndarray | None = None) -> BoardResult:
        if corners is not None:
            return self.warp_from_corners(rgb, depth, corners)
        return self._auto_detect(rgb, depth)

    def _auto_detect(self, rgb: np.ndarray, depth: np.ndarray) -> BoardResult:
        corners = self._find_board_corners(rgb)
        warped_rgb = self._warp(rgb, corners)
        warped_depth = self._warp(depth, corners)
        plane_depth = self._compute_plane_depth(warped_depth)
        return BoardResult(
            corners=corners,
            warped_rgb=warped_rgb,
            warped_depth=warped_depth,
            plane_depth=plane_depth,
        )

    def warp_from_corners(self, rgb: np.ndarray, depth: np.ndarray,
                          corners: np.ndarray) -> BoardResult:
        corners = corners.astype(np.float32)
        warped_rgb = self._warp(rgb, corners)
        warped_depth = self._warp(depth, corners)
        plane_depth = self._compute_plane_depth(warped_depth)
        return BoardResult(
            corners=corners,
            warped_rgb=warped_rgb,
            warped_depth=warped_depth,
            plane_depth=plane_depth,
        )

    def _find_board_corners(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found in image")

        board_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(board_contour, True)
        approx = cv2.approxPolyDP(board_contour, 0.02 * peri, True)

        if len(approx) < 4:
            rect = cv2.minAreaRect(board_contour)
            approx = cv2.boxPoints(rect)
            approx = approx.astype(np.float32)
        else:
            approx = approx.reshape(-1, 2).astype(np.float32)

        return self._order_corners(approx[:4])

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        return rect

    def _warp(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        dst = np.array([
            [0, 0],
            [self.output_size - 1, 0],
            [self.output_size - 1, self.output_size - 1],
            [0, self.output_size - 1],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst)
        return cv2.warpPerspective(image, M, (self.output_size, self.output_size))

    def _compute_plane_depth(self, warped_depth: np.ndarray) -> float:
        margin = self.output_size // 10
        central = warped_depth[margin:-margin, margin:-margin]
        valid = central[(central > 100) & (central < 10000)]
        if len(valid) == 0:
            return 0.0
        # Use histogram peak (dominant depth) to find the board surface.
        # More robust than median when piece pixels are present.
        bins = np.arange(100, 10001, 5)
        hist, edges = np.histogram(valid, bins=bins)
        mode_depth = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
        return float(mode_depth)
