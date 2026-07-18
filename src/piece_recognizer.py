import numpy as np
import cv2
from src.types import CellData, PieceResult, PieceType, Color


# Height ordering: King > Queen > Bishop > Knight > Rook > Pawn
# Reference heights in mm for standard Staunton pieces
PIECE_REFERENCE_HEIGHTS = [
    (PieceType.KING, 100),
    (PieceType.QUEEN, 88),
    (PieceType.BISHOP, 72),
    (PieceType.KNIGHT, 60),
    (PieceType.ROOK, 50),
    (PieceType.PAWN, 30),
]


class PieceRecognizer:
    def __init__(self, presence_threshold: float = 10.0,
                 brightness_threshold: int = 128,
                 mask_threshold: float = 5.0):
        self.presence_threshold = presence_threshold
        self.brightness_threshold = brightness_threshold
        self.mask_threshold = mask_threshold

    def recognize(self, cells: list[CellData], board_plane_depth: float) -> list[PieceResult]:
        results = []
        for cell in cells:
            piece_type, color = self._classify_cell(cell, board_plane_depth)
            results.append(PieceResult(
                row=cell.row, col=cell.col,
                piece_type=piece_type, color=color,
            ))
        return results

    def _classify_cell(self, cell: CellData, board_plane_depth: float):
        height = self._measure_height(cell.depth_patch, board_plane_depth)
        if height < self.presence_threshold:
            return None, None

        piece_mask = (cell.depth_patch > 0) & ((board_plane_depth - cell.depth_patch) > self.mask_threshold)
        color = self._classify_color(cell.rgb_patch, piece_mask)
        piece_type = self._classify_by_height(height)
        return piece_type, color

    def _measure_height(self, depth_patch: np.ndarray, board_plane_depth: float) -> float:
        # Only consider pixels near the board plane: within 150mm above it.
        # This filters sensor noise (spurious near-zero values) and background surfaces.
        lo = board_plane_depth - 150
        hi = board_plane_depth + 50
        valid = depth_patch[(depth_patch > lo) & (depth_patch < hi)]
        if len(valid) == 0:
            return 0.0
        piece_depth = np.min(valid)
        return board_plane_depth - piece_depth

    def _classify_color(self, rgb_patch: np.ndarray, piece_mask: np.ndarray) -> Color:
        masked_pixels = rgb_patch[piece_mask]
        if len(masked_pixels) == 0:
            return Color.BLACK
        hsv = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mean_brightness = np.mean(v)
        if mean_brightness > self.brightness_threshold:
            return Color.WHITE
        return Color.BLACK

    def _classify_by_height(self, height: float) -> PieceType:
        # Nearest reference height match
        best_piece = PieceType.PAWN
        best_diff = float('inf')
        for piece_type, ref_height in PIECE_REFERENCE_HEIGHTS:
            diff = abs(height - ref_height)
            if diff < best_diff:
                best_diff = diff
                best_piece = piece_type
        return best_piece
