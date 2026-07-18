import numpy as np
import cv2
from src.types import CellData, PieceType, Color
from src.piece_recognizer import PieceRecognizer


def _make_cell_with_piece(rgb_center_val, depth_val, board_depth,
                          center_size=30, cell_size=100):
    """Create a synthetic cell with a piece at given depth."""
    rgb = np.full((cell_size, cell_size, 3), (100, 80, 60), dtype=np.uint8)
    half = center_size // 2
    c = cell_size // 2
    rgb[c - half:c + half, c - half:c + half] = rgb_center_val

    depth = np.full((cell_size, cell_size), board_depth, dtype=np.float32)
    depth[c - half:c + half, c - half:c + half] = depth_val

    return CellData(row=0, col=0, rgb_patch=rgb, depth_patch=depth)


def _make_empty_cell(board_depth, cell_size=100):
    rgb = np.full((cell_size, cell_size, 3), (100, 80, 60), dtype=np.uint8)
    depth = np.full((cell_size, cell_size), board_depth, dtype=np.float32)
    return CellData(row=0, col=0, rgb_patch=rgb, depth_patch=depth)


class TestPieceRecognizer:
    def test_empty_square(self):
        board_depth = 500.0
        cell = _make_empty_cell(board_depth)
        recognizer = PieceRecognizer()
        results = recognizer.recognize([cell], board_depth)
        assert len(results) == 1
        assert results[0].piece_type is None
        assert results[0].color is None

    def test_piece_present_by_depth(self):
        board_depth = 500.0
        cell = _make_cell_with_piece((200, 180, 160), 420.0, board_depth)
        recognizer = PieceRecognizer()
        results = recognizer.recognize([cell], board_depth)
        assert results[0].piece_type is not None

    def test_white_piece_color(self):
        board_depth = 500.0
        cell = _make_cell_with_piece((220, 210, 200), 400.0, board_depth)
        recognizer = PieceRecognizer()
        results = recognizer.recognize([cell], board_depth)
        assert results[0].color == Color.WHITE

    def test_black_piece_color(self):
        board_depth = 500.0
        cell = _make_cell_with_piece((30, 25, 20), 400.0, board_depth)
        recognizer = PieceRecognizer()
        results = recognizer.recognize([cell], board_depth)
        assert results[0].color == Color.BLACK

    def test_height_classification_order(self):
        """King should be classified by greatest height (lowest depth value)."""
        board_depth = 500.0
        # Reference heights: King~100, Queen~88, Bishop~72, Knight~60, Rook~50, Pawn~30
        heights = {
            PieceType.KING: 400.0,     # height = 100
            PieceType.QUEEN: 412.0,     # height = 88
            PieceType.BISHOP: 428.0,    # height = 72
            PieceType.KNIGHT: 440.0,    # height = 60
            PieceType.ROOK: 450.0,      # height = 50
            PieceType.PAWN: 470.0,      # height = 30
        }
        for expected_type, depth_val in heights.items():
            cell = _make_cell_with_piece((50, 50, 50), depth_val, board_depth)
            recognizer = PieceRecognizer()
            results = recognizer.recognize([cell], board_depth)
            assert results[0].piece_type == expected_type, \
                f"Expected {expected_type}, got {results[0].piece_type} for depth {depth_val}"

    def test_multiple_cells(self):
        board_depth = 500.0
        cells = [
            _make_empty_cell(board_depth),
            _make_cell_with_piece((220, 210, 200), 390.0, board_depth),  # white king
            _make_cell_with_piece((30, 25, 20), 470.0, board_depth),     # black pawn
        ]
        cells[0].row, cells[0].col = 0, 0
        cells[1].row, cells[1].col = 0, 1
        cells[2].row, cells[2].col = 0, 2

        recognizer = PieceRecognizer()
        results = recognizer.recognize(cells, board_depth)
        assert results[0].piece_type is None
        assert results[1].piece_type == PieceType.KING
        assert results[1].color == Color.WHITE
        assert results[2].piece_type == PieceType.PAWN
        assert results[2].color == Color.BLACK
