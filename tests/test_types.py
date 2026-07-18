import numpy as np
from src.types import PieceType, Color, CellData, PieceResult, BoardResult


class TestPieceType:
    def test_piece_type_values(self):
        assert PieceType.KING.value == 'K'
        assert PieceType.QUEEN.value == 'Q'
        assert PieceType.ROOK.value == 'R'
        assert PieceType.BISHOP.value == 'B'
        assert PieceType.KNIGHT.value == 'N'
        assert PieceType.PAWN.value == 'P'


class TestColor:
    def test_color_values(self):
        assert Color.WHITE.value == 'white'
        assert Color.BLACK.value == 'black'


class TestCellData:
    def test_cell_data_creation(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.uint16) * 500
        cell = CellData(row=0, col=3, rgb_patch=rgb, depth_patch=depth)
        assert cell.row == 0
        assert cell.col == 3
        assert cell.rgb_patch.shape == (100, 100, 3)
        assert cell.depth_patch.shape == (100, 100)


class TestPieceResult:
    def test_piece_result_with_piece(self):
        result = PieceResult(row=0, col=0, piece_type=PieceType.KING, color=Color.WHITE)
        assert result.piece_type == PieceType.KING
        assert result.color == Color.WHITE

    def test_piece_result_empty_square(self):
        result = PieceResult(row=3, col=4, piece_type=None, color=None)
        assert result.piece_type is None
        assert result.color is None


class TestBoardResult:
    def test_board_result_creation(self):
        corners = np.array([[10, 20], [790, 20], [790, 790], [10, 790]], dtype=np.float32)
        warp_rgb = np.zeros((800, 800, 3), dtype=np.uint8)
        warp_depth = np.zeros((800, 800), dtype=np.float32)
        result = BoardResult(corners=corners, warped_rgb=warp_rgb,
                            warped_depth=warp_depth, plane_depth=500.0)
        assert result.corners.shape == (4, 2)
        assert result.warped_rgb.shape == (800, 800, 3)
        assert result.plane_depth == 500.0
