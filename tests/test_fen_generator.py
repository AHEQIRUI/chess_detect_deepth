import random

from src.types import PieceType, Color, PieceResult
from src.fen_generator import FenGenerator


def _make_piece(row, col, piece_type, color):
    return PieceResult(row=row, col=col, piece_type=piece_type, color=color)


def _make_empty(row, col):
    return PieceResult(row=row, col=col, piece_type=None, color=None)


def _build_board(layout):
    """layout is a list of 8 strings, each 8 chars. Uppercase=white, lowercase=black, .=empty"""
    char_to_piece = {
        'K': (PieceType.KING, Color.WHITE), 'k': (PieceType.KING, Color.BLACK),
        'Q': (PieceType.QUEEN, Color.WHITE), 'q': (PieceType.QUEEN, Color.BLACK),
        'R': (PieceType.ROOK, Color.WHITE), 'r': (PieceType.ROOK, Color.BLACK),
        'B': (PieceType.BISHOP, Color.WHITE), 'b': (PieceType.BISHOP, Color.BLACK),
        'N': (PieceType.KNIGHT, Color.WHITE), 'n': (PieceType.KNIGHT, Color.BLACK),
        'P': (PieceType.PAWN, Color.WHITE), 'p': (PieceType.PAWN, Color.BLACK),
    }
    pieces = []
    for row_idx, row_str in enumerate(layout):
        for col_idx, ch in enumerate(row_str):
            if ch == '.':
                pieces.append(_make_empty(row_idx, col_idx))
            else:
                pt, c = char_to_piece[ch]
                pieces.append(_make_piece(row_idx, col_idx, pt, c))
    return pieces


class TestFenGenerator:
    def test_initial_position(self):
        layout = [
            "rnbqkbnr",
            "pppppppp",
            "........",
            "........",
            "........",
            "........",
            "PPPPPPPP",
            "RNBQKBNR",
        ]
        pieces = _build_board(layout)
        gen = FenGenerator()
        fen = gen.generate(pieces)
        assert fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_empty_board(self):
        layout = ["........"] * 8
        pieces = _build_board(layout)
        gen = FenGenerator()
        fen = gen.generate(pieces)
        assert fen == "8/8/8/8/8/8/8/8 w KQkq - 0 1"

    def test_mixed_position(self):
        layout = [
            "....k...",
            "........",
            "....Q...",
            "........",
            "........",
            "....K...",
            "........",
            "........",
        ]
        pieces = _build_board(layout)
        gen = FenGenerator()
        fen = gen.generate(pieces)
        assert fen == "4k3/8/4Q3/8/8/4K3/8/8 w KQkq - 0 1"

    def test_pieces_sorted_by_position(self):
        """Pieces in random order should still produce correct FEN"""
        layout = [
            "r......r",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            "R......R",
        ]
        pieces = _build_board(layout)
        random.shuffle(pieces)
        gen = FenGenerator()
        fen = gen.generate(pieces)
        assert fen == "r6r/8/8/8/8/8/8/R6R w KQkq - 0 1"
