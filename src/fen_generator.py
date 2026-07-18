from src.types import Color, PieceResult


class FenGenerator:
    def generate(self, pieces: list[PieceResult]) -> str:
        # Build 8x8 grid, row 0 = rank 8 (top of image, black's home)
        grid = [[None] * 8 for _ in range(8)]
        for p in pieces:
            grid[p.row][p.col] = p

        ranks = []
        for row in range(8):
            rank_str = ""
            empty_count = 0
            for col in range(8):
                piece = grid[row][col]
                if piece is None or piece.piece_type is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += self._piece_to_fen_char(piece)
            if empty_count > 0:
                rank_str += str(empty_count)
            ranks.append(rank_str)

        board = "/".join(ranks)
        return f"{board} w KQkq - 0 1"

    def _piece_to_fen_char(self, piece: PieceResult) -> str:
        ch = piece.piece_type.value
        return ch if piece.color == Color.WHITE else ch.lower()
