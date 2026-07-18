"""Game state tracker: starts from standard opening, detects moves via frame diff."""

import cv2
import numpy as np
from src.types import PieceType, Color

INITIAL_BOARD = [
    ["r","n","b","q","k","b","n","r"],
    ["p","p","p","p","p","p","p","p"],
    [None]*8,
    [None]*8,
    [None]*8,
    [None]*8,
    ["P","P","P","P","P","P","P","P"],
    ["R","N","B","Q","K","B","N","R"],
]


def _uci(row, col):
    return f"{chr(ord('a')+col)}{8-row}"


class GameTracker:
    """Tracks chess game state from frame-to-frame piece presence changes.

    Two-step detection: probe() detects a move without committing,
    then user calls accept() or reject() to confirm.

    Captures are detected by color change on occupied squares: if a
    piece disappeared from A and nothing new appeared, but square B's
    piece color flipped (white↔black), then A→B is a capture.
    """

    def __init__(self, presence_threshold: float = 25.0,
                 brightness_threshold: float = 0.0,
                 expected_initial_count: int | None = 32):
        self.presence_threshold = presence_threshold
        self.brightness_threshold = brightness_threshold
        self.expected_initial_count = expected_initial_count
        self.board = [row[:] for row in INITIAL_BOARD]
        self.move_history = []
        self.side_to_move = 'w'
        self.prev_presence = None
        self.prev_colors = None
        self._pending = None
        self._ref_piece_count = 0

    def reset(self):
        self.board = [row[:] for row in INITIAL_BOARD]
        self.move_history = []
        self.side_to_move = 'w'
        self.prev_presence = None
        self.prev_colors = None
        self._pending = None
        self._ref_piece_count = 0

    def get_presence_and_colors(self, cells, board_plane_depth):
        """Return (presence 8x8 bool, brightness 8x8 float).

        Brightness is relative: piece V minus board-surface V within the same cell.
        Positive = piece brighter than square (white), negative = darker (black).
        This accounts for light vs dark board squares.
        """
        presence = np.zeros((8, 8), dtype=bool)
        brightness = np.full((8, 8), -1.0)
        for cell in cells:
            d = cell.depth_patch
            valid_depth = (d > board_plane_depth - 150) & (d < board_plane_depth + 50)
            if not valid_depth.any():
                continue
            h = board_plane_depth - np.min(d[valid_depth])
            if h > self.presence_threshold:
                presence[cell.row, cell.col] = True
                hsv = cv2.cvtColor(cell.rgb_patch, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2]
                # Board surface: depth within 5mm of plane
                board_mask = (d > board_plane_depth - 5) & (d < board_plane_depth + 5)
                # Piece pixels: above board
                piece_mask = (board_plane_depth - d) > self.presence_threshold
                if board_mask.any() and piece_mask.any():
                    board_v = np.mean(v[board_mask])
                    piece_v = np.mean(v[piece_mask & valid_depth])
                    brightness[cell.row, cell.col] = piece_v - board_v
        return presence, brightness

    def probe(self, presence, colors):
        """Detect move from presence+color diff. Does NOT commit.

        Returns None on first call (calibration). On subsequent calls,
        returns "{piece_char}{uci}" or None if no change detected.
        """
        if self.prev_presence is None:
            count = int(np.sum(presence))
            if self.expected_initial_count is not None \
               and count != self.expected_initial_count:
                return f"?init:{count}"
            # Store as pending for user confirmation
            self._pending = (presence, colors, "?calibrate", None, None)
            return "?calibrate"

        appeared = []
        disappeared = []
        for r in range(8):
            for c in range(8):
                if presence[r, c] and not self.prev_presence[r, c]:
                    appeared.append((r, c))
                elif not presence[r, c] and self.prev_presence[r, c]:
                    disappeared.append((r, c))

        cur_count = int(np.sum(presence))
        ref_count = self._ref_piece_count
        move = None
        src = dst = None

        if len(appeared) == 1 and len(disappeared) == 1:
            if cur_count != ref_count:
                return None
            src = disappeared[0]
            dst = appeared[0]
            move = _uci(*src) + _uci(*dst)
        elif len(appeared) == 1 and len(disappeared) >= 2:
            if cur_count != ref_count - (len(disappeared) - 1):
                return None
            dst = appeared[0]
            for d in disappeared:
                if d != dst:
                    src = d
                    break
            if src is not None:
                move = _uci(*src) + _uci(*dst)
        elif len(disappeared) == 1 and len(appeared) == 0:
            # Capture: 1 piece disappeared, 0 appeared, count decreased by 1
            if cur_count != ref_count - 1:
                return None
            dst = self._find_color_change(presence, colors)
            if dst is not None:
                src = disappeared[0]
                move = _uci(*src) + _uci(*dst)

        if move is not None:
            piece = self.board[src[0]][src[1]]
            move = (piece if piece else "?") + move
            # Mark captures with a flag
            if len(appeared) == 0 and len(disappeared) == 1:
                move += ":capture"
            self._pending = (presence, colors, move, src, dst)
        else:
            self._pending = None
        return move

    def _find_color_change(self, presence, colors):
        """Find a square where presence stayed True but brightness changed significantly."""
        if self.prev_colors is None:
            return None
        best = None
        best_delta = 0
        for r in range(8):
            for c in range(8):
                if presence[r, c] and self.prev_presence[r, c]:
                    prev_v = self.prev_colors[r, c]
                    cur_v = colors[r, c]
                    if prev_v == -1.0 or cur_v == -1.0:
                        continue
                    delta = abs(cur_v - prev_v)
                    # Significant brightness change: >20 in HSV V units (relative)
                    if delta > 20 and delta > best_delta:
                        best_delta = delta
                        best = (r, c)
        return best

    def accept(self):
        if self._pending is None:
            return
        presence, colors, move, src, dst = self._pending
        if move == "?calibrate":
            self.prev_presence = presence
            self.prev_colors = colors
            self._ref_piece_count = int(np.sum(presence))
        else:
            if src is not None and dst is not None:
                self._apply_move(src, dst)
            # Strip piece prefix and capture suffix for history
            uci = move
            if uci.endswith(":capture"):
                uci = uci[:-8]
            uci = uci[-4:]  # last 4 chars = UCI
            self.move_history.append(uci)
            self.side_to_move = 'b' if self.side_to_move == 'w' else 'w'
            self.prev_presence = presence
            self.prev_colors = colors
            self._ref_piece_count = int(np.sum(presence))
        self._pending = None

    def reject(self):
        self._pending = None

    @property
    def has_pending(self):
        return self._pending is not None

    @property
    def is_calibrated(self):
        return self.prev_presence is not None

    def _apply_move(self, src, dst):
        piece = self.board[src[0]][src[1]]
        self.board[src[0]][src[1]] = None
        self.board[dst[0]][dst[1]] = piece

    def get_fen(self):
        ranks = []
        for row in range(8):
            rank = ""
            empty = 0
            for col in range(8):
                p = self.board[row][col]
                if p is None:
                    empty += 1
                else:
                    if empty:
                        rank += str(empty)
                        empty = 0
                    rank += p
            if empty:
                rank += str(empty)
            ranks.append(rank)
        board_str = "/".join(ranks)
        return f"{board_str} {self.side_to_move} KQkq"
