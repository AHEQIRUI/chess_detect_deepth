import numpy as np
from src.game_tracker import GameTracker
from src.types import CellData


def _make_presence(layout):
    p = np.zeros((8, 8), dtype=bool)
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            if ch != '.':
                p[r, c] = True
    return p


def _make_colors(layout):
    """layout: 'W'=white (relative V>0), 'b'=black (relative V<0), .=no piece."""
    b = np.full((8, 8), -1.0)
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            if ch == 'W':
                b[r, c] = 50.0
            elif ch == 'b':
                b[r, c] = -50.0
    return b


def _calibrate(tracker, pres, col):
    """Helper: probe and accept calibration in one step."""
    m = tracker.probe(pres, col)
    assert m == "?calibrate"
    tracker.accept()


class TestGameTracker:
    def test_initial_fen(self):
        tracker = GameTracker()
        assert tracker.get_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq"

    def test_fen_side_to_move_after_move(self):
        tracker = GameTracker(expected_initial_count=None)
        # Use standard opening: move e2→e4 (white pawn)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "........","........","PPPPPPPP","RNBQKBNR"]),
            _make_colors(["........","........","........","........",
                          "........","........","WWWWWWWW","WWWWWWWW"]))
        tracker.probe(
            _make_presence(["........","........","........","........",
                            "....P...","........","PPP.PPPP","RNBQKBNR"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","WWWWWWWW","WWWWWWWW"]))
        tracker.accept()
        fen = tracker.get_fen()
        assert "4P3" in fen  # rank 4 has pawn on e4
        assert " b KQkq" in fen

    def test_fen_reflects_actual_position(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "........","........","PPPPPPPP","RNBQKBNR"]),
            _make_colors(["........","........","........","........",
                          "........","........","WWWWWWWW","WWWWWWWW"]))
        # e2→e4: white pawn
        tracker.probe(
            _make_presence(["........","........","........","........",
                            "....P...","........","PPP.PPPP","RNBQKBNR"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","WWWWWWWW","WWWWWWWW"]))
        tracker.accept()
        # g1→f3: white knight
        tracker.probe(
            _make_presence(["........","........","........","........",
                            "....P...",".....N..","PPP.PPPP","RNBQKB.R"]),
            _make_colors(["........","........","........","........",
                          "....W...",".....W..","WWWWWWWW","WWWWWWWW"]))
        tracker.accept()
        fen = tracker.get_fen()
        # After e4 and Nf3, FEN should reflect actual position
        assert "4P3" in fen  # rank 4 has pawn on e4
        assert "5N2" in fen  # rank 3 has knight on f3

    def test_first_probe_calibrates(self):
        tracker = GameTracker(expected_initial_count=None)
        layout = ["pppppppp","pppppppp","........","........",
                  "........","........","PPPPPPPP","PPPPPPPP"]
        pres = _make_presence(layout)
        col = _make_colors(layout)
        move = tracker.probe(pres, col)
        assert move == "?calibrate"
        assert not tracker.is_calibrated
        tracker.accept()
        assert tracker.is_calibrated

    def test_calibration_rejects_wrong_piece_count(self):
        tracker = GameTracker()
        pres = _make_presence([
            "........","........","........","........",
            "........","........","PPPPPPPP","........"])
        col = _make_colors([
            "........","........","........","........",
            "........","........","WWWWWWWW","........"])
        move = tracker.probe(pres, col)
        assert move == "?init:8"
        assert not tracker.is_calibrated

    def test_probe_returns_calibrate_not_auto_commit(self):
        tracker = GameTracker(expected_initial_count=None)
        pres = _make_presence([
            "........","........","........","........",
            "....p...","........","........","........"])
        col = _make_colors([
            "........","........","........","........",
            "....W...","........","........","........"])
        move = tracker.probe(pres, col)
        assert move == "?calibrate"
        assert not tracker.is_calibrated

    def test_move_detected(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "....p...","........","........","........"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","........","........"]))
        move = tracker.probe(
            _make_presence(["........","........","........","....p...",
                            "........","........","........","........"]),
            _make_colors(["........","........","........","....W...",
                          "........","........","........","........"]))
        assert move == "?e4e5"
        assert tracker.has_pending
        tracker.accept()
        assert not tracker.has_pending
        assert len(tracker.move_history) == 1

    def test_reject_keeps_state(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "....p...","........","........","........"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","........","........"]))
        tracker.probe(
            _make_presence(["........","........","........","....p...",
                            "........","........","........","........"]),
            _make_colors(["........","........","........","....W...",
                          "........","........","........","........"]))
        tracker.reject()
        assert not tracker.has_pending
        assert len(tracker.move_history) == 0

    def test_two_moves(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "....p...","........","........","........"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","........","........"]))
        tracker.probe(
            _make_presence(["........","........","........","....p...",
                            "........","........","........","........"]),
            _make_colors(["........","........","........","....W...",
                          "........","........","........","........"]))
        tracker.accept()
        assert len(tracker.move_history) == 1
        tracker.probe(
            _make_presence(["........","........","....p...","........",
                            "........","........","........","........"]),
            _make_colors(["........","........","....W...","........",
                          "........","........","........","........"]))
        tracker.accept()
        assert len(tracker.move_history) == 2

    def test_no_change_detected(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "....p...","........","........","........"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","........","........"]))
        move = tracker.probe(
            _make_presence(["........","........","........","........",
                            "....p...","........","........","........"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","........","........"]))
        assert move is None
        assert not tracker.has_pending

    def test_capture_by_color_change(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence([
                "........","........","........",
                "....P...","....p...","........","........","........"]),
            _make_colors([
                "........","........","........",
                "....W...","....b...","........","........","........"]))
        move = tracker.probe(
            _make_presence([
                "........","........","........",
                "....P...","........","........","........","........"]),
            _make_colors([
                "........","........","........",
                "....b...","........","........","........","........"]))
        assert move == "?e4e5:capture"

    def test_reset(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "....p...","........","........","........"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","........","........"]))
        tracker.reset()
        assert not tracker.is_calibrated
        assert tracker.move_history == []

    def test_side_to_move_alternates(self):
        tracker = GameTracker(expected_initial_count=None)
        _calibrate(tracker,
            _make_presence(["........","........","........","........",
                            "........","........","........","....p..."]),
            _make_colors(["........","........","........","........",
                          "........","........","........","....W..."]))
        tracker.probe(
            _make_presence(["........","........","........","........",
                            "........","....p...","........","........"]),
            _make_colors(["........","........","........","........",
                          "........","....W...","........","........"]))
        tracker.accept()
        assert tracker.side_to_move == 'b'
        tracker.probe(
            _make_presence(["........","........","........","........",
                            "....p...","........","........","........"]),
            _make_colors(["........","........","........","........",
                          "....W...","........","........","........"]))
        tracker.accept()
        assert tracker.side_to_move == 'w'
