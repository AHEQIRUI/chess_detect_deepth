import numpy as np

from src.camera import CameraBase
from src.board_detector import BoardDetector
from src.grid_splitter import GridSplitter
from src.piece_recognizer import PieceRecognizer
from src.fen_generator import FenGenerator


class ChessRecognizer:
    def __init__(self, camera: CameraBase | None = None,
                 board_size: int = 800, presence_threshold: float = 10.0):
        self.camera = camera
        self.detector = BoardDetector(output_size=board_size)
        self.splitter = GridSplitter(board_size=board_size)
        self.piece_recognizer = PieceRecognizer(presence_threshold=presence_threshold)
        self.fen_gen = FenGenerator()

    def run(self, corners: np.ndarray | None = None) -> str:
        if self.camera is None:
            raise RuntimeError("Camera not set. Call recognizer.camera = ... first.")
        rgb, depth = self.camera.capture()
        board = self.detector.detect(rgb, depth, corners=corners)
        cells = self.splitter.split(board.warped_rgb, board.warped_depth)
        pieces = self.piece_recognizer.recognize(cells, board.plane_depth)
        return self.fen_gen.generate(pieces)
