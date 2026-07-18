import numpy as np
from src.types import CellData


class GridSplitter:
    def __init__(self, board_size: int = 800, roi_ratio: float = 0.6):
        self.board_size = board_size
        self.roi_ratio = roi_ratio

    def split(self, warped_rgb: np.ndarray, warped_depth: np.ndarray) -> list[CellData]:
        assert warped_rgb.shape[:2] == (self.board_size, self.board_size), \
            f"Expected RGB shape ({self.board_size}, {self.board_size}), got {warped_rgb.shape[:2]}"
        assert warped_depth.shape[:2] == (self.board_size, self.board_size), \
            f"Expected depth shape ({self.board_size}, {self.board_size}), got {warped_depth.shape[:2]}"
        cell_size = self.board_size // 8
        roi_size = int(cell_size * self.roi_ratio)
        roi_offset = (cell_size - roi_size) // 2

        cells = []
        for row in range(8):
            for col in range(8):
                y_start = row * cell_size + roi_offset
                x_start = col * cell_size + roi_offset
                rgb_patch = warped_rgb[y_start:y_start + roi_size,
                                        x_start:x_start + roi_size].copy()
                depth_patch = warped_depth[y_start:y_start + roi_size,
                                           x_start:x_start + roi_size].copy()
                cells.append(CellData(row=row, col=col,
                                      rgb_patch=rgb_patch,
                                      depth_patch=depth_patch))
        return cells
