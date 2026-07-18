import numpy as np
from src.grid_splitter import GridSplitter


class TestGridSplitter:
    def test_produces_64_cells(self):
        rgb = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        depth = np.random.randint(400, 600, (800, 800), dtype=np.uint16)
        splitter = GridSplitter(board_size=800)
        cells = splitter.split(rgb, depth)
        assert len(cells) == 64

    def test_all_positions_present(self):
        rgb = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        depth = np.random.randint(400, 600, (800, 800), dtype=np.uint16)
        splitter = GridSplitter(board_size=800)
        cells = splitter.split(rgb, depth)
        positions = {(c.row, c.col) for c in cells}
        assert len(positions) == 64
        for r in range(8):
            for c in range(8):
                assert (r, c) in positions

    def test_cell_dimensions(self):
        rgb = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        depth = np.random.randint(400, 600, (800, 800), dtype=np.uint16)
        splitter = GridSplitter(board_size=800, roi_ratio=0.6)
        cells = splitter.split(rgb, depth)
        cell = cells[0]
        cell_size = 800 // 8  # 100
        expected_roi = int(cell_size * 0.6)  # 60
        assert cell.rgb_patch.shape == (expected_roi, expected_roi, 3)

    def test_top_left_cell_matches_image_region(self):
        rgb = np.zeros((800, 800, 3), dtype=np.uint8)
        rgb[5:45, 5:45] = [255, 0, 0]  # mark top-left cell center
        depth = np.zeros((800, 800), dtype=np.uint16)
        splitter = GridSplitter(board_size=800, roi_ratio=0.4)
        cells = splitter.split(rgb, depth)
        top_left = [c for c in cells if c.row == 0 and c.col == 0][0]
        # Center of the ROI should have the red mark
        assert top_left.rgb_patch.mean() > 10

    def test_row_order_is_top_to_bottom(self):
        """Row 0 should be top of image (rank 8 in chess)"""
        rgb = np.zeros((800, 800, 3), dtype=np.uint8)
        # Mark top row of cells white, bottom row black
        rgb[0:100, :] = [255, 255, 255]
        rgb[700:800, :] = [0, 0, 0]
        depth = np.zeros((800, 800), dtype=np.uint16)
        splitter = GridSplitter(board_size=800, roi_ratio=1.0)
        cells = splitter.split(rgb, depth)
        for c in cells:
            val = c.rgb_patch[:, :, 0].mean()
            if c.row == 0:
                assert val > 200, f"Row 0 should be white, got {val}"
            elif c.row == 7:
                assert val < 10, f"Row 7 should be black, got {val}"
