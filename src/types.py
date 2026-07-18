from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class PieceType(Enum):
    KING = 'K'
    QUEEN = 'Q'
    ROOK = 'R'
    BISHOP = 'B'
    KNIGHT = 'N'
    PAWN = 'P'


class Color(Enum):
    WHITE = 'white'
    BLACK = 'black'


@dataclass(eq=False)
class CellData:
    row: int
    col: int
    rgb_patch: np.ndarray
    depth_patch: np.ndarray


@dataclass
class PieceResult:
    row: int
    col: int
    piece_type: Optional[PieceType]
    color: Optional[Color]


@dataclass(eq=False)
class BoardResult:
    corners: np.ndarray
    warped_rgb: np.ndarray
    warped_depth: np.ndarray
    plane_depth: float
