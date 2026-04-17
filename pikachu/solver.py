from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int]
Path = List[Coord]


@dataclass(frozen=True)
class PairPath:
    tile_id: int
    start: Coord
    end: Coord
    path: Path

    @property
    def turns(self) -> int:
        return max(0, len(self.path) - 2)

    @property
    def path_distance(self) -> int:
        total = 0
        for (r1, c1), (r2, c2) in zip(self.path, self.path[1:]):
            total += abs(r1 - r2) + abs(c1 - c2)
        return total


class BoardSolver:
    def __init__(self, board: Sequence[Sequence[int]]):
        if not board or not board[0]:
            raise ValueError("Board must not be empty.")

        self.board = [list(row) for row in board]
        self.rows = len(self.board)
        self.cols = len(self.board[0])

        for row in self.board:
            if len(row) != self.cols:
                raise ValueError("Board rows must have the same length.")

    def find_all_pairs(self, max_pairs: Optional[int] = None) -> List[PairPath]:
        grouped: DefaultDict[int, List[Coord]] = defaultdict(list)
        for r, row in enumerate(self.board):
            for c, value in enumerate(row):
                if value > 0:
                    grouped[value].append((r, c))

        pairs: List[PairPath] = []
        for tile_id, positions in grouped.items():
            for start, end in combinations(positions, 2):
                path = self.can_connect(start, end)
                if path is None:
                    continue
                pairs.append(PairPath(tile_id=tile_id, start=start, end=end, path=path))

        pairs.sort(
            key=lambda item: (
                item.turns,
                item.path_distance,
                item.start[0],
                item.start[1],
                item.end[0],
                item.end[1],
            )
        )

        if max_pairs is None:
            return pairs
        return pairs[:max_pairs]

    def can_connect(self, start: Coord, end: Coord) -> Optional[Path]:
        if start == end:
            return None

        if not self._inside(start) or not self._inside(end):
            return None

        start_value = self.board[start[0]][start[1]]
        if start_value == 0:
            return None

        if self.board[end[0]][end[1]] != start_value:
            return None

        padded = self._make_padded_board()
        padded_start = (start[0] + 1, start[1] + 1)
        padded_end = (end[0] + 1, end[1] + 1)
        allowed = {padded_start, padded_end}

        path = self._zero_or_one_turn(padded, padded_start, padded_end, allowed)
        if path is None:
            path = self._two_turns(padded, padded_start, padded_end, allowed)

        if path is None:
            return None

        return [self._to_unpadded(point) for point in path]

    def _inside(self, point: Coord) -> bool:
        return 0 <= point[0] < self.rows and 0 <= point[1] < self.cols

    def _make_padded_board(self) -> List[List[int]]:
        padded_rows = self.rows + 2
        padded_cols = self.cols + 2
        padded = [[0] * padded_cols for _ in range(padded_rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                padded[r + 1][c + 1] = self.board[r][c]
        return padded

    @staticmethod
    def _to_unpadded(point: Coord) -> Coord:
        return point[0] - 1, point[1] - 1

    @staticmethod
    def _cell_open(board: Sequence[Sequence[int]], point: Coord, allowed: Set[Coord]) -> bool:
        return point in allowed or board[point[0]][point[1]] == 0

    def _line_clear(
        self,
        board: Sequence[Sequence[int]],
        a: Coord,
        b: Coord,
        allowed: Set[Coord],
    ) -> bool:
        if a[0] == b[0]:
            row = a[0]
            left = min(a[1], b[1]) + 1
            right = max(a[1], b[1])
            for col in range(left, right):
                if not self._cell_open(board, (row, col), allowed):
                    return False
            return True

        if a[1] == b[1]:
            col = a[1]
            top = min(a[0], b[0]) + 1
            bottom = max(a[0], b[0])
            for row in range(top, bottom):
                if not self._cell_open(board, (row, col), allowed):
                    return False
            return True

        return False

    def _zero_or_one_turn(
        self,
        board: Sequence[Sequence[int]],
        a: Coord,
        b: Coord,
        allowed: Set[Coord],
    ) -> Optional[Path]:
        if self._line_clear(board, a, b, allowed):
            return [a, b]

        corners = ((a[0], b[1]), (b[0], a[1]))
        for corner in corners:
            if corner == a or corner == b:
                continue
            if not self._cell_open(board, corner, allowed):
                continue
            if self._line_clear(board, a, corner, allowed) and self._line_clear(
                board, corner, b, allowed
            ):
                return [a, corner, b]

        return None

    def _ray_points(
        self,
        board: Sequence[Sequence[int]],
        origin: Coord,
        allowed: Set[Coord],
    ) -> Iterable[Coord]:
        max_row = len(board)
        max_col = len(board[0])

        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            row = origin[0] + delta_row
            col = origin[1] + delta_col
            while 0 <= row < max_row and 0 <= col < max_col:
                point = (row, col)
                if not self._cell_open(board, point, allowed):
                    break
                yield point
                row += delta_row
                col += delta_col

    def _two_turns(
        self,
        board: Sequence[Sequence[int]],
        a: Coord,
        b: Coord,
        allowed: Set[Coord],
    ) -> Optional[Path]:
        for pivot in self._ray_points(board, a, allowed):
            if pivot == b:
                continue
            tail = self._zero_or_one_turn(board, pivot, b, allowed)
            if tail is None:
                continue
            return [a, pivot] + tail[1:]
        return None
