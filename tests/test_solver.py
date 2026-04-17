import unittest
from collections import deque
from random import Random
from typing import List, Optional, Sequence, Tuple

from pikachu.solver import BoardSolver

Coord = Tuple[int, int]


class BoardSolverTests(unittest.TestCase):
    @staticmethod
    def _reference_can_connect(board: Sequence[Sequence[int]], start: Coord, end: Coord) -> bool:
        if start == end:
            return False

        rows = len(board)
        cols = len(board[0])

        sr, sc = start
        er, ec = end
        if board[sr][sc] == 0 or board[sr][sc] != board[er][ec]:
            return False

        padded_rows = rows + 2
        padded_cols = cols + 2
        padded = [[0] * padded_cols for _ in range(padded_rows)]
        for row in range(rows):
            for col in range(cols):
                padded[row + 1][col + 1] = board[row][col]

        start_padded = (sr + 1, sc + 1)
        end_padded = (er + 1, ec + 1)
        directions: List[Coord] = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        queue: deque[Tuple[int, int, int, int]] = deque()
        best_turns: dict[Tuple[int, int, int], int] = {}

        for direction_index, (delta_row, delta_col) in enumerate(directions):
            next_row = start_padded[0] + delta_row
            next_col = start_padded[1] + delta_col
            if not (0 <= next_row < padded_rows and 0 <= next_col < padded_cols):
                continue
            if (next_row, next_col) != end_padded and padded[next_row][next_col] != 0:
                continue

            queue.append((next_row, next_col, direction_index, 0))
            best_turns[(next_row, next_col, direction_index)] = 0

        while queue:
            row, col, direction_index, turns = queue.popleft()
            if (row, col) == end_padded:
                return True

            for next_direction_index, (delta_row, delta_col) in enumerate(directions):
                next_turns = turns + (0 if next_direction_index == direction_index else 1)
                if next_turns > 2:
                    continue

                next_row = row + delta_row
                next_col = col + delta_col
                if not (0 <= next_row < padded_rows and 0 <= next_col < padded_cols):
                    continue
                if (next_row, next_col) != end_padded and padded[next_row][next_col] != 0:
                    continue

                state = (next_row, next_col, next_direction_index)
                previous_turns = best_turns.get(state)
                if previous_turns is not None and previous_turns <= next_turns:
                    continue

                best_turns[state] = next_turns
                queue.append((next_row, next_col, next_direction_index, next_turns))

        return False

    @staticmethod
    def _random_board(rng: Random, rows: int, cols: int, max_value: int, fill: float) -> List[List[int]]:
        board: List[List[int]] = []
        for _ in range(rows):
            row: List[int] = []
            for _ in range(cols):
                if rng.random() < fill:
                    row.append(rng.randint(1, max_value))
                else:
                    row.append(0)
            board.append(row)
        return board

    def test_direct_connection_same_row(self) -> None:
        board = [[1, 0, 1]]
        solver = BoardSolver(board)

        path = solver.can_connect((0, 0), (0, 2))

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (0, 2))
        self.assertEqual(max(0, len(path) - 2), 0)

    def test_one_turn_connection(self) -> None:
        board = [
            [1, 0],
            [0, 1],
        ]
        solver = BoardSolver(board)

        path = solver.can_connect((0, 0), (1, 1))

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (1, 1))
        self.assertEqual(max(0, len(path) - 2), 1)

    def test_two_turn_connection_via_border(self) -> None:
        board = [
            [1, 2, 1],
            [2, 2, 2],
            [0, 0, 0],
        ]
        solver = BoardSolver(board)

        path = solver.can_connect((0, 0), (0, 2))

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (0, 2))
        self.assertEqual(max(0, len(path) - 2), 2)

    def test_no_path_when_tile_is_surrounded(self) -> None:
        board = [
            [2, 2, 2, 2],
            [2, 1, 2, 1],
            [2, 2, 2, 2],
            [0, 0, 0, 0],
        ]
        solver = BoardSolver(board)

        path = solver.can_connect((1, 1), (1, 3))

        self.assertIsNone(path)

    def test_find_all_pairs(self) -> None:
        board = [
            [1, 0, 1],
            [2, 2, 0],
        ]
        solver = BoardSolver(board)

        pairs = solver.find_all_pairs()

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].tile_id, 2)
        self.assertEqual(pairs[1].tile_id, 1)

    def test_matches_reference_on_random_boards(self) -> None:
        rng = Random(7)

        for _ in range(80):
            board = self._random_board(rng, rows=5, cols=6, max_value=7, fill=0.72)
            solver = BoardSolver(board)

            positions = [
                (row, col)
                for row in range(len(board))
                for col in range(len(board[0]))
                if board[row][col] > 0
            ]

            for index, start in enumerate(positions):
                for end in positions[index + 1 :]:
                    if board[start[0]][start[1]] != board[end[0]][end[1]]:
                        continue

                    expected = self._reference_can_connect(board, start, end)
                    actual = solver.can_connect(start, end) is not None
                    self.assertEqual(
                        actual,
                        expected,
                        msg=f"Mismatch for start={start}, end={end}, board={board}",
                    )


if __name__ == "__main__":
    unittest.main()
