import unittest

from pikachu.solver import BoardSolver


class BoardSolverTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
