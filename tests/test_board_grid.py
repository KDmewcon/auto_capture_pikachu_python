import unittest

from pikachu.vision import BoardGrid


class BoardGridTests(unittest.TestCase):
    def test_from_cell_centers_builds_expected_grid(self) -> None:
        grid = BoardGrid.from_cell_centers(
            top_left_center=(100, 50),
            bottom_right_center=(550, 290),
            rows=9,
            cols=16,
            frame_width=900,
            frame_height=500,
        )

        self.assertEqual(grid.left, 85)
        self.assertEqual(grid.top, 35)
        self.assertEqual(grid.width, 480)
        self.assertEqual(grid.height, 270)

    def test_from_cell_centers_accepts_reverse_click_order(self) -> None:
        grid = BoardGrid.from_cell_centers(
            top_left_center=(550, 290),
            bottom_right_center=(100, 50),
            rows=9,
            cols=16,
            frame_width=900,
            frame_height=500,
        )

        self.assertEqual(grid.left, 85)
        self.assertEqual(grid.top, 35)
        self.assertEqual(grid.width, 480)
        self.assertEqual(grid.height, 270)

    def test_clamp_keeps_grid_inside_frame(self) -> None:
        clamped = BoardGrid(left=-20, top=10, width=200, height=120).clamp(150, 100)

        self.assertEqual(clamped.left, 0)
        self.assertEqual(clamped.top, 0)
        self.assertEqual(clamped.width, 150)
        self.assertEqual(clamped.height, 100)

    def test_invalid_close_points_raise(self) -> None:
        with self.assertRaises(ValueError):
            BoardGrid.from_cell_centers(
                top_left_center=(200, 200),
                bottom_right_center=(201, 201),
                rows=9,
                cols=16,
                frame_width=800,
                frame_height=600,
            )


if __name__ == "__main__":
    unittest.main()
