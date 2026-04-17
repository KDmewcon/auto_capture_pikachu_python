import unittest

import numpy as np

from pikachu.overlay import render_connectable_overlay
from pikachu.solver import PairPath


class OverlayRenderTests(unittest.TestCase):
    def test_connectable_cells_are_bright_and_blocked_cells_are_dim(self) -> None:
        frame = np.full((90, 160, 3), 200, dtype=np.uint8)
        pair = PairPath(tile_id=1, start=(0, 0), end=(0, 1), path=[(0, 0), (0, 1)])

        overlay = render_connectable_overlay(
            frame=frame,
            pairs=[pair],
            rows=3,
            cols=4,
            max_lines=0,
        )

        bright_pixel = overlay[15, 20].mean()
        dim_pixel = overlay[75, 140].mean()

        self.assertGreater(bright_pixel, 150.0)
        self.assertLess(dim_pixel, 80.0)


if __name__ == "__main__":
    unittest.main()
