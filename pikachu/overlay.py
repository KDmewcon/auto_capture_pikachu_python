from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .solver import PairPath

Coord = Tuple[int, int]


def render_connectable_overlay(
    frame: np.ndarray,
    pairs: Sequence[PairPath],
    rows: int,
    cols: int,
    max_lines: int = 12,
    board_left: int = 0,
    board_top: int = 0,
    board_width: Optional[int] = None,
    board_height: Optional[int] = None,
    draw_grid: bool = True,
) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        return frame.copy()

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    safe_left = max(0, min(int(board_left), frame_width - 1))
    safe_top = max(0, min(int(board_top), frame_height - 1))

    if board_width is None:
        safe_width = frame_width - safe_left
    else:
        safe_width = max(1, min(int(board_width), frame_width - safe_left))

    if board_height is None:
        safe_height = frame_height - safe_top
    else:
        safe_height = max(1, min(int(board_height), frame_height - safe_top))

    cell_width = safe_width / float(cols)
    cell_height = safe_height / float(rows)
    safe_right = safe_left + safe_width
    safe_bottom = safe_top + safe_height

    output = (frame.astype(np.float32) * 0.22).astype(np.uint8)

    cv2.rectangle(output, (safe_left, safe_top), (safe_right - 1, safe_bottom - 1), (90, 180, 245), 1)

    if draw_grid:
        for row in range(1, rows):
            y = int(round(safe_top + row * cell_height))
            y = max(safe_top, min(y, safe_bottom - 1))
            cv2.line(output, (safe_left, y), (safe_right - 1, y), (80, 80, 80), 1, cv2.LINE_AA)

        for col in range(1, cols):
            x = int(round(safe_left + col * cell_width))
            x = max(safe_left, min(x, safe_right - 1))
            cv2.line(output, (x, safe_top), (x, safe_bottom - 1), (80, 80, 80), 1, cv2.LINE_AA)

    connectable_cells: Set[Coord] = set()
    for pair in pairs:
        connectable_cells.add(pair.start)
        connectable_cells.add(pair.end)

    for row, col in connectable_cells:
        x0, y0, x1, y1 = _cell_rect(
            row,
            col,
            cell_width,
            cell_height,
            safe_left,
            safe_top,
            safe_width,
            safe_height,
            frame_width,
            frame_height,
        )
        output[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (74, 240, 105), 2)

    palette = [
        (77, 201, 255),
        (255, 172, 69),
        (91, 245, 186),
        (255, 125, 164),
        (170, 137, 255),
        (255, 219, 79),
    ]

    for index, pair in enumerate(pairs[: max(0, max_lines)]):
        color = palette[index % len(palette)]
        points = [
            _path_point_to_pixel(
                point,
                rows,
                cols,
                cell_width,
                cell_height,
                safe_left,
                safe_top,
                safe_width,
                safe_height,
                frame_width,
                frame_height,
            )
            for point in pair.path
        ]

        if len(points) < 2:
            continue

        for p1, p2 in zip(points, points[1:]):
            cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)

    cv2.putText(
        output,
        f"Connectable pairs: {len(pairs)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Bright = connectable, Dark = blocked, Green boxes = clickable, Lines = sample routes",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return output


def _cell_rect(
    row: int,
    col: int,
    cell_width: float,
    cell_height: float,
    board_left: int,
    board_top: int,
    board_width: int,
    board_height: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x0 = int(round(board_left + col * cell_width))
    x1 = int(round(board_left + (col + 1) * cell_width))
    y0 = int(round(board_top + row * cell_height))
    y1 = int(round(board_top + (row + 1) * cell_height))

    board_right = min(board_left + board_width, width)
    board_bottom = min(board_top + board_height, height)

    x0 = max(board_left, min(x0, board_right - 1))
    x1 = max(x0 + 1, min(x1, board_right))
    y0 = max(board_top, min(y0, board_bottom - 1))
    y1 = max(y0 + 1, min(y1, board_bottom))

    return x0, y0, x1, y1


def _path_point_to_pixel(
    point: Coord,
    rows: int,
    cols: int,
    cell_width: float,
    cell_height: float,
    board_left: int,
    board_top: int,
    board_width: int,
    board_height: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    row, col = point

    board_right = min(board_left + board_width, width) - 1
    board_bottom = min(board_top + board_height, height) - 1

    if row < 0:
        y = board_top
    elif row >= rows:
        y = board_bottom
    else:
        y = int(round(board_top + (row + 0.5) * cell_height))

    if col < 0:
        x = board_left
    elif col >= cols:
        x = board_right
    else:
        x = int(round(board_left + (col + 0.5) * cell_width))

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y
