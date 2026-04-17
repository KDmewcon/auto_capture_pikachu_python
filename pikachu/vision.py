from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int
    monitor_index: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "monitor_index": self.monitor_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "CaptureRegion":
        return cls(
            left=int(data["left"]),
            top=int(data["top"]),
            width=int(data["width"]),
            height=int(data["height"]),
            monitor_index=int(data["monitor_index"]),
        )


@dataclass
class BoardGrid:
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "BoardGrid":
        return cls(
            left=int(data["left"]),
            top=int(data["top"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )

    @classmethod
    def full_region(cls, width: int, height: int) -> "BoardGrid":
        return cls(left=0, top=0, width=max(1, int(width)), height=max(1, int(height)))

    @classmethod
    def from_cell_centers(
        cls,
        top_left_center: Tuple[int, int],
        bottom_right_center: Tuple[int, int],
        rows: int,
        cols: int,
        frame_width: int,
        frame_height: int,
    ) -> "BoardGrid":
        if rows <= 0 or cols <= 0:
            raise ValueError("Rows and cols must be positive.")

        x1, y1 = top_left_center
        x2, y2 = bottom_right_center

        x_left = min(int(x1), int(x2))
        x_right = max(int(x1), int(x2))
        y_top = min(int(y1), int(y2))
        y_bottom = max(int(y1), int(y2))

        if cols > 1:
            cell_width = (x_right - x_left) / float(cols - 1)
        else:
            cell_width = float(frame_width)

        if rows > 1:
            cell_height = (y_bottom - y_top) / float(rows - 1)
        else:
            cell_height = float(frame_height)

        if cell_width < 2.0 or cell_height < 2.0:
            raise ValueError("Selected cell centers are too close to estimate a reliable grid.")

        board_left = int(round(x_left - (cell_width / 2.0)))
        board_top = int(round(y_top - (cell_height / 2.0)))
        board_width = int(round(cell_width * cols))
        board_height = int(round(cell_height * rows))

        return cls(
            left=board_left,
            top=board_top,
            width=max(1, board_width),
            height=max(1, board_height),
        ).clamp(frame_width, frame_height)

    def clamp(self, max_width: int, max_height: int) -> "BoardGrid":
        safe_max_width = max(1, int(max_width))
        safe_max_height = max(1, int(max_height))

        width = max(1, min(int(self.width), safe_max_width))
        height = max(1, min(int(self.height), safe_max_height))

        left = max(0, min(int(self.left), safe_max_width - width))
        top = max(0, min(int(self.top), safe_max_height - height))

        return BoardGrid(left=left, top=top, width=width, height=height)


@dataclass
class BoardScanResult:
    board: List[List[int]]
    cell_width: float
    cell_height: float
    tile_count: int


@dataclass
class TileObservation:
    row: int
    col: int
    feature: np.ndarray
    gray_signature: np.ndarray


class ScreenCapturer:
    def __init__(self) -> None:
        self._sct = mss.mss()

    def list_monitors(self) -> List[Dict[str, int]]:
        monitors = []
        for index, monitor in enumerate(self._sct.monitors[1:], start=1):
            monitors.append(
                {
                    "index": index,
                    "left": int(monitor["left"]),
                    "top": int(monitor["top"]),
                    "width": int(monitor["width"]),
                    "height": int(monitor["height"]),
                }
            )
        return monitors

    def capture_monitor(self, monitor_index: int) -> np.ndarray:
        monitor = self._monitor_by_index(monitor_index)
        raw = self._sct.grab(monitor)
        return self._to_bgr(raw)

    def capture_region(self, region: CaptureRegion) -> np.ndarray:
        raw = self._sct.grab(
            {
                "left": int(region.left),
                "top": int(region.top),
                "width": int(region.width),
                "height": int(region.height),
            }
        )
        return self._to_bgr(raw)

    def select_region(
        self,
        monitor_index: int,
        window_name: str = "Pick Pikachu Board (press ENTER)",
    ) -> Optional[CaptureRegion]:
        monitor = self._monitor_by_index(monitor_index)
        frame = self.capture_monitor(monitor_index)
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        x, y, width, height = [int(value) for value in roi]
        if width <= 0 or height <= 0:
            return None

        return CaptureRegion(
            left=int(monitor["left"] + x),
            top=int(monitor["top"] + y),
            width=width,
            height=height,
            monitor_index=monitor_index,
        )

    def select_board_grid(
        self,
        region: CaptureRegion,
        rows: int,
        cols: int,
        window_name: str = "Calibrate Pikachu Grid",
    ) -> Optional[BoardGrid]:
        frame = self.capture_region(region)
        if frame.size == 0:
            return None

        points: List[Tuple[int, int]] = []
        cursor: Optional[Tuple[int, int]] = None
        warning_text: Optional[str] = None

        def redraw() -> None:
            canvas = frame.copy()

            preview_points = list(points)
            if len(preview_points) == 1 and cursor is not None:
                preview_points.append(cursor)

            if len(preview_points) == 2:
                try:
                    preview_grid = BoardGrid.from_cell_centers(
                        preview_points[0],
                        preview_points[1],
                        rows,
                        cols,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )
                    self._draw_grid_preview(canvas, preview_grid, rows, cols)
                except ValueError:
                    pass

            for index, point in enumerate(points):
                cv2.circle(canvas, point, 5, (0, 230, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (0, 230, 255),
                    2,
                    cv2.LINE_AA,
                )

            hints = [
                "1) Click center of top-left tile",  # tile [1,1]
                "2) Click center of bottom-right tile",  # tile [rows,cols]
                "Enter/Space: confirm | R: reset | Right click: undo | C/Esc: cancel",
            ]
            y = 26
            for hint in hints:
                cv2.putText(
                    canvas,
                    hint,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                y += 24

            if warning_text:
                cv2.putText(
                    canvas,
                    warning_text,
                    (12, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (60, 150, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, canvas)

        def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
            nonlocal cursor, warning_text
            cursor = (int(x), int(y))
            warning_text = None

            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) == 2:
                    points.clear()
                points.append((int(x), int(y)))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

            redraw()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
        redraw()

        result: Optional[BoardGrid] = None
        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10, 32):
                if len(points) != 2:
                    warning_text = "Please click both cell centers before confirming."
                    redraw()
                    continue

                try:
                    result = BoardGrid.from_cell_centers(
                        points[0],
                        points[1],
                        rows,
                        cols,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )
                    break
                except ValueError as exc:
                    warning_text = str(exc)
                    redraw()
                    continue

            if key in (ord("r"),):
                points.clear()
                warning_text = None
                redraw()
                continue

            if key in (ord("c"), 27):
                result = None
                break

        cv2.destroyWindow(window_name)
        return result

    def close(self) -> None:
        self._sct.close()

    def _monitor_by_index(self, monitor_index: int) -> Dict[str, int]:
        monitors = self.list_monitors()
        for monitor in monitors:
            if monitor["index"] == monitor_index:
                return monitor
        raise ValueError(f"Monitor index {monitor_index} does not exist.")

    @staticmethod
    def _to_bgr(raw: mss.base.ScreenShot) -> np.ndarray:
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def _draw_grid_preview(frame: np.ndarray, grid: BoardGrid, rows: int, cols: int) -> None:
        x0 = int(grid.left)
        y0 = int(grid.top)
        x1 = int(grid.left + grid.width - 1)
        y1 = int(grid.top + grid.height - 1)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 235, 130), 2)

        if rows > 1:
            for row in range(1, rows):
                y = int(round(grid.top + row * (grid.height / float(rows))))
                cv2.line(frame, (x0, y), (x1, y), (70, 235, 130), 1, cv2.LINE_AA)

        if cols > 1:
            for col in range(1, cols):
                x = int(round(grid.left + col * (grid.width / float(cols))))
                cv2.line(frame, (x, y0), (x, y1), (70, 235, 130), 1, cv2.LINE_AA)


class BoardAnalyzer:
    def __init__(
        self,
        rows: int,
        cols: int,
        empty_edge_threshold: float = 0.045,
        empty_variance_threshold: float = 135.0,
        empty_saturation_threshold: float = 45.0,
        empty_ink_threshold: float = 0.16,
        template_similarity: float = 0.9,
        template_ambiguity_margin: float = 0.015,
        template_update_weight: float = 0.2,
        inner_crop_ratio: float = 0.16,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("Rows and cols must be positive integers.")

        self.rows = rows
        self.cols = cols
        self.empty_edge_threshold = empty_edge_threshold
        self.empty_variance_threshold = empty_variance_threshold
        self.empty_saturation_threshold = empty_saturation_threshold
        self.empty_ink_threshold = empty_ink_threshold
        self.template_similarity = template_similarity
        self.template_ambiguity_margin = template_ambiguity_margin
        self.template_update_weight = max(0.0, min(template_update_weight, 1.0))
        self.inner_crop_ratio = inner_crop_ratio

    def analyze(self, frame: np.ndarray) -> BoardScanResult:
        height, width = frame.shape[:2]
        cell_width = width / float(self.cols)
        cell_height = height / float(self.rows)

        board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        observations: List[TileObservation] = []

        for row in range(self.rows):
            for col in range(self.cols):
                cell = self._crop_cell(frame, row, col, cell_width, cell_height)
                inner_cell = self._crop_inner(cell)
                if self._is_empty(inner_cell):
                    continue

                observations.append(
                    TileObservation(
                        row=row,
                        col=col,
                        feature=self._extract_feature(inner_cell),
                        gray_signature=self._extract_gray_signature(inner_cell),
                    )
                )

        labels = self._cluster_tile_observations(observations)
        for observation, label in zip(observations, labels):
            board[observation.row][observation.col] = label

        return BoardScanResult(
            board=board,
            cell_width=cell_width,
            cell_height=cell_height,
            tile_count=max(labels, default=0),
        )

    def _crop_cell(
        self,
        frame: np.ndarray,
        row: int,
        col: int,
        cell_width: float,
        cell_height: float,
    ) -> np.ndarray:
        x0 = int(round(col * cell_width))
        x1 = int(round((col + 1) * cell_width))
        y0 = int(round(row * cell_height))
        y1 = int(round((row + 1) * cell_height))

        x0 = max(0, min(x0, frame.shape[1] - 1))
        x1 = max(x0 + 1, min(x1, frame.shape[1]))
        y0 = max(0, min(y0, frame.shape[0] - 1))
        y1 = max(y0 + 1, min(y1, frame.shape[0]))

        return frame[y0:y1, x0:x1]

    def _crop_inner(self, cell: np.ndarray) -> np.ndarray:
        height, width = cell.shape[:2]
        margin_x = int(round(width * self.inner_crop_ratio))
        margin_y = int(round(height * self.inner_crop_ratio))

        if width - 2 * margin_x <= 4 or height - 2 * margin_y <= 4:
            return cell

        return cell[margin_y : height - margin_y, margin_x : width - margin_x]

    def _is_empty(self, cell: np.ndarray) -> bool:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, threshold1=60, threshold2=140)

        edge_density = float(np.mean(edges > 0))
        variance = float(np.var(gray))
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        gray_float = gray.astype(np.float32)
        mean_gray = float(np.mean(gray_float))
        ink_ratio = float(np.mean(np.abs(gray_float - mean_gray) > 18.0))

        return (
            edge_density < self.empty_edge_threshold
            and variance < self.empty_variance_threshold
            and mean_saturation < self.empty_saturation_threshold
            and ink_ratio < self.empty_ink_threshold
        )

    def _cluster_tile_observations(self, observations: List[TileObservation]) -> List[int]:
        count = len(observations)
        if count == 0:
            return []

        parents = list(range(count))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a == root_b:
                return
            if root_a < root_b:
                parents[root_b] = root_a
            else:
                parents[root_a] = root_b

        # Strict merge: very likely same icon.
        for i in range(count):
            obs_i = observations[i]
            for j in range(i + 1, count):
                obs_j = observations[j]
                score = self._combined_similarity(
                    obs_i.feature,
                    obs_i.gray_signature,
                    obs_j.feature,
                    obs_j.gray_signature,
                )
                if score >= self.template_similarity:
                    union(i, j)

        def build_groups() -> Dict[int, List[int]]:
            groups: Dict[int, List[int]] = {}
            for index in range(count):
                root = find(index)
                groups.setdefault(root, []).append(index)
            return groups

        def group_prototype(indexes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
            feature_stack = np.stack([observations[index].feature for index in indexes], axis=0)
            gray_stack = np.stack([observations[index].gray_signature for index in indexes], axis=0)

            feature = np.mean(feature_stack, axis=0)
            gray = np.mean(gray_stack, axis=0)

            feature_norm = float(np.linalg.norm(feature))
            gray_norm = float(np.linalg.norm(gray))
            if feature_norm > 1e-6:
                feature = feature / feature_norm
            if gray_norm > 1e-6:
                gray = gray / gray_norm
            return feature, gray

        relaxed_threshold = max(0.0, self.template_similarity - max(0.03, self.template_ambiguity_margin * 2.5))

        # Relaxed merge for singletons: prevents over-splitting that hides valid paths.
        groups = build_groups()
        singleton_roots = [root for root, indexes in groups.items() if len(indexes) == 1]
        singleton_best: Dict[int, Tuple[int, float]] = {}

        for root in singleton_roots:
            idx = groups[root][0]
            obs = observations[idx]
            best_root = root
            best_score = -1.0
            for other_root in singleton_roots:
                if other_root == root:
                    continue
                other_idx = groups[other_root][0]
                other_obs = observations[other_idx]
                score = self._combined_similarity(
                    obs.feature,
                    obs.gray_signature,
                    other_obs.feature,
                    other_obs.gray_signature,
                )
                if score > best_score:
                    best_score = score
                    best_root = other_root
            singleton_best[root] = (best_root, best_score)

        for root, (best_root, best_score) in singleton_best.items():
            reverse = singleton_best.get(best_root)
            if reverse is None:
                continue
            reverse_root, reverse_score = reverse
            if reverse_root != root:
                continue
            if best_score < relaxed_threshold or reverse_score < relaxed_threshold:
                continue

            union(groups[root][0], groups[best_root][0])

        groups = build_groups()
        singleton_roots = [root for root, indexes in groups.items() if len(indexes) == 1]
        stable_roots = [root for root, indexes in groups.items() if len(indexes) > 1]

        if singleton_roots and stable_roots:
            stable_prototypes = {
                root: group_prototype(indexes)
                for root, indexes in groups.items()
                if len(indexes) > 1
            }

            for root in singleton_roots:
                idx = groups[root][0]
                obs = observations[idx]

                best_root: Optional[int] = None
                best_score = -1.0
                for stable_root in stable_roots:
                    feature, gray = stable_prototypes[stable_root]
                    score = self._combined_similarity(
                        obs.feature,
                        obs.gray_signature,
                        feature,
                        gray,
                    )
                    if score > best_score:
                        best_score = score
                        best_root = stable_root

                if best_root is not None and best_score >= relaxed_threshold:
                    union(idx, groups[best_root][0])

        groups = build_groups()
        ordered_roots = sorted(groups.keys(), key=lambda root: min(groups[root]))
        root_to_label = {root: index + 1 for index, root in enumerate(ordered_roots)}

        labels = [0 for _ in range(count)]
        for index in range(count):
            labels[index] = root_to_label[find(index)]
        return labels

    @staticmethod
    def _extract_feature(cell: np.ndarray) -> np.ndarray:
        resized = cv2.resize(cell, (36, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
        hist = hist.astype(np.float32)

        edges = cv2.Canny(gray, threshold1=60, threshold2=140)
        edges_small = cv2.resize(edges, (12, 12), interpolation=cv2.INTER_AREA).flatten().astype(np.float32)

        gray_small = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).flatten().astype(np.float32)

        feature = np.concatenate([hist, edges_small, gray_small], axis=0)
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-6:
            return feature
        return feature / norm

    @staticmethod
    def _extract_gray_signature(cell: np.ndarray) -> np.ndarray:
        resized = cv2.resize(cell, (20, 20), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        signature = gray.astype(np.float32).flatten()
        signature -= float(np.mean(signature))

        norm = float(np.linalg.norm(signature))
        if norm <= 1e-6:
            return signature
        return signature / norm

    def _combined_similarity(
        self,
        feature_a: np.ndarray,
        gray_a: np.ndarray,
        feature_b: np.ndarray,
        gray_b: np.ndarray,
    ) -> float:
        feature_score = self._cosine_similarity(feature_a, feature_b)
        gray_score = self._cosine_similarity(gray_a, gray_b)
        return (0.72 * feature_score) + (0.28 * gray_score)

    @staticmethod
    def _blend_and_normalize(base: np.ndarray, new: np.ndarray, weight: float) -> np.ndarray:
        if weight <= 0.0:
            return base
        if weight >= 1.0:
            return new

        mixed = ((1.0 - weight) * base) + (weight * new)
        norm = float(np.linalg.norm(mixed))
        if norm <= 1e-6:
            return mixed
        return mixed / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1e-6:
            return 0.0
        return float(np.dot(a, b) / denominator)
