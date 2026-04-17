from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyautogui

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

from pikachu.overlay import render_connectable_overlay
from pikachu.solver import BoardSolver, PairPath
from pikachu.vision import BoardAnalyzer, BoardGrid, BoardScanResult, CaptureRegion, ScreenCapturer

CONFIG_PATH = Path(__file__).with_name("pikachu_config.json")
WINDOW_TITLE = "Pikachu Classic Pro Helper"


@dataclass
class ScanSettings:
    rows: int
    cols: int
    empty_edge_threshold: float
    empty_variance_threshold: float
    empty_saturation_threshold: float
    empty_ink_threshold: float
    template_similarity: float
    template_ambiguity_margin: float
    template_update_weight: float
    max_lines: int


def sanitize_settings(settings: ScanSettings) -> ScanSettings:
    if settings.rows <= 0 or settings.cols <= 0:
        raise ValueError("Rows and cols must be positive integers.")

    settings.empty_edge_threshold = max(0.0, min(settings.empty_edge_threshold, 1.0))
    settings.empty_saturation_threshold = max(0.0, min(settings.empty_saturation_threshold, 255.0))
    settings.empty_ink_threshold = max(0.0, min(settings.empty_ink_threshold, 1.0))
    settings.template_similarity = max(0.0, min(settings.template_similarity, 1.0))
    settings.template_ambiguity_margin = max(0.0, min(settings.template_ambiguity_margin, 1.0))
    settings.template_update_weight = max(0.0, min(settings.template_update_weight, 1.0))
    settings.max_lines = max(1, int(settings.max_lines))
    return settings


class PikachuRunner:
    def __init__(
        self,
        capturer: ScreenCapturer,
        monitor_index: int,
        region: CaptureRegion,
        board_grid: BoardGrid,
        settings: ScanSettings,
        auto_interval: float,
        auto_start: bool,
    ) -> None:
        self.capturer = capturer
        self.monitor_index = monitor_index
        self.region = region
        self.board_grid = board_grid
        self.settings = settings
        self.auto_interval = auto_interval

        self.auto_mode = auto_start
        self.last_action = "Ready"
        self.clicked_cell_ignore_seconds = 1.1
        self.pending_clears: Dict[Tuple[int, int], float] = {}
        self.active_pending_clears = 0
        self.auto_pair_cooldown_seconds = 2.0
        self.auto_pair_cooldowns: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        self.last_board_signature: Optional[Tuple[Tuple[int, ...], ...]] = None
        self.stop_requested = Event()
        self.stop_message = "Emergency stop (Esc)"
        self._esc_listener = self._start_esc_listener()
        self.global_esc_supported = self._esc_listener is not None

        pyautogui.PAUSE = 0.01
        pyautogui.FAILSAFE = True

        self.analyzer = BoardAnalyzer(
            rows=settings.rows,
            cols=settings.cols,
            empty_edge_threshold=settings.empty_edge_threshold,
            empty_variance_threshold=settings.empty_variance_threshold,
            empty_saturation_threshold=settings.empty_saturation_threshold,
            empty_ink_threshold=settings.empty_ink_threshold,
            template_similarity=settings.template_similarity,
            template_ambiguity_margin=settings.template_ambiguity_margin,
            template_update_weight=settings.template_update_weight,
        )

    def scan_once(self) -> Tuple[np.ndarray, BoardScanResult, List[PairPath]]:
        frame = self.capturer.capture_region(self.region)
        board_frame = self._capture_board_frame(frame)
        scan = self.analyzer.analyze(board_frame)
        self.active_pending_clears = self._apply_pending_clears(scan.board)

        board_signature = tuple(tuple(row) for row in scan.board)
        if board_signature != self.last_board_signature:
            self.auto_pair_cooldowns.clear()
            self.last_board_signature = board_signature

        pairs = BoardSolver(scan.board).find_all_pairs()
        return frame, scan, pairs

    def click_pair(self, pair: PairPath, scan: BoardScanResult) -> bool:
        if self.stop_requested.is_set():
            return False

        x1 = self.region.left + self.board_grid.left + int(round((pair.start[1] + 0.5) * scan.cell_width))
        y1 = self.region.top + self.board_grid.top + int(round((pair.start[0] + 0.5) * scan.cell_height))
        x2 = self.region.left + self.board_grid.left + int(round((pair.end[1] + 0.5) * scan.cell_width))
        y2 = self.region.top + self.board_grid.top + int(round((pair.end[0] + 0.5) * scan.cell_height))

        pyautogui.click(x=x1, y=y1)
        if not self._interruptible_sleep(0.045):
            return False
        if self.stop_requested.is_set():
            return False
        pyautogui.click(x=x2, y=y2)
        self._remember_clicked_pair(pair)
        return True

    def request_emergency_stop(self, message: str = "Emergency stop (Esc)") -> None:
        self.auto_mode = False
        self.stop_message = message
        self.last_action = message
        self.stop_requested.set()

    def _consume_emergency_stop(self) -> bool:
        if not self.stop_requested.is_set():
            return False
        self.stop_requested.clear()
        return True

    def reselect_region(self) -> bool:
        region = self.capturer.select_region(self.monitor_index, window_name="Select Pikachu Board")
        if region is None:
            self.last_action = "Region selection canceled"
            return False

        self.region = region
        self.pending_clears.clear()
        self.active_pending_clears = 0
        self.auto_pair_cooldowns.clear()
        self.last_board_signature = None

        self.board_grid = BoardGrid.full_region(region.width, region.height)
        self.last_action = f"Region updated L={region.left} T={region.top} W={region.width} H={region.height}"
        return True

    def _capture_board_frame(self, frame: np.ndarray) -> np.ndarray:
        self.board_grid = self.board_grid.clamp(frame.shape[1], frame.shape[0])
        x0 = self.board_grid.left
        x1 = x0 + self.board_grid.width
        y0 = self.board_grid.top
        y1 = y0 + self.board_grid.height
        return frame[y0:y1, x0:x1]

    def _remember_clicked_pair(self, pair: PairPath) -> None:
        expires_at = time.monotonic() + self.clicked_cell_ignore_seconds
        for cell in (pair.start, pair.end):
            previous = self.pending_clears.get(cell, 0.0)
            if expires_at > previous:
                self.pending_clears[cell] = expires_at

    @staticmethod
    def _pair_key(pair: PairPath) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start, end = sorted((pair.start, pair.end))
        return start, end

    def _pick_auto_pair(self, pairs: List[PairPath]) -> Optional[PairPath]:
        if not pairs:
            return None

        now = time.monotonic()
        expired = [key for key, expiry in self.auto_pair_cooldowns.items() if expiry <= now]
        for key in expired:
            self.auto_pair_cooldowns.pop(key, None)

        for pair in pairs:
            key = self._pair_key(pair)
            if key not in self.auto_pair_cooldowns:
                return pair
        return None

    def _mark_auto_pair(self, pair: PairPath) -> None:
        self.auto_pair_cooldowns[self._pair_key(pair)] = time.monotonic() + self.auto_pair_cooldown_seconds

    def _apply_pending_clears(self, board: List[List[int]]) -> int:
        if not self.pending_clears:
            return 0

        now = time.monotonic()
        expired_cells: List[Tuple[int, int]] = []
        active_count = 0

        for cell, expiry in self.pending_clears.items():
            if expiry <= now:
                expired_cells.append(cell)
                continue

            row, col = cell
            if 0 <= row < self.settings.rows and 0 <= col < self.settings.cols:
                board[row][col] = 0
                active_count += 1
            else:
                expired_cells.append(cell)

        for cell in expired_cells:
            self.pending_clears.pop(cell, None)

        return active_count

    def _interruptible_sleep(self, duration: float) -> bool:
        end = time.monotonic() + max(0.0, duration)
        while True:
            if self.stop_requested.is_set():
                return False
            remaining = end - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.01, remaining))

    def _start_esc_listener(self):
        if pynput_keyboard is None:
            return None

        def on_press(key: object) -> Optional[bool]:
            if key == pynput_keyboard.Key.esc:
                self.request_emergency_stop()
                return None
            return None

        listener = pynput_keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        return listener

    def shutdown(self) -> None:
        if self._esc_listener is not None:
            try:
                self._esc_listener.stop()
            except Exception:
                pass
            self._esc_listener = None

    def run(self) -> None:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, max(920, self.region.width), max(560, self.region.height))

        last_auto_click = 0.0

        while True:
            panic_stop = self._consume_emergency_stop()
            if panic_stop:
                self.last_action = self.stop_message

            try:
                frame, scan, pairs = self.scan_once()
            except Exception as exc:
                canvas = self._error_canvas(f"Scan failed: {exc}")
                cv2.imshow(WINDOW_TITLE, canvas)
                key = cv2.waitKey(25) & 0xFF
                if key == 27:
                    self.request_emergency_stop()
                    continue
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self.reselect_region()
                continue

            overlay = render_connectable_overlay(
                frame=frame,
                pairs=pairs,
                rows=self.settings.rows,
                cols=self.settings.cols,
                max_lines=self.settings.max_lines,
                board_left=self.board_grid.left,
                board_top=self.board_grid.top,
                board_width=self.board_grid.width,
                board_height=self.board_grid.height,
                draw_grid=True,
            )
            self._draw_hud(overlay, pairs)
            cv2.imshow(WINDOW_TITLE, overlay)

            now = time.time()
            auto_pair = self._pick_auto_pair(pairs)
            if self.auto_mode and auto_pair and now - last_auto_click >= self.auto_interval:
                clicked = self.click_pair(auto_pair, scan)
                if clicked:
                    self._mark_auto_pair(auto_pair)
                    last_auto_click = now
                    self.last_action = (
                        f"Auto clicked {self._format_coord(auto_pair.start)} -> "
                        f"{self._format_coord(auto_pair.end)}"
                    )
                elif self.stop_requested.is_set():
                    self._consume_emergency_stop()
                    self.last_action = self.stop_message

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.request_emergency_stop()
                continue
            if key == ord("q"):
                break
            if key == ord("a"):
                self.auto_mode = not self.auto_mode
                self.last_action = f"Auto mode {'ON' if self.auto_mode else 'OFF'}"
            elif key == ord("c"):
                if pairs:
                    if self.click_pair(pairs[0], scan):
                        self.last_action = (
                            f"Manual clicked {self._format_coord(pairs[0].start)} -> "
                            f"{self._format_coord(pairs[0].end)}"
                        )
                    else:
                        if self.stop_requested.is_set():
                            self._consume_emergency_stop()
                            self.last_action = self.stop_message
                        else:
                            self.last_action = "Manual click interrupted"
                else:
                    self.last_action = "No connectable pairs"
            elif key == ord("r"):
                self.reselect_region()
            elif key == ord("w"):
                image_path = Path(f"overlay_{int(time.time())}.png")
                cv2.imwrite(str(image_path), overlay)
                self.last_action = f"Saved snapshot: {image_path.name}"

            time.sleep(0.01)

    def _draw_hud(self, image: np.ndarray, pairs: List[PairPath]) -> None:
        hud_lines = [
            (
                f"Monitor {self.monitor_index} | Region {self.region.width}x{self.region.height} | "
                f"Rows x Cols = {self.settings.rows} x {self.settings.cols}"
            ),
            (
                f"Grid offset {self.board_grid.left},{self.board_grid.top} | "
                f"Grid size {self.board_grid.width}x{self.board_grid.height}"
            ),
            (
                f"Pairs: {len(pairs)} | Auto: {'ON' if self.auto_mode else 'OFF'} "
                f"| Interval: {self.auto_interval:.2f}s"
            ),
            (
                f"Pending skip cells: {self.active_pending_clears} | "
                f"Auto cooldown pairs: {len(self.auto_pair_cooldowns)}"
            ),
            "Keys: [A]uto [C]lick [R]eselect ROI [W]rite [Esc] stop [Q] quit",
            f"Last action: {self.last_action}",
        ]

        y = image.shape[0] - 112
        for line in hud_lines:
            cv2.putText(
                image,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            y += 20

    @staticmethod
    def _error_canvas(message: str) -> np.ndarray:
        canvas = np.zeros((420, 860, 3), dtype=np.uint8)
        cv2.putText(canvas, message, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 180, 255), 2)
        cv2.putText(
            canvas,
            "Press R to select region again or Q to quit.",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
        )
        return canvas

    @staticmethod
    def _format_coord(point: Tuple[int, int]) -> str:
        return f"({point[0] + 1}, {point[1] + 1})"


def load_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}

    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(
    monitor_index: int,
    region: CaptureRegion,
    board_grid: BoardGrid,
    settings: ScanSettings,
    auto_interval: float,
) -> None:
    data: Dict[str, object] = {
        "monitor_index": monitor_index,
        "region": region.to_dict(),
        "board_grid": board_grid.to_dict(),
        "rows": settings.rows,
        "cols": settings.cols,
        "empty_edge_threshold": settings.empty_edge_threshold,
        "empty_variance_threshold": settings.empty_variance_threshold,
        "empty_saturation_threshold": settings.empty_saturation_threshold,
        "empty_ink_threshold": settings.empty_ink_threshold,
        "template_similarity": settings.template_similarity,
        "template_ambiguity_margin": settings.template_ambiguity_margin,
        "template_update_weight": settings.template_update_weight,
        "max_lines": settings.max_lines,
        "interval": auto_interval,
    }
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def choose_monitor_index(
    monitors: List[Dict[str, int]],
    monitor_arg: Optional[int],
    config_monitor: Optional[int],
) -> int:
    available_indexes = {item["index"] for item in monitors}

    if monitor_arg is not None:
        if monitor_arg not in available_indexes:
            raise ValueError(f"Monitor {monitor_arg} is not available.")
        return monitor_arg

    default_index = config_monitor if config_monitor in available_indexes else monitors[0]["index"]

    print("Available monitors:")
    for monitor in monitors:
        print(
            "  "
            f"{monitor['index']}: {monitor['width']}x{monitor['height']} "
            f"@ ({monitor['left']}, {monitor['top']})"
        )

    if not sys.stdin.isatty():
        return default_index

    while True:
        raw = input(f"Select monitor [{default_index}]: ").strip()
        if not raw:
            return default_index
        try:
            selected = int(raw)
        except ValueError:
            print("Please type a valid monitor index.")
            continue
        if selected not in available_indexes:
            print("Monitor index not found.")
            continue
        return selected


def pick_number(
    key: str,
    arg_value: Optional[float],
    config: Dict[str, object],
    default_value: float,
) -> float:
    if arg_value is not None:
        return float(arg_value)
    config_value = config.get(key)
    if config_value is None:
        return default_value
    return float(config_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pikachu classic helper: select screen region, detect board, and show only connectable pairs."
        )
    )
    parser.add_argument("--rows", type=int, default=None, help="Board rows")
    parser.add_argument("--cols", type=int, default=None, help="Board columns")
    parser.add_argument("--monitor", type=int, default=None, help="Monitor index")
    parser.add_argument("--interval", type=float, default=None, help="Auto-click interval in seconds")
    parser.add_argument("--empty-edge", type=float, default=None, help="Empty edge threshold")
    parser.add_argument("--empty-variance", type=float, default=None, help="Empty variance threshold")
    parser.add_argument("--empty-saturation", type=float, default=None, help="Empty saturation threshold")
    parser.add_argument("--empty-ink", type=float, default=None, help="Empty ink threshold")
    parser.add_argument("--similarity", type=float, default=None, help="Template similarity threshold")
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=None,
        help="Minimum similarity gap between best and second-best template",
    )
    parser.add_argument(
        "--template-update",
        type=float,
        default=None,
        help="Template update weight (0..1) to adapt icon prototypes during scan",
    )
    parser.add_argument("--max-lines", type=int, default=None, help="Maximum route lines to draw")
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="Always open region selector even when a saved region exists",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start auto mode immediately",
    )
    return parser.parse_args()


def resolve_region(
    capturer: ScreenCapturer,
    monitor_index: int,
    config: Dict[str, object],
    force_select: bool,
) -> CaptureRegion:
    if not force_select and isinstance(config.get("region"), dict):
        try:
            region = CaptureRegion.from_dict(config["region"])
            if region.monitor_index == monitor_index:
                return region
        except Exception:
            pass

    region = capturer.select_region(monitor_index, window_name="Select Pikachu Board")
    if region is None:
        raise RuntimeError("Region selection canceled. Cannot continue.")
    return region


def resolve_board_grid(
    region: CaptureRegion,
) -> BoardGrid:
    # The selected ROI is the board area directly.
    return BoardGrid.full_region(region.width, region.height)


def main() -> None:
    args = parse_args()
    config = load_config()

    capturer = ScreenCapturer()
    runner: Optional[PikachuRunner] = None
    try:
        monitors = capturer.list_monitors()
        if not monitors:
            raise RuntimeError("No monitors found.")

        monitor_index = choose_monitor_index(
            monitors,
            monitor_arg=args.monitor,
            config_monitor=int(config["monitor_index"]) if isinstance(config.get("monitor_index"), int) else None,
        )

        settings = ScanSettings(
            rows=int(pick_number("rows", args.rows, config, 9)),
            cols=int(pick_number("cols", args.cols, config, 16)),
            empty_edge_threshold=float(pick_number("empty_edge_threshold", args.empty_edge, config, 0.045)),
            empty_variance_threshold=float(
                pick_number("empty_variance_threshold", args.empty_variance, config, 135.0)
            ),
            empty_saturation_threshold=float(
                pick_number("empty_saturation_threshold", args.empty_saturation, config, 45.0)
            ),
            empty_ink_threshold=float(
                pick_number("empty_ink_threshold", args.empty_ink, config, 0.16)
            ),
            template_similarity=float(pick_number("template_similarity", args.similarity, config, 0.9)),
            template_ambiguity_margin=float(
                pick_number("template_ambiguity_margin", args.ambiguity_margin, config, 0.01)
            ),
            template_update_weight=float(
                pick_number("template_update_weight", args.template_update, config, 0.2)
            ),
            max_lines=int(pick_number("max_lines", args.max_lines, config, 12)),
        )
        settings = sanitize_settings(settings)

        auto_interval = float(pick_number("interval", args.interval, config, 0.25))

        region = resolve_region(
            capturer=capturer,
            monitor_index=monitor_index,
            config=config,
            force_select=args.select_region,
        )

        board_grid = resolve_board_grid(
            region=region,
        )

        runner = PikachuRunner(
            capturer=capturer,
            monitor_index=monitor_index,
            region=region,
            board_grid=board_grid,
            settings=settings,
            auto_interval=auto_interval,
            auto_start=args.auto_start,
        )

        print("\nControls inside preview window:")
        print("  A = toggle auto mode")
        print("  C = click one pair")
        print("  R = reselect board region")
        print("  W = save current overlay snapshot")
        print("  Esc = emergency stop immediately")
        print("  Q = quit")
        if not runner.global_esc_supported:
            print("  Note: global Esc listener unavailable; run pip install -r requirements.txt")
        print("\nTip: move mouse to top-left corner to trigger PyAutoGUI fail-safe if needed.\n")

        runner.run()

        save_config(
            monitor_index=monitor_index,
            region=runner.region,
            board_grid=runner.board_grid,
            settings=settings,
            auto_interval=auto_interval,
        )
    finally:
        if runner is not None:
            runner.shutdown()
        cv2.destroyAllWindows()
        capturer.close()


if __name__ == "__main__":
    main()
