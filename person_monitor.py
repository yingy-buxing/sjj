import argparse
import ctypes
import json
import math
import queue
import threading
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

try:
    import mss
except ImportError as exc:
    raise SystemExit("缺少依赖：mss。请运行：pip install -r requirements.txt") from exc

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


PERSON_CLASS_ID = 0
SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / "person_monitor_settings.json"
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120
INPUT_MOUSE = 0
WINMM = ctypes.WinDLL("winmm")
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_NOREPEAT = 0x4000
HOTKEY_ID_TOGGLE_ALIGN = 1
HOTKEY_TOGGLE_ALIGN_NAME = "F8"
VK_F8 = 0x77
KEY_PRESSED_MASK = 0x8000
DETECTION_MODE_NORMAL = "普通模型"
DETECTION_MODE_POSE = "姿态模型"
DETECTION_MODES = (DETECTION_MODE_NORMAL, DETECTION_MODE_POSE)
POSE_MODEL_PRESETS = (
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
)
NORMAL_MODEL_PRESETS = (
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
)
MODEL_PRESETS = NORMAL_MODEL_PRESETS + POSE_MODEL_PRESETS
MODEL_ASSET_VERSION = "v8.4.0"
MODEL_ASSET_BASE_URL = (
    f"https://github.com/ultralytics/assets/releases/download/{MODEL_ASSET_VERSION}"
)
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)
BODY_KEYPOINTS = (5, 6, 11, 12)
CENTER_KEYPOINTS = (0, 5, 6, 11, 12)
SMALL_TARGET_ANCHOR_KEYPOINTS = HEAD_KEYPOINTS + BODY_KEYPOINTS + (13, 14, 15, 16)
ALIGN_CONTROL_STABLE = "稳定"
ALIGN_CONTROL_FAST = "快速"
ALIGN_CONTROL_DIRECT = "直接"
ALIGN_CONTROL_MODES = (ALIGN_CONTROL_STABLE, ALIGN_CONTROL_FAST, ALIGN_CONTROL_DIRECT)


class Point(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class InputUnion(ctypes.Union):
    _fields_ = [("mi", MouseInput)]


class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", InputUnion),
    ]


def clamp(value: int, limit: int) -> int:
    return max(-limit, min(limit, value))


class Tooltip:
    """鼠标悬停一段时间后显示的说明气泡（不占用界面常驻空间）。"""

    def __init__(
        self,
        widget: tk.Misc,
        text: str,
        delay_ms: int = 450,
        wraplength: int = 300,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._tip: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Button>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel_schedule()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_schedule(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except (tk.TclError, ValueError):
                pass
            self._after_id = None

    def _destroy_tip(self) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None

    def _hide(self, _event=None) -> None:
        self._cancel_schedule()
        self._destroy_tip()

    def _show(self) -> None:
        self._after_id = None
        if not self.widget.winfo_exists():
            return
        try:
            if str(self.widget.cget("state")) == "disabled":
                return
        except tk.TclError:
            pass

        x = self.widget.winfo_rootx() + max(self.widget.winfo_width() // 2, 8)
        y = self.widget.winfo_rooty() + self.widget.winfo_height()
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        try:
            self._tip.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        tk.Label(
            self._tip,
            text=self.text,
            justify=tk.LEFT,
            wraplength=self.wraplength,
            background="#ffffe0",
            foreground="#111111",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
        ).pack()
        self._tip.update_idletasks()
        tw = self._tip.winfo_reqwidth()
        th = self._tip.winfo_reqheight()
        sw = self.widget.winfo_screenwidth()
        sh = self.widget.winfo_screenheight()
        x = min(max(8, x - tw // 2), max(8, sw - tw - 8))
        y = min(max(8, y + 8), max(8, sh - th - 8))
        self._tip.wm_geometry(f"+{x}+{y}")


def bind_tooltip_text(*widgets: tk.Misc, text: str) -> None:
    for widget in widgets:
        Tooltip(widget, text)


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as file:
            settings = json.load(file)
        return settings if isinstance(settings, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def saved_int(settings: dict, key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def saved_optional_int(settings: dict, key: str) -> Optional[int]:
    if key not in settings:
        return None
    try:
        return int(settings[key])
    except (TypeError, ValueError):
        return None


def saved_bool(settings: dict, key: str, default: bool) -> bool:
    value = settings.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def saved_str(settings: dict, key: str, default: str) -> str:
    value = settings.get(key, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def saved_float(settings: dict, key: str, default: float) -> float:
    try:
        return float(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def is_pose_model_name(model_name: str) -> bool:
    return "-pose" in Path(model_name.strip()).name.lower()


def model_mode(model_name: str) -> str:
    return DETECTION_MODE_POSE if is_pose_model_name(model_name) else DETECTION_MODE_NORMAL


def normalize_detection_mode(mode: str) -> str:
    if mode in DETECTION_MODES:
        return mode
    normalized = mode.strip().lower()
    if normalized in {"pose", "姿态", "姿态模型"}:
        return DETECTION_MODE_POSE
    return DETECTION_MODE_NORMAL


def preferred_model(candidates: tuple[str, ...], fallback: str) -> str:
    for model_name in candidates:
        if local_model_status(model_name)[0] == "已下载":
            return model_name
    return fallback


def model_file_path(model_name: str) -> Path:
    return SCRIPT_DIR / model_name


def model_download_url(model_name: str) -> str:
    return f"{MODEL_ASSET_BASE_URL}/{model_name}"


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def local_model_status(model_name: str) -> tuple[str, str]:
    path = model_file_path(model_name)
    if not path.exists():
        return "未下载", "-"
    size = path.stat().st_size
    if size == 0:
        return "空文件", "0 B"
    return "已下载", format_bytes(size)


def enable_high_resolution_timer() -> None:
    try:
        WINMM.timeBeginPeriod(1)
    except OSError:
        pass


def disable_high_resolution_timer() -> None:
    try:
        WINMM.timeEndPeriod(1)
    except OSError:
        pass


def sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.004:
            time.sleep(remaining - 0.002)
        else:
            time.sleep(0)


def bgra_to_rgb(frame: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    return frame[:, :, :3][:, :, ::-1].copy()


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "人"
    center_point: Optional[tuple[int, int]] = None
    keypoints: Optional[list[tuple[int, int, float]]] = None

    @property
    def center(self) -> tuple[int, int]:
        if self.center_point is not None:
            return self.center_point
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class PoseFilterConfig:
    enabled: bool
    min_keypoints: int
    keypoint_confidence: float
    require_head: bool
    require_body: bool
    relax_small_targets: bool
    small_target_height: int


@dataclass
class MouseAlignConfig:
    enabled: bool
    target_x: int
    target_y: int
    sensitivity: float
    deadzone: int
    max_step: int
    cooldown: float
    control_mode: str
    lead_ms: int
    invert_x: bool
    invert_y: bool
    move_mode: str
    smooth_steps: int
    smooth_duration: float
    click_on_target: bool


@dataclass
class CalibrationState:
    pulses: list[tuple[int, int]]
    pulse_index: int = 0
    phase: str = "stabilizing"
    stable_center: Optional[tuple[float, float]] = None
    stable_since: float = 0.0
    baseline_center: Optional[tuple[float, float]] = None
    pulse_sent_at: float = 0.0
    samples_x: Optional[list[float]] = None
    samples_y: Optional[list[float]] = None
    status: str = "等待目标稳定"

    def __post_init__(self) -> None:
        if self.samples_x is None:
            self.samples_x = []
        if self.samples_y is None:
            self.samples_y = []


def damped_axis_move(error: int, sensitivity: float, max_step: int, deadzone: int) -> int:
    abs_error = abs(error)
    axis_deadzone = max(1, deadzone // 2)
    if abs_error <= axis_deadzone:
        return 0

    raw_move = abs_error * sensitivity
    slow_radius = max(deadzone * 4, 24)
    if abs_error < slow_radius:
        raw_move *= max(0.2, abs_error / slow_radius)

    limit = min(max_step, max(1, abs_error - axis_deadzone))
    move = min(limit, max(1, int(round(raw_move))))
    return move if error > 0 else -move


def direct_axis_move(error: int, sensitivity: float, max_step: int, deadzone: int) -> int:
    abs_error = abs(error)
    axis_deadzone = max(1, deadzone // 2)
    if abs_error <= axis_deadzone:
        return 0
    limit = min(max_step, max(1, abs_error - axis_deadzone))
    move = min(limit, max(1, int(round(abs_error * sensitivity))))
    return move if error > 0 else -move


def align_axis_move(
    error: int,
    sensitivity: float,
    max_step: int,
    deadzone: int,
    control_mode: str,
) -> int:
    if control_mode == ALIGN_CONTROL_DIRECT:
        direct_deadzone = max(1, deadzone // 2)
        return direct_axis_move(error, sensitivity, max_step, direct_deadzone)
    if control_mode == ALIGN_CONTROL_FAST:
        return direct_axis_move(error, sensitivity, max_step, deadzone)
    return damped_axis_move(error, sensitivity, max_step, deadzone)


class MouseMover:
    def __init__(self) -> None:
        self.last_error = ""
        self._lock = threading.Lock()
        self._pending_x = 0.0
        self._pending_y = 0.0
        self._mode = "send_input"
        self._smooth_steps = 16
        self._smooth_duration = 0.12
        self._worker_started = False
        self._stop_event = threading.Event()

    def move_relative(
        self,
        dx: int,
        dy: int,
        mode: str,
        smooth_steps: int,
        smooth_duration: float,
        replace_pending: bool = False,
    ) -> None:
        if dx == 0 and dy == 0:
            return
        self._ensure_worker()
        with self._lock:
            if replace_pending:
                self._pending_x = self._blend_pending(self._pending_x, float(dx))
                self._pending_y = self._blend_pending(self._pending_y, float(dy))
            else:
                self._pending_x += dx
                self._pending_y += dy
            self._mode = mode
            self._smooth_steps = max(1, int(smooth_steps))
            self._smooth_duration = max(0.0, float(smooth_duration))

    def scroll(self, clicks: int) -> None:
        if clicks == 0:
            return
        self._send_mouse_input(0, 0, clicks * WHEEL_DELTA, MOUSEEVENTF_WHEEL)

    def _send_input_relative(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        self._send_mouse_input(int(dx), int(dy), 0, MOUSEEVENTF_MOVE)

    def _set_cursor_relative(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        point = Point()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            ctypes.windll.user32.SetCursorPos(point.x + int(dx), point.y + int(dy))
            self.last_error = ""
        else:
            self.last_error = "GetCursorPos 失败"

    def _send_mouse_button(self, flag: int) -> None:
        self._send_mouse_input(0, 0, 0, flag)

    def _send_mouse_input(self, dx: int, dy: int, mouse_data: int, flags: int) -> None:
        extra_info = ctypes.c_ulong(0)
        mouse_input = MouseInput(
            dx=dx,
            dy=dy,
            mouseData=mouse_data,
            dwFlags=flags,
            time=0,
            dwExtraInfo=ctypes.pointer(extra_info),
        )
        input_data = Input(type=INPUT_MOUSE, union=InputUnion(mi=mouse_input))
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(input_data), ctypes.sizeof(input_data))
        if sent == 0:
            self.last_error = f"SendInput 失败，错误码：{ctypes.windll.kernel32.GetLastError()}"
        else:
            self.last_error = ""

    def clear_pending(self) -> None:
        with self._lock:
            self._pending_x = 0.0
            self._pending_y = 0.0

    def pending_magnitude(self) -> float:
        with self._lock:
            return abs(self._pending_x) + abs(self._pending_y)

    def click_left(self) -> None:
        self._send_mouse_button(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.018)
        self._send_mouse_button(MOUSEEVENTF_LEFTUP)

    def close(self) -> None:
        self.clear_pending()
        self._stop_event.set()

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        threading.Thread(target=self._smooth_loop, daemon=True).start()

    def _smooth_loop(self) -> None:
        while not self._stop_event.is_set():
            step_x, step_y, mode, interval = self._take_smooth_step()

            if step_x == 0 and step_y == 0:
                time.sleep(0.005)
                continue

            if mode == "set_cursor":
                self._set_cursor_relative(step_x, step_y)
            else:
                self._send_input_relative(step_x, step_y)
            time.sleep(interval)

    def _take_smooth_step(self) -> tuple[int, int, str, float]:
        with self._lock:
            steps = max(1, self._smooth_steps)
            interval = max(0.002, self._smooth_duration / steps) if self._smooth_duration else 0.002
            mode = self._mode

            step_x = self._next_step(self._pending_x, steps)
            step_y = self._next_step(self._pending_y, steps)
            self._pending_x -= step_x
            self._pending_y -= step_y

            if abs(self._pending_x) < 0.5:
                self._pending_x = 0.0
            if abs(self._pending_y) < 0.5:
                self._pending_y = 0.0

        return step_x, step_y, mode, interval

    def _next_step(self, pending: float, steps: int) -> int:
        if abs(pending) < 0.5:
            return 0
        step = int(round(pending / steps))
        if step == 0:
            return 1 if pending > 0 else -1
        return step

    def _blend_pending(self, current: float, target: float) -> float:
        if abs(current) < 0.5:
            return target
        if current * target < 0:
            return target
        return current + (target - current) * 0.65



class HotkeyListener:
    def __init__(self, event_queue: queue.Queue) -> None:
        self.event_queue = event_queue
        self.thread: Optional[threading.Thread] = None
        self.thread_id = 0
        self._stop_event = threading.Event()
        self._polling_fallback = False

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self.thread_id:
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
        if self.thread is not None:
            self.thread.join(timeout=0.5)

    def _run(self) -> None:
        try:
            self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
            registered = bool(
                ctypes.windll.user32.RegisterHotKey(
                    None,
                    HOTKEY_ID_TOGGLE_ALIGN,
                    MOD_NOREPEAT,
                    VK_F8,
                )
            )
            if not registered:
                error_code = ctypes.windll.kernel32.GetLastError()
                self.event_queue.put(
                    (
                        "hotkey_error",
                        f"{HOTKEY_TOGGLE_ALIGN_NAME} 注册失败：{error_code}，已启用备用监听",
                    )
                )
                self._poll_key_state()
                return

            self.event_queue.put(("hotkey_ready", HOTKEY_TOGGLE_ALIGN_NAME))
            message = wintypes.MSG()
            while (
                not self._stop_event.is_set()
                and ctypes.windll.user32.GetMessageW(ctypes.byref(message), None, 0, 0) != 0
            ):
                if (
                    message.message == WM_HOTKEY
                    and message.wParam == HOTKEY_ID_TOGGLE_ALIGN
                ):
                    self.event_queue.put(("toggle_align", None))
                ctypes.windll.user32.TranslateMessage(ctypes.byref(message))
                ctypes.windll.user32.DispatchMessageW(ctypes.byref(message))
        except Exception as exc:
            self.event_queue.put(("hotkey_error", f"热键线程异常：{exc}，已启用备用监听"))
            self._poll_key_state()
        finally:
            if self.thread_id and not self._polling_fallback:
                ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_ID_TOGGLE_ALIGN)

    def _poll_key_state(self) -> None:
        self._polling_fallback = True
        was_down = False
        while not self._stop_event.is_set():
            is_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F8) & KEY_PRESSED_MASK)
            if is_down and not was_down:
                self.event_queue.put(("toggle_align", None))
            was_down = is_down
            time.sleep(0.03)


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        device: str,
        imgsz: int,
        half: bool,
        pose_filter: PoseFilterConfig,
    ) -> None:
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = self._resolve_device(device)
        self.half = half and self.device != "cpu"
        self.pose_filter = pose_filter
        self.is_pose_model = False
        self.backend = "opencv-hog"
        self.model = None
        self.hog = None

        if YOLO is not None:
            resolved_model_path = self._resolve_model_path(model_path)
            self.model = YOLO(resolved_model_path)
            model_task = str(getattr(self.model, "task", "") or "").lower()
            self.is_pose_model = (
                model_task == "pose" or "-pose" in Path(resolved_model_path).name.lower()
            )
            model_kind = "姿态" if self.is_pose_model else "检测"
            self.backend = f"yolo-{model_kind}:{resolved_model_path} 设备={self.device}"
            if torch is not None and self.device != "cpu":
                torch.backends.cudnn.benchmark = True
        elif cv2 is not None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            raise RuntimeError(
                "没有可用的检测后端。请安装 ultralytics 或 opencv-python。"
            )

    def _resolve_model_path(self, model_path: str) -> str:
        raw_path = model_path.strip()
        if not raw_path:
            raise ValueError("模型路径不能为空。")

        path = Path(raw_path).expanduser()
        if path.is_absolute():
            if path.exists() and path.stat().st_size == 0:
                raise ValueError(f"模型文件为空：{path}。请删除它后重新运行，让程序重新下载。")
            return str(path)

        script_relative_path = SCRIPT_DIR / path
        if script_relative_path.exists():
            if script_relative_path.stat().st_size == 0:
                raise ValueError(
                    f"模型文件为空：{script_relative_path}。请删除它后重新运行，让程序重新下载。"
                )
            return str(script_relative_path)

        return raw_path

    def _resolve_device(self, device: str) -> str:
        requested = device.strip().lower()
        if requested and requested != "auto":
            return requested
        if torch is not None and torch.cuda.is_available():
            return "0"
        return "cpu"

    def detect(self, rgb_frame: np.ndarray) -> list[Detection]:
        if self.model is not None:
            return self._detect_yolo(rgb_frame)
        return self._detect_hog(rgb_frame)

    def update_runtime(
        self,
        confidence: float,
        imgsz: int,
        pose_filter: PoseFilterConfig,
    ) -> None:
        self.confidence = confidence
        self.imgsz = imgsz
        self.pose_filter = pose_filter

    def _detect_yolo(self, rgb_frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=rgb_frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            keypoints_xy, keypoints_conf = self._result_keypoints(result)
            for index, box in enumerate(result.boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().item())
                keypoints = self._extract_keypoints(keypoints_xy, keypoints_conf, index)
                if self.is_pose_model and not self._passes_pose_filter(keypoints, coords):
                    continue
                detections.append(
                    Detection(
                        *coords,
                        confidence=conf,
                        center_point=self._pose_center(keypoints),
                        keypoints=keypoints,
                    )
                )
        return detections

    def _result_keypoints(self, result) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or getattr(keypoints, "xy", None) is None:
            return None, None

        xy = keypoints.xy.cpu().numpy()
        conf_tensor = getattr(keypoints, "conf", None)
        conf = conf_tensor.cpu().numpy() if conf_tensor is not None else None
        return xy, conf

    def _extract_keypoints(
        self,
        keypoints_xy: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
        index: int,
    ) -> Optional[list[tuple[int, int, float]]]:
        if keypoints_xy is None or index >= len(keypoints_xy):
            return None

        points: list[tuple[int, int, float]] = []
        for point_index, (x, y) in enumerate(keypoints_xy[index]):
            if keypoints_conf is not None and point_index < len(keypoints_conf[index]):
                conf = float(keypoints_conf[index][point_index])
            else:
                conf = 1.0 if x > 0 or y > 0 else 0.0
            points.append((int(round(float(x))), int(round(float(y))), conf))
        return points

    def _valid_keypoints(
        self,
        keypoints: Optional[list[tuple[int, int, float]]],
    ) -> list[tuple[int, int, int, float]]:
        if not keypoints:
            return []

        threshold = self.pose_filter.keypoint_confidence
        valid = []
        for index, (x, y, conf) in enumerate(keypoints):
            if conf >= threshold and (x != 0 or y != 0):
                valid.append((index, x, y, conf))
        return valid

    def _passes_pose_filter(
        self,
        keypoints: Optional[list[tuple[int, int, float]]],
        coords: list[int],
    ) -> bool:
        if not self.pose_filter.enabled:
            return True

        valid = self._valid_keypoints(keypoints)
        valid_indices = {index for index, _x, _y, _conf in valid}
        box_height = max(0, coords[3] - coords[1])
        if (
            self.pose_filter.relax_small_targets
            and box_height <= self.pose_filter.small_target_height
            and len(valid) >= 2
            and valid_indices.intersection(SMALL_TARGET_ANCHOR_KEYPOINTS)
        ):
            return True

        if len(valid) < self.pose_filter.min_keypoints:
            return False

        if self.pose_filter.require_head and not valid_indices.intersection(HEAD_KEYPOINTS):
            return False
        if self.pose_filter.require_body and not valid_indices.intersection(BODY_KEYPOINTS):
            return False
        return True

    def _pose_center(
        self,
        keypoints: Optional[list[tuple[int, int, float]]],
    ) -> Optional[tuple[int, int]]:
        valid = self._valid_keypoints(keypoints)
        if not valid:
            return None

        center_points = [
            (x, y)
            for index, x, y, _conf in valid
            if index in CENTER_KEYPOINTS
        ]
        if len(center_points) < 2:
            center_points = [(x, y) for _index, x, y, _conf in valid]
        if not center_points:
            return None

        avg_x = int(round(sum(x for x, _y in center_points) / len(center_points)))
        avg_y = int(round(sum(y for _x, y in center_points) / len(center_points)))
        return avg_x, avg_y

    def _detect_hog(self, rgb_frame: np.ndarray) -> list[Detection]:
        if cv2 is None or self.hog is None:
            return []
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        rects, weights = self.hog.detectMultiScale(
            bgr_frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            conf = float(weight)
            if conf >= self.confidence:
                detections.append(Detection(x, y, x + w, y + h, conf))
        return detections


class RegionSelector(tk.Toplevel):
    def __init__(self, parent: tk.Tk, on_selected) -> None:
        super().__init__(parent)
        self.on_selected = on_selected
        self.start_x = 0
        self.start_y = 0
        self.rect_id: Optional[int] = None

        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.25)
        self.attributes("-topmost", True)
        self.configure(bg="black")
        self.overrideredirect(True)

        self.canvas = tk.Canvas(self, cursor="crosshair", bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", lambda _event: self.destroy())

    def _on_press(self, event) -> None:
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#00ff66",
            width=3,
        )

    def _on_drag(self, event) -> None:
        if self.rect_id is not None:
            self.canvas.coords(
                self.rect_id,
                self.start_x,
                self.start_y,
                event.x_root,
                event.y_root,
            )

    def _on_release(self, event) -> None:
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x_root, event.y_root
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        self.destroy()
        if width >= 20 and height >= 20:
            self.on_selected(left, top, width, height)


class TargetPointSelector(tk.Toplevel):
    def __init__(self, parent: tk.Tk, on_selected) -> None:
        super().__init__(parent)
        self.on_selected = on_selected

        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.25)
        self.attributes("-topmost", True)
        self.configure(bg="black")
        self.overrideredirect(True)

        self.canvas = tk.Canvas(self, cursor="crosshair", bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", lambda _event: self.destroy())

    def _on_release(self, event) -> None:
        self.destroy()
        self.on_selected(event.x_root, event.y_root)


class PersonMonitorApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("屏幕人物监测")
        self.root.geometry("1120x760")
        self.root.minsize(900, 620)
        self.root.attributes("-topmost", True)

        self.args = args
        self.detector: Optional[PersonDetector] = None
        self.running = False
        self.worker: Optional[threading.Thread] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.download_queue: queue.Queue = queue.Queue()
        self.hotkey_queue: queue.Queue = queue.Queue()
        self.calibration_queue: queue.Queue = queue.Queue()
        self.preview_image = None
        self.mouse_mover = MouseMover()
        self.hotkey_listener = HotkeyListener(self.hotkey_queue)
        self.last_mouse_move = 0.0
        self.align_settled = False
        self.capture_stop_event: Optional[threading.Event] = None
        self.download_worker: Optional[threading.Thread] = None
        self.downloading_model: Optional[str] = None
        self.locked_detection_center: Optional[tuple[int, int]] = None
        self.tracked_center: Optional[tuple[float, float]] = None
        self.tracked_velocity: tuple[float, float] = (0.0, 0.0)
        self.tracked_time: Optional[float] = None
        self.calibration_lock = threading.Lock()
        self.calibration_state: Optional[CalibrationState] = None
        self.calibration_message = ""
        self.calibration_after_id: Optional[str] = None
        self.preview_enabled = bool(args.preview)
        self.detector_model_key: Optional[tuple[str, str, bool]] = None
        self.restart_warning = ""
        self.runtime_lock = threading.Lock()
        self.runtime_target_fps = max(1, int(args.fps))
        self.runtime_confidence = float(args.confidence)
        self.runtime_imgsz = int(args.imgsz)
        self.runtime_pose_filter = PoseFilterConfig(
            bool(args.pose_filter),
            int(args.pose_min_keypoints),
            float(args.pose_keypoint_confidence),
            bool(args.pose_require_head),
            bool(args.pose_require_body),
            bool(args.pose_relax_small_targets),
            int(args.pose_small_target_height),
        )
        self.runtime_detector_model_key: tuple[str, str, bool] = (
            args.model.strip(),
            args.device.strip(),
            bool(args.half),
        )

        self.left_var = tk.IntVar(value=args.left)
        self.top_var = tk.IntVar(value=args.top)
        self.width_var = tk.IntVar(value=args.width)
        self.height_var = tk.IntVar(value=args.height)
        self.conf_var = tk.DoubleVar(value=args.confidence)
        self.fps_var = tk.IntVar(value=args.fps)
        self.imgsz_var = tk.IntVar(value=args.imgsz)
        self.preview_fps_var = tk.IntVar(value=args.preview_fps)
        self.preview_enabled_var = tk.BooleanVar(value=self.preview_enabled)
        self.half_var = tk.BooleanVar(value=args.half)
        self.detection_mode_var = tk.StringVar(value=args.detection_mode)
        self.normal_model_name = args.normal_model
        self.pose_model_name = args.pose_model
        if is_pose_model_name(args.model):
            self.pose_model_name = args.model
            self.detection_mode_var.set(DETECTION_MODE_POSE)
        else:
            self.normal_model_name = args.model
            self.detection_mode_var.set(DETECTION_MODE_NORMAL)
        self.model_var = tk.StringVar(value=self._model_for_mode(self.detection_mode_var.get()))
        self.device_var = tk.StringVar(value=args.device)
        self.pose_filter_enabled_var = tk.BooleanVar(value=args.pose_filter)
        self.pose_min_keypoints_var = tk.IntVar(value=args.pose_min_keypoints)
        self.pose_keypoint_conf_var = tk.DoubleVar(value=args.pose_keypoint_confidence)
        self.pose_require_head_var = tk.BooleanVar(value=args.pose_require_head)
        self.pose_require_body_var = tk.BooleanVar(value=args.pose_require_body)
        self.pose_relax_small_targets_var = tk.BooleanVar(value=args.pose_relax_small_targets)
        self.pose_small_target_height_var = tk.IntVar(value=args.pose_small_target_height)
        self.status_var = tk.StringVar(value="就绪")
        default_target_x = args.target_x if args.target_x is not None else args.left + args.width // 2
        default_target_y = args.target_y if args.target_y is not None else args.top + args.height // 2
        self.align_enabled_var = tk.BooleanVar(value=args.auto_align)
        self.target_x_var = tk.IntVar(value=default_target_x)
        self.target_y_var = tk.IntVar(value=default_target_y)
        self.align_sensitivity_var = tk.DoubleVar(value=args.align_sensitivity)
        self.align_deadzone_var = tk.IntVar(value=args.align_deadzone)
        self.align_max_step_var = tk.IntVar(value=args.align_max_step)
        self.align_cooldown_var = tk.DoubleVar(value=args.align_cooldown)
        self.align_control_mode_var = tk.StringVar(value=args.align_control_mode)
        self.align_lead_ms_var = tk.IntVar(value=args.align_lead_ms)
        self.align_invert_x_var = tk.BooleanVar(value=args.align_invert_x)
        self.align_invert_y_var = tk.BooleanVar(value=args.align_invert_y)
        self.align_move_mode_var = tk.StringVar(value=args.align_move_mode)
        self.align_smooth_steps_var = tk.IntVar(value=args.align_smooth_steps)
        self.align_smooth_duration_var = tk.DoubleVar(value=args.align_smooth_duration)
        self.align_click_on_target_var = tk.BooleanVar(value=args.align_click_on_target)
        self.align_config = self._read_align_config()
        self.last_target_click = 0.0
        self._align_deadzone_stable_frames = 0
        self._align_click_consumed = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind_all("<F8>", self._on_f8_pressed)
        self.hotkey_listener.start()
        self.root.after(self._preview_interval_ms(), self._poll_frames)

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root_frame)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.preview_frame = ttk.Frame(root_frame)
        if self.preview_enabled:
            self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        action_bar = ttk.Frame(controls)
        action_bar.pack(fill=tk.X, pady=(0, 10))
        self.start_button = ttk.Button(action_bar, text="开始", command=self._toggle)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(action_bar, text="停止", command=self._stop).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        settings_tabs = ttk.Notebook(controls, width=280, height=430)
        settings_tabs.pack(fill=tk.BOTH, expand=True)

        region_tab = ttk.Frame(settings_tabs, padding=10)
        mouse_tab = ttk.Frame(settings_tabs, padding=10)
        detector_tab = ttk.Frame(settings_tabs, padding=10)
        model_tab = ttk.Frame(settings_tabs, padding=10)
        test_tab = ttk.Frame(settings_tabs, padding=10)
        settings_tabs.add(region_tab, text="区域")
        settings_tabs.add(mouse_tab, text="鼠标")
        settings_tabs.add(detector_tab, text="检测")
        settings_tabs.add(model_tab, text="模型")
        settings_tabs.add(test_tab, text="测试")

        region_grid = ttk.Frame(region_tab)
        region_grid.pack(fill=tk.X, pady=(0, 12))

        fields = [
            ("左侧", self.left_var),
            ("顶部", self.top_var),
            ("宽度", self.width_var),
            ("高度", self.height_var),
        ]
        region_hints = [
            "截取区域左边界在屏幕上的 X 像素坐标。",
            "截取区域上边界在屏幕上的 Y 像素坐标。",
            "截取区域的宽度（像素）。过小会导致检测不稳定。",
            "截取区域的高度（像素）。过小会导致检测不稳定。",
        ]
        for row, ((label, var), hint) in enumerate(zip(fields, region_hints)):
            rl = ttk.Label(region_grid, text=label)
            rl.grid(row=row, column=0, sticky=tk.W, pady=3)
            re = ttk.Entry(region_grid, textvariable=var, width=10)
            re.grid(row=row, column=1, sticky=tk.EW, pady=3)
            bind_tooltip_text(rl, re, text=hint)

        btn_sel_region = ttk.Button(region_tab, text="选择检测区域", command=self._select_region)
        btn_sel_region.pack(fill=tk.X, pady=(0, 8))
        bind_tooltip_text(
            btn_sel_region,
            text="在屏幕上框选一块矩形区域，仅在该区域内做人形检测（减轻负荷）。",
        )
        btn_region_target = ttk.Button(
            region_tab, text="使用区域中心作目标", command=self._target_region_center
        )
        btn_region_target.pack(fill=tk.X)
        bind_tooltip_text(
            btn_region_target,
            text="把鼠标对齐目标点设为当前截取区域的中心（写在「鼠标」页的目标 X/Y）。",
        )

        align_grid = ttk.Frame(mouse_tab)
        align_grid.pack(fill=tk.X, pady=(0, 12))

        cb_align_enable = ttk.Checkbutton(
            align_grid,
            text="启用自动移动",
            variable=self.align_enabled_var,
        )
        cb_align_enable.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=3)
        bind_tooltip_text(
            cb_align_enable,
            text=(
                "开启后根据检测结果移动鼠标，使人物尽量对准「目标标点」。"
                "FPS 建议先试下方「稳一点 / 均衡 / 跟手」预设，再微调单项参数。"
            ),
        )

        lbl_target_x = ttk.Label(align_grid, text="目标 X")
        lbl_target_x.grid(row=1, column=0, sticky=tk.W, pady=3)
        ent_target_x = ttk.Entry(align_grid, textvariable=self.target_x_var, width=10)
        ent_target_x.grid(row=1, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_target_x,
            ent_target_x,
            text="屏幕上的准星或十字参考点的 X。可与「选择目标标点」配合设置。",
        )

        lbl_target_y = ttk.Label(align_grid, text="目标 Y")
        lbl_target_y.grid(row=2, column=0, sticky=tk.W, pady=3)
        ent_target_y = ttk.Entry(align_grid, textvariable=self.target_y_var, width=10)
        ent_target_y.grid(row=2, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_target_y,
            ent_target_y,
            text=(
                "屏幕上的准星或十字参考点的 Y。"
                "人物检测中心进入「停止范围」即视为已对准该点。"
            ),
        )

        lbl_sens = ttk.Label(align_grid, text="移动倍率")
        lbl_sens.grid(row=3, column=0, sticky=tk.W, pady=3)
        sb_sens = ttk.Spinbox(
            align_grid,
            textvariable=self.align_sensitivity_var,
            from_=0.05,
            to=5.0,
            increment=0.05,
            width=8,
        )
        sb_sens.grid(row=3, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_sens,
            sb_sens,
            text=(
                "单次修正力度：越大越容易过头，越小追得越慢。"
                "方向明显不对时优先用「校准鼠标倍率」，再调此项。"
            ),
        )

        lbl_dead = ttk.Label(align_grid, text="停止范围")
        lbl_dead.grid(row=4, column=0, sticky=tk.W, pady=3)
        sb_dead = ttk.Spinbox(
            align_grid,
            textvariable=self.align_deadzone_var,
            from_=0,
            to=200,
            increment=1,
            width=8,
        )
        sb_dead.grid(row=4, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_dead,
            sb_dead,
            text=(
                "对准判定的像素容忍范围（屏幕像素）。"
                "略大可减少抖动；过小容易在中心附近来回微抖。"
            ),
        )

        lbl_max = ttk.Label(align_grid, text="单次上限")
        lbl_max.grid(row=5, column=0, sticky=tk.W, pady=3)
        sb_max = ttk.Spinbox(
            align_grid,
            textvariable=self.align_max_step_var,
            from_=1,
            to=500,
            increment=1,
            width=8,
        )
        sb_max.grid(row=5, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_max,
            sb_max,
            text=(
                "每一帧鼠标移动量的上限（像素），用来限制甩枪幅度。"
                "过大易 overshoot，过小转身慢。"
            ),
        )

        lbl_lead = ttk.Label(align_grid, text="移动预判(ms)")
        lbl_lead.grid(row=6, column=0, sticky=tk.W, pady=3)
        sb_lead = ttk.Spinbox(
            align_grid,
            textvariable=self.align_lead_ms_var,
            from_=0,
            to=250,
            increment=5,
            width=8,
        )
        sb_lead.grid(row=6, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_lead,
            sb_lead,
            text=(
                "根据目标移动速度做提前量。横移目标可适当加大；"
                "静止或卡点时可调小，避免「抢跑」过头。"
            ),
        )

        lbl_mode = ttk.Label(align_grid, text="移动模式")
        lbl_mode.grid(row=7, column=0, sticky=tk.W, pady=3)
        cb_mode = ttk.Combobox(
            align_grid,
            textvariable=self.align_move_mode_var,
            values=("send_input", "set_cursor"),
            width=10,
            state="readonly",
        )
        cb_mode.grid(row=7, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_mode,
            cb_mode,
            text=(
                "send_input：相对移动，多数 FPS 可用。"
                "set_cursor：绝对坐标移动，少数场景更顺手或被游戏屏蔽。"
            ),
        )

        lbl_steps = ttk.Label(align_grid, text="拆分步数")
        lbl_steps.grid(row=8, column=0, sticky=tk.W, pady=3)
        sb_steps = ttk.Spinbox(
            align_grid,
            textvariable=self.align_smooth_steps_var,
            from_=1,
            to=80,
            increment=1,
            width=8,
        )
        sb_steps.grid(row=8, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_steps,
            sb_steps,
            text=(
                "把一次修正拆成多步发送。FPS 常用 1～3；"
                "步数过大动作会变「绵」，体感延迟升高。"
            ),
        )

        lbl_dur = ttk.Label(align_grid, text="移动耗时")
        lbl_dur.grid(row=9, column=0, sticky=tk.W, pady=3)
        sb_dur = ttk.Spinbox(
            align_grid,
            textvariable=self.align_smooth_duration_var,
            from_=0.0,
            to=0.2,
            increment=0.005,
            width=8,
        )
        sb_dur.grid(row=9, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_dur,
            sb_dur,
            text=(
                "拆分移动的总跨度（秒）。与步数配合：总耗时÷步数≈每步间隔；"
                "FPS 宜短促。"
            ),
        )

        cb_inv_x = ttk.Checkbutton(
            align_grid,
            text="X 轴反向",
            variable=self.align_invert_x_var,
        )
        cb_inv_x.grid(row=10, column=0, sticky=tk.W, pady=3)
        cb_inv_y = ttk.Checkbutton(
            align_grid,
            text="Y 轴反向",
            variable=self.align_invert_y_var,
        )
        cb_inv_y.grid(row=10, column=1, sticky=tk.W, pady=3)
        bind_tooltip_text(
            cb_inv_x,
            text="水平方向与游戏视角相反时勾选。也可用「校准鼠标倍率」自动推断。",
        )
        bind_tooltip_text(cb_inv_y, text="垂直方向与游戏视角相反时勾选。")

        cb_click = ttk.Checkbutton(
            align_grid,
            text="对准后单击左键",
            variable=self.align_click_on_target_var,
        )
        cb_click.grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=3)
        bind_tooltip_text(
            cb_click,
            text=(
                "当人物中心连续多帧落在「停止范围」内、且队列中剩余微调移动已排空后，"
                "自动左键单击一次；目标再次甩开后再对准可再次触发。"
                "若游戏中仍不响应，多半是游戏屏蔽合成输入。"
            ),
        )

        preset_bar = ttk.Frame(mouse_tab)
        preset_bar.pack(fill=tk.X, pady=(0, 8))
        btn_preset_stable = ttk.Button(
            preset_bar,
            text="稳一点",
            command=lambda: self._apply_align_preset("stable"),
        )
        btn_preset_stable.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        bind_tooltip_text(
            btn_preset_stable,
            text="更平滑、更容易压住抖动，适合远距离或后坐较大的武器。",
        )
        btn_preset_balanced = ttk.Button(
            preset_bar,
            text="均衡",
            command=lambda: self._apply_align_preset("balanced"),
        )
        btn_preset_balanced.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        bind_tooltip_text(btn_preset_balanced, text="跟踪与稳定折中，多数 FPS 场景的起步档。")
        btn_preset_fast = ttk.Button(
            preset_bar,
            text="跟手",
            command=lambda: self._apply_align_preset("fast"),
        )
        btn_preset_fast.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        bind_tooltip_text(
            btn_preset_fast,
            text="响应更快、更贴目标移动；近距离或高机动时更明显，需小心过头。",
        )

        btn_calib = ttk.Button(
            mouse_tab,
            text="校准鼠标倍率",
            command=self._start_mouse_calibration,
        )
        btn_calib.pack(fill=tk.X, pady=(0, 8))
        bind_tooltip_text(
            btn_calib,
            text=(
                "通过小幅试探移动推断「游戏灵敏度 ↔ 像素位移」比例，"
                "自动给出移动倍率与轴向反向建议。运行中且画面内有静止人物时使用。"
            ),
        )

        btn_pick_target = ttk.Button(mouse_tab, text="选择目标标点", command=self._select_target)
        btn_pick_target.pack(fill=tk.X, pady=(0, 6))
        bind_tooltip_text(
            btn_pick_target,
            text="在屏幕上点选准星/十字参考点，写入「目标 X/Y」。",
        )

        btn_region_center = ttk.Button(
            mouse_tab, text="使用区域中心", command=self._target_region_center
        )
        btn_region_center.pack(fill=tk.X)
        bind_tooltip_text(
            btn_region_center,
            text="把目标标点设为当前检测区域的中心（与「区域」页宽高联动）。",
        )

        lbl_hotkey = ttk.Label(
            mouse_tab,
            text=f"全局热键：{HOTKEY_TOGGLE_ALIGN_NAME} 开关自动移动",
            wraplength=230,
        )
        lbl_hotkey.pack(anchor=tk.W, fill=tk.X, pady=(10, 0))
        bind_tooltip_text(
            lbl_hotkey,
            text=(
                f"按下 {HOTKEY_TOGGLE_ALIGN_NAME} 可快速开关「启用自动移动」，"
                "无需切回本窗口（注册失败时会改用轮询兜底）。"
            ),
        )

        detector_grid = ttk.Frame(detector_tab)
        detector_grid.pack(fill=tk.X)

        lbl_det_mode = ttk.Label(detector_grid, text="模式")
        lbl_det_mode.grid(row=0, column=0, sticky=tk.W, pady=3)
        self.mode_combo = ttk.Combobox(
            detector_grid,
            textvariable=self.detection_mode_var,
            values=DETECTION_MODES,
            width=18,
            state="readonly",
        )
        self.mode_combo.grid(row=0, column=1, sticky=tk.EW, pady=3)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_detection_mode_changed)
        bind_tooltip_text(
            lbl_det_mode,
            self.mode_combo,
            text=(
                "普通框检测或姿态关键点。"
                "FPS 人物对准一般用普通即可；姿态模式可配合过滤减少误检。"
            ),
        )

        lbl_det_model = ttk.Label(detector_grid, text="模型")
        lbl_det_model.grid(row=1, column=0, sticky=tk.W, pady=3)
        self.model_combo = ttk.Combobox(
            detector_grid,
            textvariable=self.model_var,
            values=self._model_values_for_mode(),
            width=18,
        )
        self.model_combo.grid(row=1, column=1, sticky=tk.EW, pady=3)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)
        bind_tooltip_text(
            lbl_det_model,
            self.model_combo,
            text="越大通常越准但更慢。实时跟枪可先 n/s；需要更远距离再换 m/l。",
        )

        lbl_det_dev = ttk.Label(detector_grid, text="设备")
        lbl_det_dev.grid(row=2, column=0, sticky=tk.W, pady=3)
        dev_combo = ttk.Combobox(
            detector_grid,
            textvariable=self.device_var,
            values=("auto", "0", "cpu"),
            width=8,
        )
        dev_combo.grid(row=2, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_det_dev,
            dev_combo,
            text=(
                "auto：有可用 GPU 则用显卡，否则 CPU。"
                "改动后需停止再点「开始」才会重建检测器。"
            ),
        )

        lbl_det_conf = ttk.Label(detector_grid, text="置信度")
        lbl_det_conf.grid(row=3, column=0, sticky=tk.W, pady=3)
        sb_det_conf = ttk.Spinbox(
            detector_grid,
            textvariable=self.conf_var,
            from_=0.05,
            to=0.95,
            increment=0.05,
            width=8,
        )
        sb_det_conf.grid(row=3, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_det_conf,
            sb_det_conf,
            text="人物检测框保留阈值：越低越容易检出但误报多；过高会丢远处小人。",
        )

        lbl_det_fps = ttk.Label(detector_grid, text="FPS")
        lbl_det_fps.grid(row=4, column=0, sticky=tk.W, pady=3)
        sb_det_fps = ttk.Spinbox(
            detector_grid,
            textvariable=self.fps_var,
            from_=1,
            to=120,
            increment=1,
            width=8,
        )
        sb_det_fps.grid(row=4, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_det_fps,
            sb_det_fps,
            text=(
                "检测循环的目标帧率上限。越高跟得越紧但占 CPU/GPU；"
                "若单次推理已超过帧间隔，实际 FPS 会低于此处。"
            ),
        )

        lbl_det_imgsz = ttk.Label(detector_grid, text="推理尺寸")
        lbl_det_imgsz.grid(row=5, column=0, sticky=tk.W, pady=3)
        cb_det_imgsz = ttk.Combobox(
            detector_grid,
            textvariable=self.imgsz_var,
            values=(320, 416, 512, 640, 768),
            width=8,
        )
        cb_det_imgsz.grid(row=5, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_det_imgsz,
            cb_det_imgsz,
            text=(
                "YOLO 输入边长。越大远处人越稳但更耗时；"
                "想减负可先试 512 / 416，再根据漏检微调。"
            ),
        )

        lbl_det_prevfps = ttk.Label(detector_grid, text="预览FPS")
        lbl_det_prevfps.grid(row=6, column=0, sticky=tk.W, pady=3)
        sb_det_prevfps = ttk.Spinbox(
            detector_grid,
            textvariable=self.preview_fps_var,
            from_=10,
            to=120,
            increment=10,
            width=8,
        )
        sb_det_prevfps.grid(row=6, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            lbl_det_prevfps,
            sb_det_prevfps,
            text="仅影响右侧预览刷新频率，不改变上面「检测 FPS」。预览关掉更省资源。",
        )

        cb_preview = ttk.Checkbutton(
            detector_grid,
            text="显示预览",
            variable=self.preview_enabled_var,
            command=self._toggle_preview,
        )
        cb_preview.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=3)
        bind_tooltip_text(cb_preview, text="开启后在右侧显示截取区域的实时预览；游戏中建议关闭以免挡视野。")

        cb_half = ttk.Checkbutton(
            detector_grid,
            text="半精度 FP16",
            variable=self.half_var,
        )
        cb_half.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=3)
        bind_tooltip_text(
            cb_half,
            text="GPU 推理时可加速并省显存；CPU 无意义。改动后需重新开始检测才生效。",
        )

        self.pose_filter_check = ttk.Checkbutton(
            detector_grid,
            text="姿态过滤",
            variable=self.pose_filter_enabled_var,
        )
        self.pose_filter_check.grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=(10, 3))
        bind_tooltip_text(
            self.pose_filter_check,
            text="仅在姿态模式下生效：按关键点数量与部位过滤误识别（例如只露手臂）。",
        )

        self.pose_conf_label = ttk.Label(detector_grid, text="关键点置信度")
        self.pose_conf_label.grid(row=10, column=0, sticky=tk.W, pady=3)
        self.pose_conf_spin = ttk.Spinbox(
            detector_grid,
            textvariable=self.pose_keypoint_conf_var,
            from_=0.05,
            to=0.95,
            increment=0.05,
            width=8,
        )
        self.pose_conf_spin.grid(row=10, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            self.pose_conf_label,
            self.pose_conf_spin,
            text="关键点置信度下限，低于则视为无效点。放宽可减少漏检，收紧可减少噪声。",
        )

        self.pose_min_label = ttk.Label(detector_grid, text="最少关键点")
        self.pose_min_label.grid(row=11, column=0, sticky=tk.W, pady=3)
        self.pose_min_spin = ttk.Spinbox(
            detector_grid,
            textvariable=self.pose_min_keypoints_var,
            from_=1,
            to=17,
            increment=1,
            width=8,
        )
        self.pose_min_spin.grid(row=11, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            self.pose_min_label,
            self.pose_min_spin,
            text="至少多少个有效关键点才算一个人。增大更严格，远处小人可能被判无效。",
        )

        self.pose_head_check = ttk.Checkbutton(
            detector_grid,
            text="必须有头部",
            variable=self.pose_require_head_var,
        )
        self.pose_head_check.grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=3)
        bind_tooltip_text(
            self.pose_head_check,
            text="要求至少看到一个头部关键点，减少「只看到手脚」的假人框。",
        )

        self.pose_body_check = ttk.Checkbutton(
            detector_grid,
            text="必须有肩/髋",
            variable=self.pose_require_body_var,
        )
        self.pose_body_check.grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=3)
        bind_tooltip_text(
            self.pose_body_check,
            text="要求肩部或髋部关键点至少一侧可见，更像完整人体轮廓时才保留。",
        )

        self.pose_small_check = ttk.Checkbutton(
            detector_grid,
            text="小目标放宽",
            variable=self.pose_relax_small_targets_var,
        )
        self.pose_small_check.grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=(8, 3))
        bind_tooltip_text(
            self.pose_small_check,
            text=(
                "远处框高度小于「小目标高度」时放宽姿态条件，避免小人被过滤。"
                "关闭则更苛刻、画面更干净但可能丢远处目标。"
            ),
        )

        self.pose_small_label = ttk.Label(detector_grid, text="小目标高度")
        self.pose_small_label.grid(row=15, column=0, sticky=tk.W, pady=3)
        self.pose_small_spin = ttk.Spinbox(
            detector_grid,
            textvariable=self.pose_small_target_height_var,
            from_=20,
            to=300,
            increment=5,
            width=8,
        )
        self.pose_small_spin.grid(row=15, column=1, sticky=tk.EW, pady=3)
        bind_tooltip_text(
            self.pose_small_label,
            self.pose_small_spin,
            text="判定「小目标」的检测框高度（像素）。越大则有越多目标走放宽分支。",
        )

        self._update_pose_controls_state()

        self.model_tree = ttk.Treeview(
            model_tab,
            columns=("model", "status", "size"),
            show="headings",
            height=12,
            selectmode="browse",
        )
        self.model_tree.heading("model", text="模型")
        self.model_tree.heading("status", text="状态")
        self.model_tree.heading("size", text="大小")
        self.model_tree.column("model", width=122, anchor=tk.W)
        self.model_tree.column("status", width=58, anchor=tk.CENTER)
        self.model_tree.column("size", width=58, anchor=tk.E)
        self.model_tree.pack(fill=tk.BOTH, expand=True)
        self.model_tree.bind("<Double-1>", lambda _event: self._use_selected_model())

        model_buttons = ttk.Frame(model_tab)
        model_buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(model_buttons, text="刷新", command=self._refresh_model_list).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6)
        )
        ttk.Button(model_buttons, text="使用所选", command=self._use_selected_model).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(model_tab, text="下载所选模型", command=self._download_selected_model).pack(
            fill=tk.X, pady=(8, 0)
        )
        self.download_progress = ttk.Progressbar(model_tab, mode="determinate", maximum=100)
        self.download_progress.pack(fill=tk.X, pady=(10, 0))
        self.download_status_var = tk.StringVar(value="选择模型后可下载。")
        ttk.Label(model_tab, textvariable=self.download_status_var, wraplength=230).pack(
            anchor=tk.W, fill=tk.X, pady=(6, 0)
        )
        self._refresh_model_list()

        btn_test_move = ttk.Button(test_tab, text="测试移动", command=self._schedule_move_test)
        btn_test_move.pack(fill=tk.X, pady=(0, 8))
        bind_tooltip_text(
            btn_test_move,
            text="约 1.5 秒后按当前「移动模式」向右发送一段测试位移，用于确认 SendInput / 光标模式是否生效。",
        )
        btn_test_click = ttk.Button(test_tab, text="测试单击", command=self._schedule_click_test)
        btn_test_click.pack(fill=tk.X, pady=(0, 8))
        bind_tooltip_text(btn_test_click, text="约 1.5 秒后模拟一次左键单击，确认按键注入是否正常。")
        btn_test_scroll = ttk.Button(test_tab, text="测试滚轮", command=self._schedule_scroll_test)
        btn_test_scroll.pack(fill=tk.X)
        bind_tooltip_text(btn_test_scroll, text="约 1.5 秒后发送一段滚轮事件，用于确认滚轮注入是否正常。")

        ttk.Separator(controls).pack(fill=tk.X, pady=10)
        ttk.Label(controls, textvariable=self.status_var, wraplength=250).pack(
            anchor=tk.W, fill=tk.X
        )

        self.canvas = tk.Canvas(self.preview_frame, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _model_values_for_mode(self, mode: Optional[str] = None) -> tuple[str, ...]:
        selected_mode = normalize_detection_mode(mode or self.detection_mode_var.get())
        if selected_mode == DETECTION_MODE_POSE:
            return POSE_MODEL_PRESETS
        return NORMAL_MODEL_PRESETS

    def _model_for_mode(self, mode: str) -> str:
        if normalize_detection_mode(mode) == DETECTION_MODE_POSE:
            return self.pose_model_name
        return self.normal_model_name

    def _remember_current_model(self) -> None:
        model_name = self.model_var.get().strip()
        if not model_name:
            return
        if is_pose_model_name(model_name):
            self.pose_model_name = model_name
        else:
            self.normal_model_name = model_name

    def _ensure_model_matches_mode(self) -> None:
        mode = normalize_detection_mode(self.detection_mode_var.get())
        self.detection_mode_var.set(mode)
        model_name = self.model_var.get().strip()
        if not model_name or model_mode(model_name) != mode:
            self.model_var.set(self._model_for_mode(mode))

    def _on_detection_mode_changed(self, _event=None) -> None:
        self._remember_current_model()
        mode = normalize_detection_mode(self.detection_mode_var.get())
        self.detection_mode_var.set(mode)
        self.model_combo.configure(values=self._model_values_for_mode(mode))
        self.model_var.set(self._model_for_mode(mode))
        self._update_pose_controls_state()
        self._save_settings()

    def _on_model_changed(self, _event=None) -> None:
        model_name = self.model_var.get().strip()
        expected_mode = model_mode(model_name)
        if expected_mode != normalize_detection_mode(self.detection_mode_var.get()):
            self.detection_mode_var.set(expected_mode)
            self.model_combo.configure(values=self._model_values_for_mode(expected_mode))
            self._update_pose_controls_state()
        self._remember_current_model()
        self._save_settings()

    def _update_pose_controls_state(self) -> None:
        state = tk.NORMAL if self.detection_mode_var.get() == DETECTION_MODE_POSE else tk.DISABLED
        for widget in (
            self.pose_filter_check,
            self.pose_conf_spin,
            self.pose_min_spin,
            self.pose_head_check,
            self.pose_body_check,
            self.pose_small_check,
            self.pose_small_spin,
        ):
            widget.configure(state=state)
        label_state = "normal" if state == tk.NORMAL else "disabled"
        for widget in (
            self.pose_conf_label,
            self.pose_min_label,
            self.pose_small_label,
        ):
            widget.configure(state=label_state)

    def _select_region(self) -> None:
        RegionSelector(self.root, self._set_region)

    def _select_target(self) -> None:
        TargetPointSelector(self.root, self._set_target_point)

    def _set_region(self, left: int, top: int, width: int, height: int) -> None:
        self.left_var.set(left)
        self.top_var.set(top)
        self.width_var.set(width)
        self.height_var.set(height)
        self._save_settings()
        self.status_var.set(f"检测区域已设置：{left}, {top}, {width}x{height}")

    def _set_target_point(self, x: int, y: int) -> None:
        self.target_x_var.set(x)
        self.target_y_var.set(y)
        self._save_settings()
        self.status_var.set(f"目标标点已设置：{x}, {y}")

    def _target_region_center(self) -> None:
        try:
            region = self._read_region()
        except ValueError as exc:
            messagebox.showerror("无效检测区域", str(exc))
            return
        self._set_target_point(
            region["left"] + region["width"] // 2,
            region["top"] + region["height"] // 2,
        )

    def _toggle_preview(self) -> None:
        self.preview_enabled = bool(self.preview_enabled_var.get())
        self._save_settings()
        if self.preview_enabled:
            self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        else:
            self.preview_frame.pack_forget()
            self.canvas.delete("all")
            self.preview_image = None

    def _apply_align_preset(self, preset: str) -> None:
        presets = {
            "stable": (ALIGN_CONTROL_FAST, 0.42, 10, 80, 28, 3, 0.018, "稳一点(FPS)"),
            "balanced": (ALIGN_CONTROL_FAST, 0.58, 8, 110, 38, 2, 0.014, "均衡(FPS)"),
            "fast": (ALIGN_CONTROL_DIRECT, 0.78, 6, 150, 52, 1, 0.010, "跟手(FPS)"),
        }
        (
            control_mode,
            sensitivity,
            deadzone,
            max_step,
            lead_ms,
            smooth_steps,
            smooth_duration,
            label,
        ) = presets[preset]
        self.align_control_mode_var.set(control_mode)
        self.align_sensitivity_var.set(sensitivity)
        self.align_deadzone_var.set(deadzone)
        self.align_max_step_var.set(max_step)
        self.align_lead_ms_var.set(lead_ms)
        self.align_smooth_steps_var.set(smooth_steps)
        self.align_smooth_duration_var.set(smooth_duration)
        self.mouse_mover.clear_pending()
        self.align_settled = False
        try:
            self.align_config = self._read_align_config()
        except (tk.TclError, ValueError):
            pass
        self._save_settings()
        self.status_var.set(f"已应用鼠标预设：{label}")

    def _start_mouse_calibration(self) -> None:
        if not self.running:
            messagebox.showinfo("校准鼠标倍率", "请先开始检测，并让画面里有一个静止人物。")
            return

        if self.calibration_after_id is not None:
            self.root.after_cancel(self.calibration_after_id)
            self.calibration_after_id = None
        self.mouse_mover.clear_pending()
        self.align_settled = False
        self.align_enabled_var.set(False)
        self.calibration_message = "校准准备：3 秒后开始，请松开鼠标并保持画面静止"
        self.status_var.set("校准准备：3 秒后开始，请松开鼠标并保持人物静止。")
        self.calibration_after_id = self.root.after(3000, self._begin_mouse_calibration)

    def _begin_mouse_calibration(self) -> None:
        self.calibration_after_id = None
        if not self.running:
            self.calibration_message = ""
            return

        self.mouse_mover.clear_pending()
        self.align_settled = False
        self.calibration_message = "校准中：请保持人物和视角静止"
        with self.calibration_lock:
            self.calibration_state = CalibrationState(
                pulses=[(50, 0), (-50, 0), (0, 50), (0, -50)]
            )
        self.status_var.set("校准中：请保持人物和视角静止，不要动鼠标。")

    def _refresh_model_list(self) -> None:
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        current_model = self.model_var.get().strip()
        for model_name in MODEL_PRESETS:
            status, size = local_model_status(model_name)
            self.model_tree.insert(
                "",
                tk.END,
                iid=model_name,
                values=(model_name, status, size),
            )
        if current_model in MODEL_PRESETS:
            self.model_tree.selection_set(current_model)
            self.model_tree.see(current_model)

    def _selected_model_name(self) -> Optional[str]:
        selection = self.model_tree.selection()
        if not selection:
            self.download_status_var.set("请先选择一个模型。")
            return None
        return selection[0]

    def _use_selected_model(self) -> None:
        model_name = self._selected_model_name()
        if model_name is None:
            return
        selected_mode = model_mode(model_name)
        self.detection_mode_var.set(selected_mode)
        self.model_combo.configure(values=self._model_values_for_mode(selected_mode))
        self.model_var.set(model_name)
        self._remember_current_model()
        self._update_pose_controls_state()
        self._save_settings()
        status, _size = local_model_status(model_name)
        if status != "已下载":
            self.download_status_var.set(f"{model_name} 未下载，下载后再使用。")
        else:
            self.download_status_var.set(f"已选择{selected_mode}：{model_name}")

    def _download_selected_model(self) -> None:
        model_name = self._selected_model_name()
        if model_name is None:
            return
        if self.downloading_model is not None:
            self.download_status_var.set(f"正在下载 {self.downloading_model}，请稍候。")
            return
        status, _size = local_model_status(model_name)
        if status == "已下载":
            self.download_status_var.set(f"{model_name} 已下载。")
            self.download_progress["value"] = 100
            return

        self.downloading_model = model_name
        self.download_progress["value"] = 0
        self.download_status_var.set(f"开始下载 {model_name} ...")
        self.download_worker = threading.Thread(
            target=self._download_model_worker,
            args=(model_name,),
            daemon=True,
        )
        self.download_worker.start()

    def _download_model_worker(self, model_name: str) -> None:
        target_path = model_file_path(model_name)
        temp_path = target_path.with_suffix(target_path.suffix + ".download")
        url = model_download_url(model_name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                with temp_path.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        downloaded += len(chunk)
                        self.download_queue.put(("progress", model_name, downloaded, total))
            if target_path.exists():
                target_path.unlink()
            temp_path.replace(target_path)
            self.download_queue.put(("done", model_name, target_path.stat().st_size))
        except Exception as exc:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            self.download_queue.put(("error", model_name, str(exc)))

    def _poll_downloads(self) -> None:
        try:
            while True:
                message = self.download_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _kind, model_name, downloaded, total = message
                    if total:
                        percent = min(100.0, downloaded * 100.0 / total)
                        self.download_progress["value"] = percent
                        self.download_status_var.set(
                            f"{model_name} 下载中：{percent:.1f}% "
                            f"({format_bytes(downloaded)} / {format_bytes(total)})"
                        )
                    else:
                        self.download_progress["value"] = 0
                        self.download_status_var.set(
                            f"{model_name} 下载中：{format_bytes(downloaded)}"
                        )
                elif kind == "done":
                    _kind, model_name, size = message
                    self.downloading_model = None
                    self.download_progress["value"] = 100
                    self.download_status_var.set(
                        f"{model_name} 下载完成：{format_bytes(size)}"
                    )
                    self._refresh_model_list()
                elif kind == "error":
                    _kind, model_name, error = message
                    self.downloading_model = None
                    self.download_progress["value"] = 0
                    self.download_status_var.set(f"{model_name} 下载失败：{error}")
                    self._refresh_model_list()
        except queue.Empty:
            pass

    def _toggle(self) -> None:
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        try:
            self._ensure_model_matches_mode()
            region = self._read_region()
            self.align_config = self._read_align_config()
            pose_filter = self._read_pose_filter_config()
        except ValueError as exc:
            messagebox.showerror("无效设置", str(exc))
            return
        self._remember_current_model()
        self._save_settings()

        try:
            self._ensure_selected_model_ready()
            self.detector = PersonDetector(
                self.model_var.get().strip(),
                self.conf_var.get(),
                self.device_var.get(),
                self._read_imgsz(),
                bool(self.half_var.get()),
                pose_filter,
            )
            self.detector_model_key = self._current_detector_model_key()
            self.restart_warning = ""
        except Exception as exc:
            messagebox.showerror("检测器错误", str(exc))
            return

        self.running = True
        self.capture_stop_event = threading.Event()
        self.start_button.configure(text="暂停")
        self.status_var.set(f"正在运行：{self.detector.backend}")
        self.worker = threading.Thread(
            target=self._capture_loop,
            args=(region, self.capture_stop_event),
            daemon=True,
        )
        self.worker.start()

    def _stop(self) -> None:
        self.running = False
        if self.capture_stop_event is not None:
            self.capture_stop_event.set()
        self.mouse_mover.clear_pending()
        self.locked_detection_center = None
        self.tracked_center = None
        self.tracked_velocity = (0.0, 0.0)
        self.tracked_time = None
        if self.calibration_after_id is not None:
            self.root.after_cancel(self.calibration_after_id)
            self.calibration_after_id = None
        with self.calibration_lock:
            self.calibration_state = None
        self.calibration_message = ""
        self.align_settled = False
        self._align_deadzone_stable_frames = 0
        self._align_click_consumed = False
        self._save_settings()
        self.start_button.configure(text="开始")
        self.status_var.set("已停止")

    def _ensure_selected_model_ready(self) -> None:
        model_name = self.model_var.get().strip()
        if model_name in MODEL_PRESETS:
            status, _size = local_model_status(model_name)
            if status != "已下载":
                raise ValueError(f"{model_name} 尚未下载，请先在“模型”页下载。")

    def _schedule_move_test(self) -> None:
        self.status_var.set("1.5 秒后测试移动，请把鼠标放到目标页面上。")
        self.root.after(1500, lambda: self._run_mouse_test("move"))

    def _schedule_click_test(self) -> None:
        self.status_var.set("1.5 秒后测试单击，请把鼠标放到目标页面上。")
        self.root.after(1500, lambda: self._run_mouse_test("click"))

    def _schedule_scroll_test(self) -> None:
        self.status_var.set("1.5 秒后测试滚轮，请把鼠标放到目标页面上。")
        self.root.after(1500, lambda: self._run_mouse_test("scroll"))

    def _run_mouse_test(self, kind: str) -> None:
        try:
            config = self._read_align_config()
        except ValueError as exc:
            messagebox.showerror("无效设置", str(exc))
            return

        if kind == "scroll":
            self.mouse_mover.scroll(-5)
            action = "滚轮"
        elif kind == "click":
            self.mouse_mover.click_left()
            action = "单击"
        else:
            self.mouse_mover.move_relative(
                180,
                0,
                config.move_mode,
                max(config.smooth_steps, 10),
                max(config.smooth_duration, 0.08),
            )
            action = "移动"

        if self.mouse_mover.last_error:
            self.status_var.set(f"{action}测试失败：{self.mouse_mover.last_error}")
        else:
            self.status_var.set(f"{action}测试已发送。")

    def _read_region(self) -> dict[str, int]:
        left = int(self.left_var.get())
        top = int(self.top_var.get())
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        if width < 20 or height < 20:
            raise ValueError("宽度和高度都必须至少为 20 像素。")
        return {"left": left, "top": top, "width": width, "height": height}

    def _read_imgsz(self) -> int:
        imgsz = int(self.imgsz_var.get())
        if imgsz < 160:
            raise ValueError("推理尺寸必须至少为 160。")
        return imgsz

    def _read_pose_filter_config(self) -> PoseFilterConfig:
        min_keypoints = int(self.pose_min_keypoints_var.get())
        keypoint_confidence = float(self.pose_keypoint_conf_var.get())
        small_target_height = int(self.pose_small_target_height_var.get())
        if min_keypoints < 1 or min_keypoints > 17:
            raise ValueError("最少关键点必须在 1 到 17 之间。")
        if keypoint_confidence <= 0 or keypoint_confidence > 1:
            raise ValueError("关键点置信度必须在 0 到 1 之间。")
        if small_target_height < 20:
            raise ValueError("小目标高度必须至少为 20。")
        return PoseFilterConfig(
            enabled=(
                normalize_detection_mode(self.detection_mode_var.get()) == DETECTION_MODE_POSE
                and bool(self.pose_filter_enabled_var.get())
            ),
            min_keypoints=min_keypoints,
            keypoint_confidence=keypoint_confidence,
            require_head=bool(self.pose_require_head_var.get()),
            require_body=bool(self.pose_require_body_var.get()),
            relax_small_targets=bool(self.pose_relax_small_targets_var.get()),
            small_target_height=small_target_height,
        )

    def _current_detector_model_key(self) -> tuple[str, str, bool]:
        return (
            self.model_var.get().strip(),
            self.device_var.get().strip(),
            bool(self.half_var.get()),
        )

    def _refresh_runtime_settings(self) -> None:
        try:
            self._ensure_model_matches_mode()
            target_fps = max(1, int(self.fps_var.get()))
            confidence = float(self.conf_var.get())
            imgsz = self._read_imgsz()
            pose_filter = self._read_pose_filter_config()
            detector_model_key = self._current_detector_model_key()
        except (tk.TclError, ValueError):
            return

        with self.runtime_lock:
            self.runtime_target_fps = target_fps
            self.runtime_confidence = confidence
            self.runtime_imgsz = imgsz
            self.runtime_pose_filter = pose_filter
            self.runtime_detector_model_key = detector_model_key

    def _runtime_snapshot(
        self,
    ) -> tuple[int, float, int, PoseFilterConfig, tuple[str, str, bool], bool]:
        with self.runtime_lock:
            return (
                self.runtime_target_fps,
                self.runtime_confidence,
                self.runtime_imgsz,
                self.runtime_pose_filter,
                self.runtime_detector_model_key,
                self.preview_enabled,
            )

    def _apply_runtime_detector_settings(
        self,
        confidence: float,
        imgsz: int,
        pose_filter: PoseFilterConfig,
        detector_model_key: tuple[str, str, bool],
    ) -> bool:
        if self.detector is None:
            return True
        if self.detector_model_key != detector_model_key:
            self.restart_warning = "模型/设备/半精度已改动，需停止后重新开始才生效。"
        else:
            self.restart_warning = ""

        self.detector.update_runtime(confidence, imgsz, pose_filter)
        return True

    def _preview_interval_ms(self) -> int:
        try:
            preview_fps = int(self.preview_fps_var.get())
        except (tk.TclError, ValueError):
            preview_fps = 60
        preview_fps = max(10, min(120, preview_fps))
        return max(1, int(round(1000 / preview_fps)))

    def _read_align_config(self) -> MouseAlignConfig:
        sensitivity = float(self.align_sensitivity_var.get())
        deadzone = int(self.align_deadzone_var.get())
        max_step = int(self.align_max_step_var.get())
        cooldown = float(self.align_cooldown_var.get())
        control_mode = self.align_control_mode_var.get()
        lead_ms = int(self.align_lead_ms_var.get())
        move_mode = self.align_move_mode_var.get()
        smooth_steps = int(self.align_smooth_steps_var.get())
        smooth_duration = float(self.align_smooth_duration_var.get())
        if sensitivity <= 0:
            raise ValueError("移动倍率必须大于 0。")
        if deadzone < 0:
            raise ValueError("停止范围必须大于或等于 0。")
        if max_step < 1:
            raise ValueError("单次上限必须至少为 1。")
        if cooldown < 0:
            raise ValueError("冷却时间必须大于或等于 0。")
        if lead_ms < 0:
            raise ValueError("移动预判必须大于或等于 0。")
        if move_mode not in {"send_input", "set_cursor"}:
            raise ValueError("移动模式必须是 send_input 或 set_cursor。")
        if control_mode not in ALIGN_CONTROL_MODES:
            raise ValueError("追踪模式必须是 稳定、快速 或 直接。")
        if smooth_steps < 1:
            raise ValueError("拆分步数必须至少为 1。")
        if smooth_duration < 0:
            raise ValueError("移动耗时必须大于或等于 0。")
        return MouseAlignConfig(
            enabled=bool(self.align_enabled_var.get()),
            target_x=int(self.target_x_var.get()),
            target_y=int(self.target_y_var.get()),
            sensitivity=sensitivity,
            deadzone=deadzone,
            max_step=max_step,
            cooldown=cooldown,
            control_mode=control_mode,
            lead_ms=lead_ms,
            invert_x=bool(self.align_invert_x_var.get()),
            invert_y=bool(self.align_invert_y_var.get()),
            move_mode=move_mode,
            smooth_steps=smooth_steps,
            smooth_duration=smooth_duration,
            click_on_target=bool(self.align_click_on_target_var.get()),
        )

    def _capture_loop(
        self,
        region: dict[str, int],
        stop_event: threading.Event,
    ) -> None:
        fps_window_started = time.perf_counter()
        fps_window_frames = 0
        actual_fps = 0.0
        last_status_update = 0.0
        next_frame_time = time.perf_counter()
        with mss.mss() as screen_capture:
            while not stop_event.is_set():
                (
                    target_fps,
                    confidence,
                    imgsz,
                    pose_filter,
                    detector_model_key,
                    preview_enabled,
                ) = self._runtime_snapshot()
                delay = 1.0 / target_fps
                now_for_pacing = time.perf_counter()
                if next_frame_time < now_for_pacing - delay:
                    next_frame_time = now_for_pacing
                self._apply_runtime_detector_settings(
                    confidence,
                    imgsz,
                    pose_filter,
                    detector_model_key,
                )
                started = time.perf_counter()
                grab_started = time.perf_counter()
                frame = np.asarray(screen_capture.grab(region))
                screenshot_ms = (time.perf_counter() - grab_started) * 1000.0

                convert_started = time.perf_counter()
                rgb_frame = bgra_to_rgb(frame)
                convert_ms = (time.perf_counter() - convert_started) * 1000.0

                inference_started = time.perf_counter()
                detections = self.detector.detect(rgb_frame) if self.detector else []
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                target = (self.align_config.target_x, self.align_config.target_y)
                selected_detection = self._select_single_detection(detections, region, target)
                shown_detections = [selected_detection] if selected_detection else []

                calibrating = self._update_mouse_calibration(selected_detection, region)
                if not calibrating:
                    self._align_mouse(shown_detections, region)
                elapsed = time.perf_counter() - started
                fps_window_frames += 1
                now = time.perf_counter()
                fps_window_elapsed = now - fps_window_started
                if fps_window_elapsed >= 1.0:
                    actual_fps = fps_window_frames / fps_window_elapsed
                    fps_window_started = now
                    fps_window_frames = 0

                stats = {
                    "actual_fps": actual_fps or (1.0 / elapsed if elapsed > 0 else 0.0),
                    "target_fps": float(target_fps),
                    "screenshot_ms": screenshot_ms,
                    "convert_ms": convert_ms,
                    "inference_ms": inference_ms,
                    "loop_ms": elapsed * 1000.0,
                }
                if preview_enabled:
                    self._put_frame(rgb_frame, shown_detections, region, stats)
                elif now - last_status_update >= 0.1:
                    self._put_status(shown_detections, region, stats)
                    last_status_update = now
                next_frame_time += delay
                sleep_until(next_frame_time)

    def _select_single_detection(
        self,
        detections: list[Detection],
        region: dict[str, int],
        target: tuple[int, int],
    ) -> Optional[Detection]:
        if not detections:
            self.locked_detection_center = None
            self.tracked_center = None
            self.tracked_velocity = (0.0, 0.0)
            self.tracked_time = None
            return None

        best = min(
            detections,
            key=lambda detection: self._distance_to_target(detection, region, target),
        )
        if self.locked_detection_center is not None:
            locked = self._nearest_to_screen_point(
                detections,
                region,
                self.locked_detection_center,
            )
            if locked is not None:
                best_distance = self._distance_to_target(best, region, target)
                locked_distance = self._distance_to_target(locked, region, target)
                if locked_distance <= best_distance * 1.35 + 900:
                    best = locked

        cx, cy = best.center
        screen_center = (region["left"] + cx, region["top"] + cy)
        self.locked_detection_center = screen_center
        self._update_target_motion(screen_center)
        return best

    def _update_target_motion(self, screen_center: tuple[int, int]) -> None:
        now = time.perf_counter()
        x, y = float(screen_center[0]), float(screen_center[1])
        if self.tracked_center is None or self.tracked_time is None:
            self.tracked_center = (x, y)
            self.tracked_velocity = (0.0, 0.0)
            self.tracked_time = now
            return

        previous_x, previous_y = self.tracked_center
        dt = now - self.tracked_time
        jump = math.hypot(x - previous_x, y - previous_y)
        if dt <= 0.005 or dt > 0.5 or jump > 250:
            self.tracked_center = (x, y)
            self.tracked_velocity = (0.0, 0.0)
            self.tracked_time = now
            return

        instant_vx = (x - previous_x) / dt
        instant_vy = (y - previous_y) / dt
        old_vx, old_vy = self.tracked_velocity
        self.tracked_velocity = (
            old_vx * 0.55 + instant_vx * 0.45,
            old_vy * 0.55 + instant_vy * 0.45,
        )
        self.tracked_center = (x, y)
        self.tracked_time = now

    def _predicted_target_center(
        self,
        fallback_center: tuple[int, int],
        lead_ms: int,
    ) -> tuple[float, float]:
        if self.tracked_center is None or lead_ms <= 0:
            return float(fallback_center[0]), float(fallback_center[1])
        lead_seconds = min(0.25, max(0.0, lead_ms / 1000.0))
        vx, vy = self.tracked_velocity
        x, y = self.tracked_center
        return x + vx * lead_seconds, y + vy * lead_seconds

    def _update_mouse_calibration(
        self,
        detection: Optional[Detection],
        region: dict[str, int],
    ) -> bool:
        with self.calibration_lock:
            state = self.calibration_state
        if state is None:
            return False

        self.mouse_mover.clear_pending()
        now = time.perf_counter()
        if detection is None:
            state.phase = "stabilizing"
            state.stable_center = None
            state.stable_since = now
            state.status = "校准中：等待检测到人物"
            self.calibration_message = state.status
            return True

        cx, cy = detection.center
        center = (float(region["left"] + cx), float(region["top"] + cy))

        if state.phase == "stabilizing":
            if state.stable_center is None:
                state.stable_center = center
                state.stable_since = now
            elif math.hypot(center[0] - state.stable_center[0], center[1] - state.stable_center[1]) <= 2.0:
                if now - state.stable_since >= 0.35:
                    state.phase = "send_pulse"
                    state.baseline_center = center
            else:
                state.stable_center = center
                state.stable_since = now
            state.status = "校准中：等待目标稳定"
            self.calibration_message = state.status
            return True

        if state.phase == "send_pulse":
            pulse_x, pulse_y = state.pulses[state.pulse_index]
            state.baseline_center = center
            state.pulse_sent_at = now
            state.phase = "measure"
            state.status = (
                f"校准中：测试 {state.pulse_index + 1}/{len(state.pulses)} "
                f"({pulse_x},{pulse_y})"
            )
            self.mouse_mover.move_relative(
                pulse_x,
                pulse_y,
                self.align_move_mode_var.get(),
                1,
                0.0,
                replace_pending=True,
            )
            self.calibration_message = state.status
            return True

        if state.phase == "measure":
            if now - state.pulse_sent_at < 0.22:
                self.calibration_message = state.status
                return True

            pulse_x, pulse_y = state.pulses[state.pulse_index]
            if state.baseline_center is not None:
                delta_x = center[0] - state.baseline_center[0]
                delta_y = center[1] - state.baseline_center[1]
                if pulse_x:
                    sample = delta_x / pulse_x
                    if abs(sample) >= 0.02:
                        state.samples_x.append(sample)
                if pulse_y:
                    sample = delta_y / pulse_y
                    if abs(sample) >= 0.02:
                        state.samples_y.append(sample)

            state.pulse_index += 1
            if state.pulse_index >= len(state.pulses):
                self._finish_mouse_calibration(state)
                return True

            state.phase = "stabilizing"
            state.stable_center = center
            state.stable_since = now
            state.status = "校准中：等待目标稳定"
            self.calibration_message = state.status
            return True

        return True

    def _finish_mouse_calibration(self, state: CalibrationState) -> None:
        x_gain = self._median_sample(state.samples_x or [])
        y_gain = self._median_sample(state.samples_y or [])
        if x_gain is None and y_gain is None:
            self.calibration_queue.put(
                ("calibration_error", "校准失败：画面变化太小，请靠近目标或增大游戏鼠标灵敏度后重试。")
            )
        else:
            gains = [abs(gain) for gain in (x_gain, y_gain) if gain is not None]
            sensitivity = max(0.05, min(5.0, sum(1.0 / gain for gain in gains) / len(gains)))
            invert_x = bool(x_gain is not None and x_gain > 0)
            invert_y = bool(y_gain is not None and y_gain > 0)
            self.calibration_queue.put(
                ("calibration_done", sensitivity, invert_x, invert_y, x_gain, y_gain)
            )
        with self.calibration_lock:
            self.calibration_state = None
        self.calibration_message = ""

    def _median_sample(self, samples: list[float]) -> Optional[float]:
        valid = sorted(sample for sample in samples if abs(sample) >= 0.02)
        if not valid:
            return None
        return valid[len(valid) // 2]

    def _nearest_to_screen_point(
        self,
        detections: list[Detection],
        region: dict[str, int],
        point: tuple[int, int],
    ) -> Optional[Detection]:
        if not detections:
            return None
        return min(
            detections,
            key=lambda detection: self._distance_to_target(detection, region, point),
        )

    def _align_mouse(self, detections: list[Detection], region: dict[str, int]) -> None:
        config = self.align_config

        if not detections:
            self.align_settled = False
            self.mouse_mover.clear_pending()
            self._align_deadzone_stable_frames = 0
            self._align_click_consumed = False
            return

        if not config.enabled:
            self.align_settled = False
            self.mouse_mover.clear_pending()
            self._align_deadzone_stable_frames = 0
            self._align_click_consumed = False
            return

        now = time.perf_counter()

        target = (config.target_x, config.target_y)
        nearest = min(
            detections,
            key=lambda detection: self._distance_to_target(detection, region, target),
        )
        cx, cy = nearest.center
        screen_cx = region["left"] + cx
        screen_cy = region["top"] + cy
        predicted_cx, predicted_cy = self._predicted_target_center(
            (screen_cx, screen_cy),
            config.lead_ms,
        )
        dx = predicted_cx - config.target_x
        dy = predicted_cy - config.target_y
        distance = math.hypot(dx, dy)

        if distance <= config.deadzone:
            self.mouse_mover.clear_pending()
            self._align_deadzone_stable_frames += 1
            click_stable_need = 4
            if config.click_on_target:
                pending_ok = self.mouse_mover.pending_magnitude() < 1.0
                if (
                    pending_ok
                    and self._align_deadzone_stable_frames >= click_stable_need
                    and not self._align_click_consumed
                    and now - self.last_target_click >= 0.28
                ):
                    self.mouse_mover.click_left()
                    self.last_target_click = now
                    self._align_click_consumed = True
            self.align_settled = True
            return

        reengage_distance = max(config.deadzone * 2.5, config.deadzone + 12)
        if self.align_settled and distance <= reengage_distance:
            self.mouse_mover.clear_pending()
            return
        self.align_settled = False
        self._align_deadzone_stable_frames = 0
        self._align_click_consumed = False

        if now - self.last_mouse_move < config.cooldown:
            return

        move_x = align_axis_move(
            int(round(dx)),
            config.sensitivity,
            config.max_step,
            config.deadzone,
            config.control_mode,
        )
        move_y = align_axis_move(
            int(round(dy)),
            config.sensitivity,
            config.max_step,
            config.deadzone,
            config.control_mode,
        )
        if move_x == 0 and move_y == 0:
            self.mouse_mover.clear_pending()
            return
        if config.invert_x:
            move_x = -move_x
        if config.invert_y:
            move_y = -move_y
        self.mouse_mover.move_relative(
            move_x,
            move_y,
            config.move_mode,
            config.smooth_steps,
            config.smooth_duration,
            replace_pending=True,
        )
        self.last_mouse_move = now

    def _distance_to_target(
        self,
        detection: Detection,
        region: dict[str, int],
        target: tuple[int, int],
    ) -> int:
        cx, cy = detection.center
        screen_cx = region["left"] + cx
        screen_cy = region["top"] + cy
        return (screen_cx - target[0]) ** 2 + (screen_cy - target[1]) ** 2

    def _put_frame(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
        region: dict[str, int],
        stats: dict[str, float],
    ) -> None:
        item = (frame, list(detections), region, stats, time.strftime("%H:%M:%S"))
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(item)

    def _put_status(
        self,
        detections: Iterable[Detection],
        region: dict[str, int],
        stats: dict[str, float],
    ) -> None:
        item = (None, list(detections), region, stats, time.strftime("%H:%M:%S"))
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(item)

    def _poll_frames(self) -> None:
        self._poll_downloads()
        self._poll_hotkeys()
        self._poll_calibration()
        self._refresh_runtime_settings()
        try:
            self.align_config = self._read_align_config()
        except (tk.TclError, ValueError):
            pass

        try:
            while True:
                frame, detections, region, stats, stamp = self.frame_queue.get_nowait()
                if frame is None or not self.preview_enabled:
                    self._draw_status(detections, region, stats, stamp)
                else:
                    self._draw_frame(frame, detections, region, stats, stamp)
        except queue.Empty:
            pass
        self.root.after(self._preview_interval_ms(), self._poll_frames)

    def _poll_calibration(self) -> None:
        try:
            while True:
                message = self.calibration_queue.get_nowait()
                kind = message[0]
                if kind == "calibration_done":
                    (
                        _kind,
                        sensitivity,
                        invert_x,
                        invert_y,
                        x_gain,
                        y_gain,
                    ) = message
                    self.align_sensitivity_var.set(round(float(sensitivity), 3))
                    self.align_invert_x_var.set(bool(invert_x))
                    self.align_invert_y_var.set(bool(invert_y))
                    self.align_settled = False
                    self.mouse_mover.clear_pending()
                    try:
                        self.align_config = self._read_align_config()
                    except (tk.TclError, ValueError):
                        pass
                    self._save_settings()
                    self.status_var.set(
                        "校准完成："
                        f"移动倍率={sensitivity:.3f}，"
                        f"X反向={'开' if invert_x else '关'}，"
                        f"Y反向={'开' if invert_y else '关'}，"
                        f"X响应={x_gain if x_gain is not None else '-'}，"
                        f"Y响应={y_gain if y_gain is not None else '-'}"
                    )
                elif kind == "calibration_error":
                    _kind, text = message
                    self.calibration_message = ""
                    self.status_var.set(text)
        except queue.Empty:
            pass

    def _poll_hotkeys(self) -> None:
        try:
            while True:
                kind, payload = self.hotkey_queue.get_nowait()
                if kind == "toggle_align":
                    self._toggle_auto_align_from_hotkey()
                elif kind == "hotkey_error":
                    self.status_var.set(f"全局热键不可用：{payload}")
        except queue.Empty:
            pass

    def _on_f8_pressed(self, _event=None) -> str:
        self._toggle_auto_align_from_hotkey()
        return "break"

    def _toggle_auto_align_from_hotkey(self) -> None:
        enabled = not bool(self.align_enabled_var.get())
        self.align_enabled_var.set(enabled)
        if not enabled:
            self.mouse_mover.clear_pending()
            self.align_settled = False
        try:
            self.align_config = self._read_align_config()
        except (tk.TclError, ValueError):
            pass
        self._save_settings()
        state = "开启" if enabled else "关闭"
        self.status_var.set(f"{HOTKEY_TOGGLE_ALIGN_NAME}：自动鼠标移动已{state}")

    def _draw_status(
        self,
        detections: list[Detection],
        region: dict[str, int],
        stats: dict[str, float],
        stamp: str,
    ) -> None:
        self._set_status(detections, region, stats, stamp)

    def _save_settings(self) -> None:
        try:
            self._remember_current_model()
            settings = {
                "left": int(self.left_var.get()),
                "top": int(self.top_var.get()),
                "width": int(self.width_var.get()),
                "height": int(self.height_var.get()),
                "confidence": float(self.conf_var.get()),
                "fps": int(self.fps_var.get()),
                "imgsz": int(self.imgsz_var.get()),
                "preview_fps": int(self.preview_fps_var.get()),
                "target_x": int(self.target_x_var.get()),
                "target_y": int(self.target_y_var.get()),
                "preview": bool(self.preview_enabled_var.get()),
                "half": bool(self.half_var.get()),
                "detection_mode": normalize_detection_mode(self.detection_mode_var.get()),
                "model": self.model_var.get().strip(),
                "normal_model": self.normal_model_name,
                "pose_model": self.pose_model_name,
                "device": self.device_var.get().strip(),
                "pose_filter": bool(self.pose_filter_enabled_var.get()),
                "pose_min_keypoints": int(self.pose_min_keypoints_var.get()),
                "pose_keypoint_confidence": float(self.pose_keypoint_conf_var.get()),
                "pose_require_head": bool(self.pose_require_head_var.get()),
                "pose_require_body": bool(self.pose_require_body_var.get()),
                "pose_relax_small_targets": bool(self.pose_relax_small_targets_var.get()),
                "pose_small_target_height": int(self.pose_small_target_height_var.get()),
                "align_control_mode": self.align_control_mode_var.get(),
                "align_sensitivity": float(self.align_sensitivity_var.get()),
                "align_deadzone": int(self.align_deadzone_var.get()),
                "align_max_step": int(self.align_max_step_var.get()),
                "auto_align": bool(self.align_enabled_var.get()),
                "align_lead_ms": int(self.align_lead_ms_var.get()),
                "align_cooldown": float(self.align_cooldown_var.get()),
                "align_invert_x": bool(self.align_invert_x_var.get()),
                "align_invert_y": bool(self.align_invert_y_var.get()),
                "align_move_mode": self.align_move_mode_var.get(),
                "align_smooth_steps": int(self.align_smooth_steps_var.get()),
                "align_smooth_duration": float(self.align_smooth_duration_var.get()),
                "align_click_on_target": bool(self.align_click_on_target_var.get()),
            }
        except (tk.TclError, ValueError):
            return
        save_settings(settings)

    def _draw_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        region: dict[str, int],
        stats: dict[str, float],
        stamp: str,
    ) -> None:
        image = Image.fromarray(frame)
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        scale = min(canvas_width / image.width, canvas_height / image.height)
        display_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(display_size, Image.Resampling.BILINEAR)

        self.preview_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        offset_x = (canvas_width - display_size[0]) // 2
        offset_y = (canvas_height - display_size[1]) // 2
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.preview_image)

        target_x = int(self.target_x_var.get())
        target_y = int(self.target_y_var.get())
        target_region_x = target_x - region["left"]
        target_region_y = target_y - region["top"]
        if 0 <= target_region_x <= region["width"] and 0 <= target_region_y <= region["height"]:
            target_display_x = offset_x + target_region_x * scale
            target_display_y = offset_y + target_region_y * scale
            self.canvas.create_line(
                target_display_x - 12,
                target_display_y,
                target_display_x + 12,
                target_display_y,
                fill="#ffcc00",
                width=2,
            )
            self.canvas.create_line(
                target_display_x,
                target_display_y - 12,
                target_display_x,
                target_display_y + 12,
                fill="#ffcc00",
                width=2,
            )
            self.canvas.create_oval(
                target_display_x - 6,
                target_display_y - 6,
                target_display_x + 6,
                target_display_y + 6,
                outline="#ffcc00",
                width=2,
            )

        for detection in detections:
            x1 = offset_x + detection.x1 * scale
            y1 = offset_y + detection.y1 * scale
            x2 = offset_x + detection.x2 * scale
            y2 = offset_y + detection.y2 * scale
            cx, cy = detection.center
            dcx = offset_x + cx * scale
            dcy = offset_y + cy * scale
            screen_cx = region["left"] + cx
            screen_cy = region["top"] + cy

            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff66", width=2)
            if detection.keypoints:
                threshold = (
                    self.detector.pose_filter.keypoint_confidence
                    if self.detector
                    else 0.35
                )
                for point_x, point_y, point_conf in detection.keypoints:
                    if point_conf < threshold or (point_x == 0 and point_y == 0):
                        continue
                    display_x = offset_x + point_x * scale
                    display_y = offset_y + point_y * scale
                    self.canvas.create_oval(
                        display_x - 3,
                        display_y - 3,
                        display_x + 3,
                        display_y + 3,
                        fill="#ffcc00",
                        outline="",
                    )
            self.canvas.create_line(dcx - 8, dcy, dcx + 8, dcy, fill="#00ff66", width=2)
            self.canvas.create_line(dcx, dcy - 8, dcx, dcy + 8, fill="#00ff66", width=2)
            self.canvas.create_text(
                x1 + 4,
                max(offset_y + 10, y1 - 12),
                anchor=tk.W,
                fill="#00ff66",
                text=f"{detection.confidence:.2f} 中心=({screen_cx},{screen_cy})",
            )

        self._set_status(detections, region, stats, stamp)

    def _set_status(
        self,
        detections: list[Detection],
        region: dict[str, int],
        stats: dict[str, float],
        stamp: str,
    ) -> None:
        self.status_var.set(
            f"{stamp} | 实际FPS：{stats['actual_fps']:.1f}/{stats['target_fps']:.0f} | "
            f"截图：{stats.get('screenshot_ms', 0.0):.1f}ms | "
            f"转换：{stats.get('convert_ms', 0.0):.1f}ms | "
            f"推理：{stats['inference_ms']:.1f}ms | "
            f"总：{stats.get('loop_ms', 0.0):.1f}ms | "
            f"设备：{self.detector.device if self.detector else '-'} | "
            f"FP16：{'开' if self.detector and self.detector.half else '关'} | "
            f"{self.restart_warning + ' | ' if self.restart_warning else ''}"
            f"{self.calibration_message + ' | ' if self.calibration_message else ''}"
            f"鼠标：{'开' if self.align_enabled_var.get() else '关'} | "
            f"热键：{HOTKEY_TOGGLE_ALIGN_NAME} | "
            f"人数：{len(detections)} | 区域："
            f"{region['left']},{region['top']} {region['width']}x{region['height']} | "
            f"目标：{int(self.target_x_var.get())},{int(self.target_y_var.get())}"
        )

    def _on_close(self) -> None:
        self.running = False
        if self.capture_stop_event is not None:
            self.capture_stop_event.set()
        self._save_settings()
        self.hotkey_listener.close()
        self.mouse_mover.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    detection_mode = normalize_detection_mode(
        saved_str(settings, "detection_mode", DETECTION_MODE_NORMAL)
    )
    normal_model = saved_str(
        settings,
        "normal_model",
        preferred_model(NORMAL_MODEL_PRESETS, "yolo26n.pt"),
    )
    pose_model = saved_str(
        settings,
        "pose_model",
        preferred_model(POSE_MODEL_PRESETS, "yolo26n-pose.pt"),
    )
    current_model = saved_str(
        settings,
        "model",
        pose_model if detection_mode == DETECTION_MODE_POSE else normal_model,
    )
    if model_mode(current_model) != detection_mode:
        current_model = pose_model if detection_mode == DETECTION_MODE_POSE else normal_model
    parser = argparse.ArgumentParser(
        description="监测屏幕区域中的人物。",
        add_help=False,
    )
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")
    parser.add_argument("--left", type=int, default=saved_int(settings, "left", 0), help="检测区域左侧坐标")
    parser.add_argument("--top", type=int, default=saved_int(settings, "top", 0), help="检测区域顶部坐标")
    parser.add_argument("--width", type=int, default=saved_int(settings, "width", 960), help="检测区域宽度")
    parser.add_argument("--height", type=int, default=saved_int(settings, "height", 540), help="检测区域高度")
    parser.add_argument(
        "--confidence",
        type=float,
        default=saved_float(settings, "confidence", 0.35),
        help="人物检测置信度",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=saved_int(settings, "fps", 60),
        help="检测帧率",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=saved_int(settings, "imgsz", 640),
        help="YOLO 推理尺寸，越小越省 CPU",
    )
    parser.add_argument(
        "--preview-fps",
        type=int,
        default=saved_int(settings, "preview_fps", 60),
        help="预览刷新率，最高 120",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "preview", False),
        help="是否显示预览窗口",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "half", True),
        help="GPU 推理时启用半精度 FP16",
    )
    parser.add_argument(
        "--detection-mode",
        choices=DETECTION_MODES,
        default=detection_mode,
        help="检测模式：普通模型或姿态模型",
    )
    parser.add_argument("--model", default=current_model, help="YOLO 模型路径")
    parser.add_argument("--normal-model", default=normal_model, help="普通模式使用的 YOLO 模型")
    parser.add_argument("--pose-model", default=pose_model, help="姿态模式使用的 YOLO 模型")
    parser.add_argument(
        "--pose-filter",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "pose_filter", False),
        help="姿态模型启用关键点过滤，减少只露手时误识别",
    )
    parser.add_argument(
        "--pose-min-keypoints",
        type=int,
        default=saved_int(settings, "pose_min_keypoints", 3),
        help="姿态过滤要求的最少有效关键点数",
    )
    parser.add_argument(
        "--pose-keypoint-confidence",
        type=float,
        default=saved_float(settings, "pose_keypoint_confidence", 0.20),
        help="姿态关键点最低置信度",
    )
    parser.add_argument(
        "--pose-require-head",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "pose_require_head", False),
        help="姿态过滤要求至少看到一个头部关键点",
    )
    parser.add_argument(
        "--pose-require-body",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "pose_require_body", False),
        help="姿态过滤要求至少看到肩部或髋部关键点",
    )
    parser.add_argument(
        "--pose-relax-small-targets",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "pose_relax_small_targets", True),
        help="远处小目标使用更宽松的姿态过滤",
    )
    parser.add_argument(
        "--pose-small-target-height",
        type=int,
        default=saved_int(settings, "pose_small_target_height", 90),
        help="小目标放宽的框高度阈值，单位像素",
    )
    parser.add_argument(
        "--auto-align",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "auto_align", False),
        help="启动时启用自动鼠标移动",
    )
    parser.add_argument("--target-x", type=int, default=saved_optional_int(settings, "target_x"), help="目标标点 X 坐标")
    parser.add_argument("--target-y", type=int, default=saved_optional_int(settings, "target_y"), help="目标标点 Y 坐标")
    parser.add_argument(
        "--align-control-mode",
        choices=ALIGN_CONTROL_MODES,
        default=saved_str(settings, "align_control_mode", ALIGN_CONTROL_FAST),
        help="鼠标追踪模式：稳定、快速、直接",
    )
    parser.add_argument(
        "--align-sensitivity",
        type=float,
        default=saved_float(settings, "align_sensitivity", 0.58),
        help="鼠标移动倍率",
    )
    parser.add_argument(
        "--align-deadzone",
        type=int,
        default=saved_int(settings, "align_deadzone", 8),
        help="停止移动的目标范围，单位像素",
    )
    parser.add_argument(
        "--align-max-step",
        type=int,
        default=saved_int(settings, "align_max_step", 110),
        help="每次鼠标移动的最大像素上限",
    )
    parser.add_argument(
        "--align-lead-ms",
        type=int,
        default=saved_int(settings, "align_lead_ms", 38),
        help="移动目标预判时间，单位毫秒",
    )
    parser.add_argument(
        "--align-cooldown",
        type=float,
        default=saved_float(settings, "align_cooldown", 0.0),
        help="鼠标移动冷却时间，单位秒",
    )
    parser.add_argument(
        "--align-invert-x",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "align_invert_x", False),
        help="反转 X 轴移动方向",
    )
    parser.add_argument(
        "--align-invert-y",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "align_invert_y", False),
        help="反转 Y 轴移动方向",
    )
    parser.add_argument(
        "--align-move-mode",
        choices=("send_input", "set_cursor"),
        default=saved_str(settings, "align_move_mode", "send_input"),
        help="鼠标移动模式：send_input 或 set_cursor",
    )
    parser.add_argument(
        "--align-smooth-steps",
        type=int,
        default=saved_int(settings, "align_smooth_steps", 2),
        help="每次移动拆分成几步发送",
    )
    parser.add_argument(
        "--align-smooth-duration",
        type=float,
        default=saved_float(settings, "align_smooth_duration", 0.014),
        help="每次移动持续秒数",
    )
    parser.add_argument(
        "--align-click-on-target",
        action=argparse.BooleanOptionalAction,
        default=saved_bool(settings, "align_click_on_target", False),
        help="目标点对准人物中心时单击左键一次",
    )
    parser.add_argument(
        "--device",
        default=saved_str(settings, "device", "auto"),
        help="YOLO 设备：auto、cpu、0、1 等。默认：auto",
    )
    return parser.parse_args()


def main() -> None:
    enable_high_resolution_timer()
    args = parse_args()
    root = tk.Tk()
    PersonMonitorApp(root, args)
    try:
        root.mainloop()
    finally:
        disable_high_resolution_timer()


if __name__ == "__main__":
    main()
