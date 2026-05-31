"""
collab.py — Autonomous pick-and-place with collaborative safety.

Merges pickCVBlock.py (CV-guided pick/place state machine) with
hand_collab.py (MediaPipe hand detection for speed reduction).

Two cameras run simultaneously:
  - Laptop camera  (LAPTOP_CAM  = 0) : detects plates and red blocks
  - Orbbec Astra   (ORBBEC_CAM  = 1) : watches for human hands via MediaPipe

When a hand is visible to the Orbbec camera, the arm's PTP speed is halved
(SPEED_SLOW). When the hand leaves, full speed (SPEED_NORMAL) is restored.
This runs in a background thread so it never blocks the pick/place loop.

State machine (same as pickCVBlock.py):
  scanning plate → scanning target → pick place → scanning plate → …

Setup
-----
1. Connect the Dobot and set COM_PORT below.
2. Run getTransformationMatrix.py and calibrateCamera.py first to generate
   HomographyMatrix.npy and camera_params.npz.
3. source .venv/bin/activate && python collab.py
"""

import os
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "mediapipe is required. Run: pip install mediapipe"
    ) from exc

import dobotArm
import lib.DobotDllType as dType

# ── Configuration ─────────────────────────────────────────────────────────────

COM_PORT    = "COM5"   # change to match DobotLab
LAPTOP_CAM  = 0        # laptop built-in camera — used for plate/block detection
ORBBEC_CAM  = 1        # Orbbec Astra UVC index — used for hand safety monitoring

# Pick/place parameters
Z_SAFE           = 40    # clearance height (mm) when moving horizontally
Z_PICK           = -25   # height (mm) to lower gripper for pick
STABILITY_LIMIT  = 60    # consecutive stable frames before locking detection (~2s at 30fps)
PIXEL_TOLERANCE  = 10    # max pixel drift to count as stationary

# Collaborative safety speed (% of max, passed to SetPTPCommonParams)
SPEED_NORMAL = 50        # normal operating speed
SPEED_SLOW   = 25        # speed when hand is detected (50% of normal)

# MediaPipe confidence thresholds
MIN_DETECT_CONF = 0.60
MIN_TRACK_CONF  = 0.50

# ── MediaPipe setup ───────────────────────────────────────────────────────────

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Shared state for collab monitor ──────────────────────────────────────────

_hand_visible = False        # set by background thread, read by main thread
_speed_slow   = False        # tracks current speed state
_monitor_lock = threading.Lock()


# ── Collaborative safety monitor (background thread) ─────────────────────────

class CollabMonitor(threading.Thread):
    """
    Reads the Orbbec Astra camera in a daemon thread and adjusts the Dobot's
    PTP speed based on hand presence. Runs independently of the main loop.
    """

    def __init__(self, api):
        super().__init__(daemon=True)
        self.api     = api
        self._stop   = threading.Event()
        self.cap     = None

    def stop(self):
        self._stop.set()

    def run(self):
        global _hand_visible, _speed_slow

        # Open Orbbec camera
        self.cap = cv2.VideoCapture(ORBBEC_CAM)
        if not self.cap.isOpened():
            print(f"[collab] WARNING: Could not open Orbbec camera at index {ORBBEC_CAM}. "
                  "Hand safety monitoring disabled.")
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[collab] WARNING: Orbbec camera opened but returned no frames. "
                  "Hand safety monitoring disabled.")
            self.cap.release()
            return
        print(f"[collab] Orbbec Astra opened at index {ORBBEC_CAM}. Safety monitor active.")

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECT_CONF,
            min_tracking_confidence=MIN_TRACK_CONF,
        ) as detector:

            while not self._stop.is_set():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = detector.process(rgb)
                frame.flags.writeable = True

                hand_detected = bool(results.multi_hand_landmarks)

                # Draw landmarks on Orbbec feed
                if hand_detected:
                    for hand_lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                # Adjust arm speed
                with _monitor_lock:
                    if hand_detected and not _speed_slow:
                        dType.SetPTPCommonParams(self.api, SPEED_SLOW, SPEED_SLOW, isQueued=0)
                        _speed_slow   = True
                        _hand_visible = True
                        print("[collab] Hand detected — speed reduced to 50%.")
                    elif not hand_detected and _speed_slow:
                        dType.SetPTPCommonParams(self.api, SPEED_NORMAL, SPEED_NORMAL, isQueued=0)
                        _speed_slow   = False
                        _hand_visible = False
                        print("[collab] Hand cleared — speed restored to normal.")
                    else:
                        _hand_visible = hand_detected

                # Draw HUD on Orbbec window
                _draw_collab_hud(frame, hand_detected, _speed_slow)
                cv2.imshow("Collab Safety — Orbbec", frame)
                cv2.waitKey(1)

        self.cap.release()

    def restore_speed(self):
        dType.SetPTPCommonParams(self.api, SPEED_NORMAL, SPEED_NORMAL, isQueued=0)


def _draw_collab_hud(frame, hand_visible, speed_slow):
    h, w = frame.shape[:2]
    GREEN = (0, 220, 0)
    RED   = (0, 0, 220)
    AMBER = (0, 170, 255)
    GREY  = (180, 180, 180)
    CYAN  = (220, 220, 0)

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 90), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "COLLAB SAFETY MONITOR", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

    hand_text, hand_col = ("HAND DETECTED", RED) if hand_visible else ("NO HAND", GREEN)
    cv2.putText(frame, hand_text, (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_col, 2)

    spd_text, spd_col = ("SPEED: 50% SLOW", AMBER) if speed_slow else ("SPEED: NORMAL", GREEN)
    cv2.putText(frame, spd_text, (w - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, spd_col, 2)

    cv2.putText(frame, "Press Q in main window to quit", (w - 310, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)


# ── Camera helpers ────────────────────────────────────────────────────────────

def open_laptop_camera():
    cap = cv2.VideoCapture(LAPTOP_CAM)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open laptop camera at index {LAPTOP_CAM}. "
            "Check macOS > System Settings > Privacy & Security > Camera."
        )
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Laptop camera {LAPTOP_CAM} opened but returned no frames.")
    print(f"Laptop camera opened at index {LAPTOP_CAM}.")
    return cap, frame


# ── Coordinate mapping ────────────────────────────────────────────────────────

def pixel_to_robot(u, v, H):
    p  = np.array([u, v, 1.0])
    xy = H @ p
    xy /= xy[2]
    return float(xy[0]), float(xy[1])


# ── Coordinate smoothing ──────────────────────────────────────────────────────

_coord_histories = defaultdict(lambda: deque(maxlen=5))

def get_stable_target(idx, new_x, new_y):
    _coord_histories[idx].append((new_x, new_y))
    return np.mean(_coord_histories[idx], axis=0)


# ── State machine ─────────────────────────────────────────────────────────────

_machine_state = "scanning plate"

def next_state():
    global _machine_state
    transitions = {
        "scanning plate":   "scanning target",
        "scanning target":  "pick place",
        "pick place":       "scanning plate",
    }
    _machine_state = transitions.get(_machine_state, "scanning plate")


# ── Phase 1: detect drop zones (plates) ──────────────────────────────────────

def phase_detect_plates(cap, H_matrix, map1, map2):
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame         = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 150,
            param1=100, param2=35, minRadius=25, maxRadius=55,
        )

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count        = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show collab status on main window
        _draw_main_status(display_frame)
        cv2.imshow("Pick & Place — Laptop Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

        if stability_counter >= STABILITY_LIMIT:
            print(f"[PHASE 1] Locked {len(current_list)} drop zones.")
            return current_list


# ── Phase 2: detect red targets ───────────────────────────────────────────────

def phase_detect_targets(cap, H_matrix, map1, map2):
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_count        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame         = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        hsv  = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([0,   100,  50]), np.array([10,  255, 255])) +
                cv2.inRange(hsv, np.array([160, 120,  70]), np.array([180, 255, 255])))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid        = [c for c in contours if cv2.contourArea(c) > 150]

        current_list = []
        for idx, cnt in enumerate(valid):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy         = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                rx, ry         = pixel_to_robot(cx, cy, H_matrix)
                sx, sy         = get_stable_target(idx, rx, ry)
                current_list.append((sx, sy))
                cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)

        if len(current_list) != 0:
            if len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count        = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if progress < 100 else (255, 255, 0), 2)

        _draw_main_status(display_frame)
        cv2.imshow("Pick & Place — Laptop Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

        if stability_counter >= STABILITY_LIMIT:
            print(f"[PHASE 2] Locked {len(current_list)} targets.")
            return current_list


# ── Phase 3: pick and place ───────────────────────────────────────────────────

def phase_execute_batch(api, pick_list, drop_list):
    time.sleep(0.5)

    if not pick_list or not drop_list:
        print("[PHASE 3] Missing targets or drop zones — aborting.")
        return False

    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y = pick_list[i]
        drop_x, drop_y = drop_list[i]
        print(f"  Task {i + 1}: pick {pick_x:.1f},{pick_y:.1f}  →  drop {drop_x:.1f},{drop_y:.1f}")

        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
        dobotArm.close_gripper(api)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    # If more picks than drops, pile remaining onto first drop zone
    if len(pick_list) > len(drop_list):
        drop_x, drop_y = drop_list[0]
        for i in range(len(drop_list), len(pick_list)):
            pick_x, pick_y = pick_list[i]
            print(f"  Overflow task {i + 1}: pick {pick_x:.1f},{pick_y:.1f}  →  drop {drop_x:.1f},{drop_y:.1f}")
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
            time.sleep(0.2)
            dobotArm.close_gripper(api)
            time.sleep(0.5)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("[PHASE 3] Batch complete.")
    return True


# ── HUD overlay for main (laptop) window ─────────────────────────────────────

def _draw_main_status(frame):
    """Overlay collab safety status on the laptop camera window."""
    h, w = frame.shape[:2]
    with _monitor_lock:
        hand = _hand_visible
        slow = _speed_slow

    color = (0, 0, 220) if hand else (0, 220, 0)
    label = "HAND IN WORKSPACE — SLOW" if hand else "NO HAND — NORMAL SPEED"
    cv2.putText(frame, label, (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Check required calibration files
    for fname in ("HomographyMatrix.npy", "camera_params.npz"):
        if not Path(fname).exists():
            raise RuntimeError(
                f"{fname} not found. "
                "Run getTransformationMatrix.py and calibrateCamera.py first."
            )

    H_matrix = np.load("HomographyMatrix.npy")
    data     = np.load("camera_params.npz")
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]

    # Open laptop camera and compute undistort maps once
    cap, first_frame = open_laptop_camera()
    h, w = first_frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
    )

    # Connect to Dobot
    api = dType.load()
    print(f"Connecting to Dobot on {COM_PORT} …")
    dobotArm.initialize_robot(api)
    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)
    print("Robot ready.\n")

    # Start collaborative safety monitor on Orbbec camera
    monitor = CollabMonitor(api)
    monitor.start()

    drop_zone   = None
    pick_target = None

    try:
        while True:
            global _machine_state

            if _machine_state == "scanning plate":
                drop_zone = phase_detect_plates(cap, H_matrix, map1, map2)
                if drop_zone is None:
                    break          # user pressed Q
                next_state()

            elif _machine_state == "scanning target":
                pick_target = phase_detect_targets(cap, H_matrix, map1, map2)
                if pick_target is None:
                    break          # user pressed Q
                next_state()

            elif _machine_state == "pick place":
                completed = phase_execute_batch(api, drop_zone, pick_target)
                if completed:
                    next_state()
                else:
                    break

    finally:
        print("\nShutting down…")
        monitor.stop()
        monitor.restore_speed()
        cap.release()
        cv2.destroyAllWindows()
        dType.DisconnectDobot(api)
        print("Done.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(f"\nError: {err}", file=sys.stderr)
        print("\nRun with the project venv:", file=sys.stderr)
        print("  source .venv/bin/activate && python collab.py", file=sys.stderr)
        raise SystemExit(1) from err
