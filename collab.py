"""
collab.py — Autonomous pick-and-place with collaborative hand safety.

Single Orbbec Astra camera (Windows UVC) handles everything:
  - Detects metal plates (drop zones) via HoughCircles
  - Detects red blocks (pick targets) via HSV masking
  - Detects human hands via MediaPipe — halves arm speed while hand is visible

State machine:
  scanning plate → scanning target → pick place → scanning plate → …

Setup
-----
1. Set COM_PORT to match DobotLab (e.g. "COM5").
2. Set ORBBEC_CAM to the camera index Windows assigns the Orbbec Astra
   (usually 0 if it is the only camera, 1 if a built-in webcam is present).
3. Run getTransformationMatrix.py and calibrateCamera.py first to generate
   HomographyMatrix.npy and camera_params.npz.
4. .venv\\Scripts\\activate && python collab.py
"""

import os
import sys
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

COM_PORT   = "COM5"   # change to match DobotLab
ORBBEC_CAM = 0        # Orbbec Astra UVC index (0 if only camera, 1 if laptop webcam also present)

# Pick/place parameters
Z_SAFE          = 40    # clearance height (mm) when moving horizontally
Z_PICK          = -25   # height (mm) to lower gripper for pick
STABILITY_LIMIT = 60    # consecutive stable frames before locking detection (~2s at 30fps)

# Collaborative safety speed (% of max)
SPEED_NORMAL = 50
SPEED_SLOW   = 25       # applied while hand is visible (50% of normal)

# MediaPipe confidence thresholds
MIN_DETECT_CONF = 0.60
MIN_TRACK_CONF  = 0.50

# ── MediaPipe setup ───────────────────────────────────────────────────────────

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Coordinate smoothing ──────────────────────────────────────────────────────

_coord_histories = defaultdict(lambda: deque(maxlen=5))

def get_stable_target(idx, new_x, new_y):
    _coord_histories[idx].append((new_x, new_y))
    return np.mean(_coord_histories[idx], axis=0)


# ── Coordinate mapping ────────────────────────────────────────────────────────

def pixel_to_robot(u, v, H):
    p  = np.array([u, v, 1.0])
    xy = H @ p
    xy /= xy[2]
    return float(xy[0]), float(xy[1])


# ── State machine ─────────────────────────────────────────────────────────────

_machine_state = "scanning plate"

def next_state():
    global _machine_state
    _machine_state = {
        "scanning plate":  "scanning target",
        "scanning target": "pick place",
        "pick place":      "scanning plate",
    }.get(_machine_state, "scanning plate")


# ── HUD helpers ───────────────────────────────────────────────────────────────

def _draw_hud(frame, phase_text, hand_visible, speed_slow):
    h, w = frame.shape[:2]
    GREEN = (0, 220, 0)
    RED   = (0, 0, 220)
    AMBER = (0, 170, 255)
    GREY  = (180, 180, 180)
    CYAN  = (220, 220, 0)

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 95), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, phase_text, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

    hand_text, hand_col = ("HAND DETECTED", RED) if hand_visible else ("NO HAND", GREEN)
    cv2.putText(frame, hand_text, (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_col, 2)

    spd_text, spd_col = ("SPEED: 50% SLOW", AMBER) if speed_slow else ("SPEED: NORMAL", GREEN)
    cv2.putText(frame, spd_text, (w - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, spd_col, 2)

    cv2.putText(frame, "Press Q to quit", (w - 190, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)


# ── Hand detection (runs on every frame) ─────────────────────────────────────

def check_hand_and_update_speed(api, frame, detector, speed_slow):
    """
    Run MediaPipe on frame, draw landmarks, adjust arm speed.
    Returns (hand_visible, speed_slow, annotated_frame).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = detector.process(rgb)
    frame.flags.writeable = True

    hand_visible = bool(results.multi_hand_landmarks)

    if hand_visible:
        for hand_lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    if hand_visible and not speed_slow:
        dType.SetPTPCommonParams(api, SPEED_SLOW, SPEED_SLOW, isQueued=0)
        speed_slow = True
        print("[collab] Hand detected — speed reduced to 50%.")
    elif not hand_visible and speed_slow:
        dType.SetPTPCommonParams(api, SPEED_NORMAL, SPEED_NORMAL, isQueued=0)
        speed_slow = False
        print("[collab] Hand cleared — speed restored to normal.")

    return hand_visible, speed_slow


# ── Phase 1: detect drop zones (metal plates) ────────────────────────────────

def phase_detect_plates(cap, api, H_matrix, map1, map2, detector):
    print("\n[PHASE 1] Scanning for metal plates. Waiting for stability...")
    stability_counter = 0
    last_count        = 0
    speed_slow        = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame         = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        # CV detection
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
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Hand detection on same frame
        hand_visible, speed_slow = check_hand_and_update_speed(
            api, display_frame, detector, speed_slow
        )
        _draw_hud(display_frame, "PHASE 1 — SCANNING PLATES", hand_visible, speed_slow)
        cv2.imshow("Collab Pick & Place — Orbbec", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None, speed_slow

        if stability_counter >= STABILITY_LIMIT:
            print(f"[PHASE 1] Locked {len(current_list)} metal plates.")
            return current_list, speed_slow


# ── Phase 2: detect red targets (red blocks) ─────────────────────────────────

def phase_detect_targets(cap, api, H_matrix, map1, map2, detector):
    print("\n[PHASE 2] Scanning for red blocks. Waiting for stability...")
    stability_counter = 0
    last_count        = 0
    speed_slow        = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame         = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        # CV detection
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
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                rx, ry = pixel_to_robot(cx, cy, H_matrix)
                sx, sy = get_stable_target(idx, rx, ry)
                current_list.append((sx, sy))
                cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)

        if len(current_list) != 0:
            if len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count        = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if progress < 100 else (255, 255, 0), 2)

        # Hand detection on same frame
        hand_visible, speed_slow = check_hand_and_update_speed(
            api, display_frame, detector, speed_slow
        )
        _draw_hud(display_frame, "PHASE 2 — SCANNING RED BLOCKS", hand_visible, speed_slow)
        cv2.imshow("Collab Pick & Place — Orbbec", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None, speed_slow

        if stability_counter >= STABILITY_LIMIT:
            print(f"[PHASE 2] Locked {len(current_list)} red blocks.")
            return current_list, speed_slow


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
        print(f"  Task {i + 1}: pick ({pick_x:.1f}, {pick_y:.1f})  →  drop ({drop_x:.1f}, {drop_y:.1f})")

        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
        dobotArm.close_gripper(api)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    # More picks than drops — pile remaining onto first drop zone
    if len(pick_list) > len(drop_list):
        drop_x, drop_y = drop_list[0]
        for i in range(len(drop_list), len(pick_list)):
            pick_x, pick_y = pick_list[i]
            print(f"  Overflow {i + 1}: pick ({pick_x:.1f}, {pick_y:.1f})  →  drop ({drop_x:.1f}, {drop_y:.1f})")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for fname in ("HomographyMatrix.npy", "camera_params.npz"):
        if not Path(fname).exists():
            raise RuntimeError(
                f"{fname} not found. "
                "Run getTransformationMatrix.py and calibrateCamera.py first."
            )

    H_matrix      = np.load("HomographyMatrix.npy")
    data          = np.load("camera_params.npz")
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]

    # Open Orbbec camera
    cap = cv2.VideoCapture(ORBBEC_CAM)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open Orbbec Astra at camera index {ORBBEC_CAM}. "
            "Ensure it is plugged in and Windows has granted camera access. "
            "Update ORBBEC_CAM if the OS assigned a different index."
        )
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Orbbec camera opened but returned no frames.")
    print(f"Orbbec Astra opened at index {ORBBEC_CAM}.")

    # Compute undistort maps once
    h, w     = first_frame.shape[:2]
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

    drop_zone   = None
    pick_target = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECT_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    ) as detector:
        try:
            while True:
                if _machine_state == "scanning plate":
                    drop_zone, _ = phase_detect_plates(
                        cap, api, H_matrix, map1, map2, detector
                    )
                    if drop_zone is None:
                        break
                    next_state()

                elif _machine_state == "scanning target":
                    pick_target, _ = phase_detect_targets(
                        cap, api, H_matrix, map1, map2, detector
                    )
                    if pick_target is None:
                        break
                    next_state()

                elif _machine_state == "pick place":
                    completed = phase_execute_batch(api, pick_target, drop_zone)
                    if completed:
                        next_state()
                    else:
                        break

        finally:
            print("\nShutting down…")
            dType.SetPTPCommonParams(api, SPEED_NORMAL, SPEED_NORMAL, isQueued=0)
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
        print("  .venv\\Scripts\\activate && python collab.py", file=sys.stderr)
        raise SystemExit(1) from err
