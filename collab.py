"""
collab.py — Autonomous pick-and-place with collaborative hand safety.

Single Orbbec Astra camera (Windows UVC) handles everything:
  - Tracks a light green block continuously for visual reference
  - Detects metal plates (drop zones) via HoughCircles
  - Detects red blocks (pick targets) via HSV masking
  - Detects human hands via MediaPipe — halves arm speed while hand is visible

State machine:
  scanning plate → scanning target → pick place → wait_for_input
                                         ↑ SPACE ────────────────┘

Setup
-----
1. Set COM_PORT to match DobotLab (e.g. "COM5").
2. Set ORBBEC_CAM to the index Windows assigns the Orbbec Astra
   (usually 1 if a built-in webcam is also present, 0 if it is the only camera).
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
ORBBEC_CAM = 1        # Orbbec Astra UVC index (1 if laptop webcam also present, else 0)

# Pick/place parameters
Z_SAFE          = 40    # clearance height (mm) when moving horizontally
Z_PICK          = -30   # height (mm) to lower gripper for pick
STABILITY_LIMIT = 60    # consecutive stable frames before locking (~2 s at 30 fps)
PIXEL_TOLERANCE = 10    # max pixel drift to count as stationary

# Collaborative safety speed (% of max)
SPEED_NORMAL = 50
SPEED_SLOW   = 25       # applied while hand is visible (50 % of normal)

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
    # Return position directly — averaging by contour index causes jumps when
    # contour ordering changes between frames.
    return new_x, new_y


# ── Coordinate mapping ────────────────────────────────────────────────────────

def pixel_to_robot(u, v, H):
    p  = np.array([u, v, 1.0])
    xy = H @ p
    xy /= xy[2]
    return float(xy[0]), float(xy[1])


# ── Position validation ───────────────────────────────────────────────────────

def is_valid_position(x, y, z):
    """Return False and print a warning if (x, y, z) is outside safe workspace."""
    if not (-100 < x < 400):
        print(f"[ERROR] X out of range: {x:.1f}")
        return False
    if not (-250 < y < 250):
        print(f"[ERROR] Y out of range: {y:.1f}")
        return False
    if not (-1100 < z < 150):
        print(f"[ERROR] Z out of range: {z:.1f}")
        return False
    return True


# ── Green block tracker ───────────────────────────────────────────────────────

def track_green_block(frame, display_frame, H_matrix):
    """Detect light green block and overlay its robot coordinates on display_frame."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            rx, ry = pixel_to_robot(cx, cy, H_matrix)
            cv2.drawContours(display_frame, [cnt], -1, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Green: ({rx:.1f}, {ry:.1f})",
                        (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# ── HUD ───────────────────────────────────────────────────────────────────────

def _draw_hud(frame, phase_text, hand_visible, speed_slow):
    h, w  = frame.shape[:2]
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

    cv2.putText(frame, "Q: quit", (w - 110, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)


# ── Hand detection (inline, runs every frame) ─────────────────────────────────

def check_hand_and_update_speed(api, frame, detector, speed_slow):
    """Run MediaPipe on frame, draw landmarks, adjust arm speed if needed.
    Returns (hand_visible, updated_speed_slow)."""
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

        # Plate detection via Hough circles
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 150,
            param1=100, param2=35, minRadius=15, maxRadius=65,
        )

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        # Green block tracking overlay
        track_green_block(frame, display_frame, H_matrix)

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
            print("\n[INFO] Q pressed. Exiting...")
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

        # Red block detection
        hsv  = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([0,   70,  50]), np.array([15,  255, 255])) |
                cv2.inRange(hsv, np.array([155, 70,  50]), np.array([180, 255, 255])))
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

        # Green block tracking overlay
        track_green_block(frame, display_frame, H_matrix)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Q pressed. Exiting...")
            return None, speed_slow

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

        if stability_counter >= STABILITY_LIMIT:
            print(f"[PHASE 2] Locked {len(current_list)} red blocks.")
            return current_list, speed_slow


# ── Phase 3: pick and place ───────────────────────────────────────────────────

def phase_execute_batch(api, pick_list, drop_list):
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api)
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

        if not is_valid_position(pick_x, pick_y, Z_PICK):
            print("  Skipping — invalid pick position.")
            continue
        if not is_valid_position(drop_x, drop_y, Z_SAFE):
            print("  Skipping — invalid drop position.")
            continue

        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        time.sleep(0.6)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
        time.sleep(0.6)
        dobotArm.close_gripper(api)
        time.sleep(0.6)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        time.sleep(0.6)
        dobotArm.move_to_xyz(api, drop_x, drop_y, 30)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        time.sleep(0.4)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    # More picks than drops — pile remaining onto first drop zone
    if len(pick_list) > len(drop_list):
        drop_x, drop_y = drop_list[0]
        for i in range(batch_size, len(pick_list)):
            pick_x, pick_y = pick_list[i]
            print(f"  Overflow {i + 1}: pick ({pick_x:.1f}, {pick_y:.1f})  →  drop ({drop_x:.1f}, {drop_y:.1f})")
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
            time.sleep(0.2)
            dobotArm.close_gripper(api)
            time.sleep(0.5)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            time.sleep(0.1)
            dobotArm.move_to_xyz(api, drop_x, drop_y, 25)
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("[PHASE 3] Batch complete.")
    return True


# ── Wait-for-input state ──────────────────────────────────────────────────────

def phase_wait_for_input(cap, api, H_matrix, map1, map2, detector):
    """Show live feed after a completed batch. SPACE re-scans targets, Q quits."""
    print("\n[WAIT] Batch done. Press SPACE to pick again, Q to quit.")
    speed_slow = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame         = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        track_green_block(frame, display_frame, H_matrix)

        hand_visible, speed_slow = check_hand_and_update_speed(
            api, display_frame, detector, speed_slow
        )
        _draw_hud(display_frame, "DONE — SPACE: pick again  |  Q: quit",
                  hand_visible, speed_slow)
        cv2.putText(display_frame, "SPACE: pick again", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Collab Pick & Place — Orbbec", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            print("\n[INFO] SPACE pressed — scanning for targets again...")
            _coord_histories.clear()
            return "scanning target"
        elif key == ord("q"):
            print("\n[INFO] Q pressed. Exiting...")
            return "exit"


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

    drop_zone    = None
    pick_target  = None
    machine_state = "scanning plate"

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECT_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    ) as detector:
        try:
            while machine_state != "exit":

                if machine_state == "scanning plate":
                    drop_zone, _ = phase_detect_plates(
                        cap, api, H_matrix, map1, map2, detector
                    )
                    if drop_zone is None:
                        break
                    if drop_zone:
                        machine_state = "scanning target"

                elif machine_state == "scanning target":
                    pick_target, _ = phase_detect_targets(
                        cap, api, H_matrix, map1, map2, detector
                    )
                    if pick_target is None:
                        break
                    if pick_target:
                        machine_state = "pick place"

                elif machine_state == "pick place":
                    if pick_target is None or drop_zone is None:
                        print("[WARN] Missing scan data — returning to plate scan.")
                        machine_state = "scanning plate"
                    else:
                        completed = phase_execute_batch(api, pick_target, drop_zone)
                        if completed:
                            machine_state = "wait_for_input"
                        else:
                            print("[WARN] Batch failed — re-scanning targets.")
                            machine_state = "scanning target"

                elif machine_state == "wait_for_input":
                    machine_state = phase_wait_for_input(
                        cap, api, H_matrix, map1, map2, detector
                    )

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
