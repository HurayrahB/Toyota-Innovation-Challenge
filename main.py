"""
hand_control.py — Gesture-controlled Dobot Magician arm via MediaPipe Hands.
(Now unified with Collaborative mode)
"""

import os
import sys
import queue
import threading
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "mediapipe is required. Activate the venv and run: pip install mediapipe"
    ) from exc

import dobotArm
import lib.DobotDllType as dType
from modules.collab_mode import CollabModeLogic

# ── Configuration ─────────────────────────────────────────────────────────────

COM_PORT           = "COM5"   # change to match DobotLab (e.g. "COM3")

CAMERA_CANDIDATES  = (1, 0, 2)
MIN_DETECT_CONF    = 0.70
MIN_TRACK_CONF     = 0.50

DEBOUNCE_FRAMES    = 18       # frames a gesture must hold before it fires
TRACK_DEADZONE_MM  = 8        # mm — minimum hand movement to trigger a TRACK move
Z_HOVER            = 50       # mm — Z height used during TRACK mode
Z_STEP             = 15       # mm — how much to raise/lower per MOVE_UP/DOWN command
MOVE_HOLD_INTERVAL = 0.7      # seconds — repeat rate when MOVE_UP/DN gesture is held

# Safe workspace bounds (mm in Dobot frame). Arm will not be commanded outside these.
WS_X = (150, 310)
WS_Y = (-140, 140)
WS_Z = (-25,  100)

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Gesture Recognition ───────────────────────────────────────────────────────

def _fingers_extended(lm, handedness):
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    if handedness == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x
    return thumb_up, index_up, middle_up, ring_up, pinky_up

def classify_gesture(hand_landmarks, handedness="Right"):
    lm = hand_landmarks.landmark
    thumb, idx, mid, ring, pinky = _fingers_extended(lm, handedness)
    if thumb and not idx and not mid and not ring and not pinky:
        return "HOME"
    n = sum([thumb, idx, mid, ring, pinky])
    if n == 5:                                         return "TRACK"
    if n == 0:                                         return "GRIP_TOGGLE"
    if idx and not mid and not ring and not pinky:     return "MOVE_UP"
    if idx and mid and not ring and not pinky:         return "MOVE_DN"
    if idx and mid and ring and not pinky:             return "OPEN_GRIP"
    return "NONE"


# ── Gesture Debouncer ─────────────────────────────────────────────────────────

class GestureDebouncer:
    def __init__(self, required=DEBOUNCE_FRAMES):
        self.required   = required
        self.current    = "NONE"
        self.count      = 0
        self._fired     = False

    def update(self, gesture):
        if gesture == self.current:
            self.count = min(self.count + 1, self.required)
        else:
            self.current = gesture
            self.count   = 1
            self._fired  = False
        return self.current if self.count >= self.required else None

    @property
    def confirmed(self):
        return self.current if self.count >= self.required else None

    def fire_once(self):
        if self.count >= self.required and not self._fired:
            self._fired = True
            return True
        return False

    def continuous(self):
        return self.count >= self.required


# ── Coordinate Mapping ────────────────────────────────────────────────────────

def map_hand_to_robot(hand_landmarks, frame_shape, H_matrix):
    h, w  = frame_shape[:2]
    wrist = hand_landmarks.landmark[0]
    px    = int(wrist.x * w)
    py    = int(wrist.y * h)
    p  = np.array([px, py, 1.0], dtype=np.float64)
    xy = H_matrix @ p
    xy /= xy[2]
    return float(xy[0]), float(xy[1])

def clamp(x, y, z):
    x = max(WS_X[0], min(WS_X[1], x))
    y = max(WS_Y[0], min(WS_Y[1], y))
    z = max(WS_Z[0], min(WS_Z[1], z))
    return x, y, z

def hand_in_workspace(rx, ry):
    return WS_X[0] < rx < WS_X[1] and WS_Y[0] < ry < WS_Y[1]


# ── Robot Worker Thread ───────────────────────────────────────────────────────

class RobotController:
    def __init__(self, api):
        self.api            = api
        self.busy           = threading.Event()
        self._q             = queue.Queue(maxsize=1)
        self.gripper_closed = False
        self._t             = threading.Thread(target=self._run, daemon=True)
        self._t.start()

        self.SPEED_NORMAL = 50
        self.SPEED_SLOW = 25
        self.speed_is_slow = False

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            action, args = item
            self.busy.set()
            try:
                if action == "xyz":
                    dobotArm.move_to_xyz(self.api, *args)
                elif action == "close":
                    dobotArm.close_gripper(self.api)
                    self.gripper_closed = True
                elif action == "open":
                    dobotArm.open_gripper(self.api)
                    dobotArm.stop_pump(self.api)
                    self.gripper_closed = False
                elif action == "home":
                    dobotArm.move_to_home(self.api)
                    dobotArm.open_gripper(self.api)
                    dobotArm.stop_pump(self.api)
                    self.gripper_closed = False
                elif action == "batch":
                    self._execute_batch(*args)
            except Exception as exc:
                print(f"[robot] {action} failed: {exc}")
            finally:
                self.busy.clear()

    def send(self, action, *args, drop_if_busy=False):
        if drop_if_busy and self.busy.is_set():
            return
        cmd = (action, args)
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(cmd)
        except queue.Full:
            pass

    def stop(self):
        self._q.put(None)

    def set_speed(self, slow=False):
        if slow and not self.speed_is_slow:
            dType.SetPTPCommonParams(self.api, self.SPEED_SLOW, self.SPEED_SLOW, isQueued=0)
            self.speed_is_slow = True
            print("Hand detected — speed reduced to 50%.")
        elif not slow and self.speed_is_slow:
            dType.SetPTPCommonParams(self.api, self.SPEED_NORMAL, self.SPEED_NORMAL, isQueued=0)
            self.speed_is_slow = False
            print("Hand cleared — speed restored to normal.")

    def _execute_batch(self, pick_list, drop_list, z_safe, z_pick):
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        time.sleep(0.5)
        batch_size = min(len(pick_list), len(drop_list))
        for i in range(batch_size):
            pick_x, pick_y = pick_list[i]
            drop_x, drop_y = drop_list[i]
            dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
            time.sleep(0.6)
            dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_pick)
            time.sleep(0.6)
            dobotArm.close_gripper(self.api)
            time.sleep(0.6)
            dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
            time.sleep(0.6)
            dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)
            dobotArm.open_gripper(self.api)
            dobotArm.stop_pump(self.api)
            time.sleep(0.4)
            dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)
        if len(pick_list) > len(drop_list):
            drop_x, drop_y = drop_list[0]
            for i in range(batch_size, len(pick_list)):
                pick_x, pick_y = pick_list[i]
                dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
                dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_pick)
                time.sleep(0.2)
                dobotArm.close_gripper(self.api)
                time.sleep(0.5)
                dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
                time.sleep(0.1)
                dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)
                dobotArm.open_gripper(self.api)
                dobotArm.stop_pump(self.api)
                dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera(candidates=CAMERA_CANDIDATES):
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue
        print(f"Camera opened at index {idx}.")
        return cap
    raise RuntimeError("No camera found.")


# ── HUD ───────────────────────────────────────────────────────────────────────

_LEGEND = [
    "Open palm   → TRACK XY",
    "Index only  → MOVE UP",
    "Peace sign  → MOVE DOWN",
    "Fist        → GRIP toggle",
    "3 fingers   → OPEN grip",
    "Thumb up    → HOME",
    "No gesture  → HOLD",
]

def draw_hud(frame, raw, confirmed, rx, ry, z, gripper_closed, busy, safety):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 115), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    GREEN  = (0, 220, 0)
    AMBER  = (0, 170, 255)
    RED    = (0, 0, 220)
    GREY   = (180, 180, 180)

    cv2.putText(frame, f"Raw : {raw}",       (14, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREY,  1)
    cv2.putText(frame, f"Act : {confirmed or '---'}",
                (14, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                GREEN if confirmed and confirmed != "NONE" else AMBER, 2)
    cv2.putText(frame, f"XY ({rx:+.0f}, {ry:+.0f}) mm   Z {z:.0f} mm",
                (14, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREY, 1)

    if safety:
        status, col = "SAFETY HOLD", RED
    elif busy:
        status, col = "ROBOT BUSY", AMBER
    elif gripper_closed:
        status, col = "GRIPPER CLOSED", AMBER
    else:
        status, col = "READY", GREEN

    cv2.putText(frame, status, (w - 310, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    for i, line in enumerate(_LEGEND):
        cv2.putText(frame, line, (w - 268, h - len(_LEGEND) * 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    if not Path("HomographyMatrix.npy").exists():
        raise RuntimeError("HomographyMatrix.npy not found.")
    H_matrix = np.load("HomographyMatrix.npy")

    api = dType.load()
    print("Connecting to Dobot on", COM_PORT, "…")
    dobotArm.initialize_robot(api, COM_PORT)
    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)
    print("Robot ready.\n")

    robot  = RobotController(api)
    dbnc   = GestureDebouncer(required=DEBOUNCE_FRAMES)
    
    print("Opening Laptop Camera...")
    cap = open_camera()
    
    print("Opening Ceiling Camera...")
    cap1 = cv2.VideoCapture(1)
    
    if Path("camera_params.npz").exists() and cap1.isOpened():
        data = np.load("camera_params.npz")
        camera_matrix = data["camera_matrix"]
        dist_coeffs   = data["dist_coeffs"]
        ret1, frame1 = cap1.read()
        if ret1:
            h_c, w_c = frame1.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w_c,h_c), 1)
            map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w_c,h_c), cv2.CV_16SC2)
        else:
            map1, map2 = None, None
    else:
        map1, map2 = None, None

    collab_logic = CollabModeLogic(map1, map2, H_matrix)
    mode = "CONTROL"

    arm_x, arm_y = float(dobotArm.home_pos[0]), float(dobotArm.home_pos[1])
    arm_z = float(Z_HOVER)
    last_move_time = 0.0

    print("Press 'm' to manually toggle modes. Press 'q' to quit.")

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=MIN_DETECT_CONF, min_tracking_confidence=MIN_TRACK_CONF
    ) as detector:

        while True:
            ret, frame = cap.read()
            ret1, frame1 = cap1.read()
            
            # CEILING CAMERA HAND DETECTION
            hand_in_ceiling = False
            display_frame1 = None
            if ret1:
                display_frame1 = frame1.copy()
                frame1_rgb = cv2.cvtColor(cv2.flip(frame1, 1), cv2.COLOR_BGR2RGB)
                frame1_rgb.flags.writeable = False
                results1 = detector.process(frame1_rgb)
                if results1.multi_hand_landmarks:
                    hand_in_ceiling = True
                    for hand_lm in results1.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame1, hand_lm, mp_hands.HAND_CONNECTIONS)

            if mode == "CONTROL":
                if hand_in_ceiling:
                    print("\n[ALERT] Hand detected on ceiling camera! Switching to COLLAB mode.")
                    mode = "COLLAB"
                    continue

                if not ret: continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = detector.process(rgb)
                frame.flags.writeable = True

                raw, confirmed = "NONE", None
                rx, ry, in_safety = arm_x, arm_y, False

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    side = results.multi_handedness[0].classification[0].label if results.multi_handedness else "Right"
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())
                    rx, ry = map_hand_to_robot(hand, frame.shape, H_matrix)
                    raw    = classify_gesture(hand, side)

                    if hand_in_workspace(rx, ry):
                        raw, in_safety = "NONE", True

                    confirmed = dbnc.update(raw)
                    now = time.time()

                    if confirmed == "TRACK" and dbnc.continuous():
                        if not robot.busy.is_set():
                            tx, ty = clamp(rx, ry, arm_z)[:2]
                            if abs(tx - arm_x) > TRACK_DEADZONE_MM or abs(ty - arm_y) > TRACK_DEADZONE_MM:
                                robot.send("xyz", tx, ty, arm_z, drop_if_busy=False)
                                arm_x, arm_y = tx, ty
                    elif confirmed == "MOVE_UP":
                        if dbnc.fire_once() or (dbnc.continuous() and now - last_move_time > MOVE_HOLD_INTERVAL and not robot.busy.is_set()):
                            new_z = clamp(arm_x, arm_y, arm_z + Z_STEP)[2]
                            robot.send("xyz", arm_x, arm_y, new_z)
                            arm_z, last_move_time = new_z, now
                    elif confirmed == "MOVE_DN":
                        if dbnc.fire_once() or (dbnc.continuous() and now - last_move_time > MOVE_HOLD_INTERVAL and not robot.busy.is_set()):
                            new_z = clamp(arm_x, arm_y, arm_z - Z_STEP)[2]
                            robot.send("xyz", arm_x, arm_y, new_z)
                            arm_z, last_move_time = new_z, now
                    elif confirmed == "GRIP_TOGGLE" and dbnc.fire_once():
                        robot.send("open") if robot.gripper_closed else robot.send("close")
                    elif confirmed == "OPEN_GRIP" and dbnc.fire_once():
                        robot.send("open")
                    elif confirmed == "HOME" and dbnc.fire_once():
                        arm_x, arm_y, arm_z = float(dobotArm.home_pos[0]), float(dobotArm.home_pos[1]), float(Z_HOVER)
                        robot.send("home")
                else:
                    dbnc.update("NONE")

                draw_hud(frame, raw, confirmed, rx, ry, arm_z, robot.gripper_closed, robot.busy.is_set(), in_safety)
                cv2.putText(frame, "MODE: CONTROL", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow("Laptop Camera", frame)
                if ret1:
                    cv2.imshow("Ceiling Camera", display_frame1)

            elif mode == "COLLAB":
                robot.set_speed(slow=hand_in_ceiling)
                if ret1:
                    collab_logic.update_collab_state(frame1, display_frame1, robot)
                    if hand_in_ceiling:
                        cv2.putText(display_frame1, "HAND DETECTED! SPEED 50%", (10, display_frame1.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(display_frame1, "MODE: COLLAB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.imshow("Ceiling Camera", display_frame1)
                if ret:
                    cv2.imshow("Laptop Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode = "CONTROL" if mode == "COLLAB" else "COLLAB"
                print(f"\n[INFO] Manually switched to {mode} mode.")
                if mode == "CONTROL":
                    robot.set_speed(slow=False)

    print("Shutting down…")
    robot.stop()
    cap.release()
    if cap1.isOpened(): cap1.release()
    cv2.destroyAllWindows()
    dType.DisconnectDobot(api)

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(f"\nError: {err}", file=sys.stderr)
        raise SystemExit(1) from err