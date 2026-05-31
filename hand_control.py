"""
hand_control.py — Gesture-controlled Dobot Magician arm via MediaPipe Hands.

Gesture map
-----------
Open palm   (5 fingers) → TRACK      : arm follows your hand's XY position at hover height
Index only  (1 finger)  → MOVE UP    : raise end-effector by Z_STEP mm
Peace sign  (2 fingers) → MOVE DOWN  : lower end-effector by Z_STEP mm
Fist        (0 fingers) → GRIP       : toggle gripper closed / open
3 fingers               → OPEN GRIP  : open gripper (safe override)
Thumb up                → HOME       : return to home position, open gripper
Pinky only              → MOVE LEFT  : move arm in -Y direction by Y_STEP mm
Ring only               → MOVE RIGHT : move arm in +Y direction by Y_STEP mm
Middle only             → BOMB       : plunge arm to lowest Z position (WS_Z[0])
No clear gesture        → HOLD       : arm stays at last position

Safety: if the wrist landmark maps into the robot's physical workspace on the
table, any gesture is overridden with HOLD so the arm never moves toward a hand
that is already near it.

Setup: edit COM_PORT below to match the port shown in DobotLab (e.g. "COM3").
       Camera calibration files HomographyMatrix.npy and camera_params.npz must
       be present in the same directory (run getTransformationMatrix.py first).
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

# ── Configuration ─────────────────────────────────────────────────────────────

COM_PORT           = "COM7"   # change to match DobotLab (e.g. "COM3")

CAMERA_INDEX       = 0        # laptop built-in camera
MIN_DETECT_CONF    = 0.70
MIN_TRACK_CONF     = 0.50

DEBOUNCE_FRAMES    = 18       # frames a gesture must hold before it fires
TRACK_DEADZONE_MM  = 8        # mm — minimum hand movement to trigger a TRACK move
Z_HOVER            = 50       # mm — Z height used during TRACK mode
Z_STEP             = 15       # mm — how much to raise/lower per MOVE_UP/DOWN command
Y_STEP             = 15       # mm — how much to move left/right per command
MOVE_HOLD_INTERVAL = 0.7      # seconds — repeat rate when directional gesture is held
THUMB_THRESHOLD    = 0.04     # normalised units — how far thumb must stick out

# Safe workspace bounds (mm in Dobot frame). Arm will not be commanded outside these.
WS_X = (150, 310)
WS_Y = (-140, 140)
WS_Z = (-25,  100)


mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Gesture Recognition ───────────────────────────────────────────────────────

def _fingers_extended(lm, handedness):
    """Return (thumb_up, index_up, middle_up, ring_up, pinky_up)."""
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    # Thumb: requires clearly extended past knuckle (less sensitive)
    if handedness == "Right":
        thumb_up = (lm[3].x - lm[4].x) > THUMB_THRESHOLD
    else:
        thumb_up = (lm[4].x - lm[3].x) > THUMB_THRESHOLD
    return thumb_up, index_up, middle_up, ring_up, pinky_up


def classify_gesture(hand_landmarks, handedness="Right"):
    lm = hand_landmarks.landmark
    thumb, idx, mid, ring, pinky = _fingers_extended(lm, handedness)

    # Thumb-up: thumb out, all fingers curled
    if thumb and not idx and not mid and not ring and not pinky:
        return "HOME"

    # Middle finger only → BOMB (plunge to lowest Z)
    if mid and not idx and not ring and not pinky and not thumb:
        return "BOMB"

    n = sum([thumb, idx, mid, ring, pinky])

    if n == 5:                                                      return "TRACK"
    if n == 0:                                                      return "GRIP_TOGGLE"
    if idx and not mid and not ring and not pinky:                  return "MOVE_UP"
    if idx and mid and not ring and not pinky:                      return "MOVE_DN"
    if idx and mid and ring and not pinky:                          return "OPEN_GRIP"
    if pinky and not idx and not mid and not ring and not thumb:    return "MOVE_LEFT"
    if ring and not idx and not mid and not pinky and not thumb:    return "MOVE_RIGHT"

    return "NONE"


# ── Gesture Debouncer ─────────────────────────────────────────────────────────

class GestureDebouncer:
    """
    Confirms a gesture only after it holds for `required` consecutive frames.
    Tracks whether one-shot gestures have already fired so they don't repeat
    while the hand stays still.
    """

    def __init__(self, required=DEBOUNCE_FRAMES):
        self.required = required
        self.current  = "NONE"
        self.count    = 0
        self._fired   = False

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
        """Returns True exactly once per hold — use for discrete commands."""
        if self.count >= self.required and not self._fired:
            self._fired = True
            return True
        return False

    def continuous(self):
        """Returns True every frame while gesture is confirmed — use for TRACK."""
        return self.count >= self.required


# ── Coordinate Mapping ────────────────────────────────────────────────────────

def map_hand_to_robot(hand_landmarks, frame_shape, H_matrix):
    """Project the wrist landmark through the homography into robot mm coords."""
    h, w  = frame_shape[:2]
    wrist = hand_landmarks.landmark[0]
    px    = int(wrist.x * w)
    py    = int(wrist.y * h)
    p     = np.array([px, py, 1.0], dtype=np.float64)
    xy    = H_matrix @ p
    xy   /= xy[2]
    return float(xy[0]), float(xy[1])


def clamp(x, y, z):
    x = max(WS_X[0], min(WS_X[1], x))
    y = max(WS_Y[0], min(WS_Y[1], y))
    z = max(WS_Z[0], min(WS_Z[1], z))
    return x, y, z


def hand_in_workspace(rx, ry):
    """True when the hand's projected position is inside the robot's table zone."""
    return WS_X[0] < rx < WS_X[1] and WS_Y[0] < ry < WS_Y[1]


# ── Robot Worker Thread ───────────────────────────────────────────────────────

class RobotController:
    """
    Runs Dobot commands in a background thread so the camera loop never blocks.
    A maxsize-1 queue means only one pending command waits at a time — stale
    TRACK positions are discarded when a newer one arrives.
    """

    def __init__(self, api):
        self.api            = api
        self.busy           = threading.Event()
        self._q             = queue.Queue(maxsize=1)
        self.gripper_closed = False
        self._t             = threading.Thread(target=self._run, daemon=True)
        self._t.start()

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


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open laptop camera at index {CAMERA_INDEX}. "
            "Check macOS > System Settings > Privacy & Security > Camera."
        )
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Camera {CAMERA_INDEX} opened but returned no frames.")
    print(f"Laptop camera opened at index {CAMERA_INDEX}.")
    return cap


# ── HUD ───────────────────────────────────────────────────────────────────────

_LEGEND = [
    "Open palm   -> TRACK XY",
    "Index only  -> MOVE UP",
    "Peace sign  -> MOVE DOWN",
    "Fist        -> GRIP toggle",
    "3 fingers   -> OPEN grip",
    "Thumb up    -> HOME",
    "Pinky only  -> MOVE LEFT",
    "Ring only   -> MOVE RIGHT",
    "Middle only -> BOMB (lowest Z)",
    "No gesture  -> HOLD",
]

GESTURE_COLORS = {
    "TRACK":       (0, 220, 0),
    "MOVE_UP":     (0, 200, 255),
    "MOVE_DN":     (0, 140, 255),
    "GRIP_TOGGLE": (255, 180, 0),
    "OPEN_GRIP":   (255, 220, 0),
    "HOME":        (200, 0, 255),
    "MOVE_LEFT":   (0, 255, 200),
    "MOVE_RIGHT":  (255, 100, 200),
    "BOMB":        (0, 0, 255),
}


def draw_hud(frame, raw, confirmed, rx, ry, z, gripper_closed, busy, safety):
    h, w = frame.shape[:2]

    GREEN = (0, 220, 0)
    AMBER = (0, 170, 255)
    RED   = (0, 0, 220)
    GREY  = (180, 180, 180)

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 115), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Raw : {raw}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREY, 1)

    act_color = GESTURE_COLORS.get(confirmed, AMBER) if confirmed and confirmed != "NONE" else AMBER
    cv2.putText(frame, f"Act : {confirmed or '---'}", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, act_color, 2)

    cv2.putText(frame, f"XY ({rx:+.0f}, {ry:+.0f}) mm   Z {z:.0f} mm",
                (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREY, 1)

    if safety:
        status, col = "SAFETY HOLD — hand in workspace", RED
    elif busy:
        status, col = "ROBOT BUSY", AMBER
    elif gripper_closed:
        status, col = "GRIPPER CLOSED", AMBER
    else:
        status, col = "READY", GREEN

    cv2.putText(frame, status, (w - 330, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    # Legend bottom-right
    for i, line in enumerate(_LEGEND):
        cv2.putText(frame, line, (w - 285, h - len(_LEGEND) * 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    # Load homography (pixel → robot mm)
    if not Path("HomographyMatrix.npy").exists():
        raise RuntimeError(
            "HomographyMatrix.npy not found. Run getTransformationMatrix.py first."
        )
    H_matrix = np.load("HomographyMatrix.npy")

    # Robot setup
    api = dType.load()
    print("Connecting to Dobot on", COM_PORT, "…")
    dobotArm.initialize_robot(api)
    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)
    print("Robot ready. Press Q in the camera window to quit.\n")

    robot  = RobotController(api)
    dbnc   = GestureDebouncer(required=DEBOUNCE_FRAMES)
    cap    = open_camera()

    arm_x, arm_y = float(dobotArm.home_pos[0]), float(dobotArm.home_pos[1])
    arm_z = float(Z_HOVER)

    last_move_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECT_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    ) as detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = detector.process(rgb)
            frame.flags.writeable = True

            raw       = "NONE"
            confirmed = None
            rx, ry    = arm_x, arm_y
            in_safety = False

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                side = "Right"
                if results.multi_handedness:
                    side = results.multi_handedness[0].classification[0].label

                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                rx, ry = map_hand_to_robot(hand, frame.shape, H_matrix)
                raw    = classify_gesture(hand, side)

                if hand_in_workspace(rx, ry):
                    raw       = "NONE"
                    in_safety = True

                confirmed = dbnc.update(raw)
                now = time.time()

                # ── TRACK: continuous XY follow ───────────────────────────────
                if confirmed == "TRACK" and dbnc.continuous():
                    if not robot.busy.is_set():
                        tx, ty = clamp(rx, ry, arm_z)[:2]
                        if (abs(tx - arm_x) > TRACK_DEADZONE_MM or
                                abs(ty - arm_y) > TRACK_DEADZONE_MM):
                            robot.send("xyz", tx, ty, arm_z)
                            arm_x, arm_y = tx, ty

                # ── MOVE UP ───────────────────────────────────────────────────
                elif confirmed == "MOVE_UP":
                    if dbnc.fire_once() or (dbnc.continuous() and
                                            now - last_move_time > MOVE_HOLD_INTERVAL
                                            and not robot.busy.is_set()):
                        new_z = clamp(arm_x, arm_y, arm_z + Z_STEP)[2]
                        robot.send("xyz", arm_x, arm_y, new_z)
                        arm_z = new_z
                        last_move_time = now

                # ── MOVE DOWN ─────────────────────────────────────────────────
                elif confirmed == "MOVE_DN":
                    if dbnc.fire_once() or (dbnc.continuous() and
                                            now - last_move_time > MOVE_HOLD_INTERVAL
                                            and not robot.busy.is_set()):
                        new_z = clamp(arm_x, arm_y, arm_z - Z_STEP)[2]
                        robot.send("xyz", arm_x, arm_y, new_z)
                        arm_z = new_z
                        last_move_time = now

                # ── MOVE LEFT (-Y) ────────────────────────────────────────────
                elif confirmed == "MOVE_LEFT":
                    if dbnc.fire_once() or (dbnc.continuous() and
                                            now - last_move_time > MOVE_HOLD_INTERVAL
                                            and not robot.busy.is_set()):
                        new_y = clamp(arm_x, arm_y - Y_STEP, arm_z)[1]
                        robot.send("xyz", arm_x, new_y, arm_z)
                        arm_y = new_y
                        last_move_time = now

                # ── MOVE RIGHT (+Y) ───────────────────────────────────────────
                elif confirmed == "MOVE_RIGHT":
                    if dbnc.fire_once() or (dbnc.continuous() and
                                            now - last_move_time > MOVE_HOLD_INTERVAL
                                            and not robot.busy.is_set()):
                        new_y = clamp(arm_x, arm_y + Y_STEP, arm_z)[1]
                        robot.send("xyz", arm_x, new_y, arm_z)
                        arm_y = new_y
                        last_move_time = now

                # ── GRIP TOGGLE ───────────────────────────────────────────────
                elif confirmed == "GRIP_TOGGLE" and dbnc.fire_once():
                    if robot.gripper_closed:
                        robot.send("open")
                    else:
                        robot.send("close")

                # ── OPEN GRIP ─────────────────────────────────────────────────
                elif confirmed == "OPEN_GRIP" and dbnc.fire_once():
                    robot.send("open")

                # ── HOME ──────────────────────────────────────────────────────
                elif confirmed == "HOME" and dbnc.fire_once():
                    arm_x = float(dobotArm.home_pos[0])
                    arm_y = float(dobotArm.home_pos[1])
                    arm_z = float(Z_HOVER)
                    robot.send("home")

                # ── BOMB: plunge to lowest Z ──────────────────────────────────
                elif confirmed == "BOMB" and dbnc.fire_once():
                    arm_z = float(WS_Z[0])
                    robot.send("xyz", arm_x, arm_y, arm_z)

            else:
                dbnc.update("NONE")

            draw_hud(frame, raw, confirmed, rx, ry, arm_z,
                     robot.gripper_closed, robot.busy.is_set(), in_safety)
            cv2.imshow("Hand Control — Dobot", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print("Shutting down…")
    robot.stop()
    cap.release()
    cv2.destroyAllWindows()
    dType.DisconnectDobot(api)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(f"\nError: {err}", file=sys.stderr)
        print("\nRun with the project venv:", file=sys.stderr)
        print("  source .venv/bin/activate && python hand_control.py", file=sys.stderr)
        raise SystemExit(1) from err