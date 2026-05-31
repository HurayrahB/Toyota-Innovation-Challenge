"""
hand_collab.py — Collaborative safety mode for the Dobot Magician arm.

A dedicated camera watches the workspace. Whenever a human hand enters the
camera frame, the arm's speed is immediately reduced to 50% of its normal
operating speed. When the hand leaves the frame, full speed is restored.

This script runs independently of hand_control.py. Both can be connected to
the same Dobot at the same time via separate processes, but only one should
send motion commands. This script only ever changes the PTP speed parameter —
it never moves the arm.

Setup
-----
- Set COM_PORT to match your Dobot's serial port.
- Connect the Orbbec Astra camera (separate from the laptop gesture camera).
  The Astra registers as a standard UVC webcam. Change CAMERA_INDEX if the
  OS assigns it a different index (check with cv2.VideoCapture(n)).
- Run getTransformationMatrix.py first if HomographyMatrix.npy is missing
  (it is not required here, but must exist for hand_control.py).

Usage
-----
    source .venv/bin/activate
    python hand_collab.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))

import cv2

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "mediapipe is required. Activate the venv: pip install mediapipe"
    ) from exc

import dobotArm
import lib.DobotDllType as dType

# ── Configuration ─────────────────────────────────────────────────────────────

COM_PORT      = "COM5"   # change to match DobotLab
CAMERA_INDEX  = 1        # Orbbec Astra UVC index (change if OS assigns a different index)

MIN_DETECT_CONF = 0.60
MIN_TRACK_CONF  = 0.50

SPEED_NORMAL  = 50       # % of max — matches dobotArm.initialize_robot default
SPEED_SLOW    = 25       # 50% of SPEED_NORMAL — applied while hand is visible

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open Orbbec Astra at camera index {index}. "
            "Ensure the Astra is plugged in and macOS has granted camera access. "
            "Update CAMERA_INDEX if the OS assigned it a different number."
        )
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Orbbec Astra (index {index}) opened but returned no frames.")
    print(f"Orbbec Astra opened at index {index}.")
    return cap


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(frame, hand_visible, speed_slow):
    h, w = frame.shape[:2]

    GREEN = (0, 220, 0)
    RED   = (0, 0, 220)
    AMBER = (0, 170, 255)
    GREY  = (180, 180, 180)
    CYAN  = (220, 220, 0)

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 90), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "MODE: COLLABORATIVE", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, CYAN, 2)

    if hand_visible:
        hand_text, hand_col = "HAND DETECTED", RED
    else:
        hand_text, hand_col = "NO HAND", GREEN
    cv2.putText(frame, hand_text, (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, hand_col, 2)

    if speed_slow:
        spd_text, spd_col = "SPEED: 50% (SLOW)", AMBER
    else:
        spd_text, spd_col = "SPEED: NORMAL", GREEN
    cv2.putText(frame, spd_text, (w - 290, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, spd_col, 2)

    cv2.putText(frame, "Press Q to quit", (w - 200, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Connect to robot — minimal init, no homing, so the arm is not interrupted
    # if hand_control.py is already running
    api = dType.load()
    print(f"Connecting to Dobot on {COM_PORT} …")
    state = dType.ConnectDobot(api, COM_PORT, 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        raise RuntimeError(
            f"Failed to connect to Dobot on {COM_PORT}. "
            "Check the port in DobotLab and update COM_PORT."
        )
    print("Robot ready. Collaborative mode active.\n")

    cap          = open_camera(CAMERA_INDEX)
    speed_slow   = False   # current speed state

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
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

            hand_visible = bool(results.multi_hand_landmarks)

            # Draw hand landmarks when detected
            if hand_visible:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Adjust arm speed based on hand presence
            if hand_visible and not speed_slow:
                dType.SetPTPCommonParams(api, SPEED_SLOW, SPEED_SLOW, isQueued=0)
                speed_slow = True
                print("Hand detected — speed reduced to 50%.")
            elif not hand_visible and speed_slow:
                dType.SetPTPCommonParams(api, SPEED_NORMAL, SPEED_NORMAL, isQueued=0)
                speed_slow = False
                print("Hand cleared — speed restored to normal.")

            draw_hud(frame, hand_visible, speed_slow)
            cv2.imshow("Collaborative Mode — Dobot Safety", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Restore normal speed before exiting
    dType.SetPTPCommonParams(api, SPEED_NORMAL, SPEED_NORMAL, isQueued=0)
    cap.release()
    cv2.destroyAllWindows()
    dType.DisconnectDobot(api)
    print("Collaborative mode stopped.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(f"\nError: {err}", file=sys.stderr)
        print("\nRun with the project venv:", file=sys.stderr)
        print("  source .venv/bin/activate && python hand_collab.py", file=sys.stderr)
        raise SystemExit(1) from err
