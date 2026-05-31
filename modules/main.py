import cv2
import numpy as np
import sys
import os
from pathlib import Path
import mediapipe as mp

# Add parent directory to sys.path so we can import dobotArm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import dobotArm
import lib.DobotDllType as dType

from robot_worker import RobotController
from control_mode import ControlModeLogic
from collab_mode import CollabModeLogic

def main():
    # Load transformation and calibration for Ceiling Camera (cap1)
    homography_path = os.path.join(parent_dir, "HomographyMatrix.npy")
    if not Path(homography_path).exists():
        print(f"{homography_path} not found!")
        sys.exit(1)
    H_matrix = np.load(homography_path)

    camera_params_path = os.path.join(parent_dir, "camera_params.npz")
    if not Path(camera_params_path).exists():
        print(f"{camera_params_path} not found!")
        sys.exit(1)
    data = np.load(camera_params_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]

    print("Opening Laptop Camera (0)...")
    cap0 = cv2.VideoCapture(0)
    if not cap0.isOpened():
        print("Warning: Could not open Laptop Camera 0")

    print("Opening Ceiling Camera (1)...")
    cap1 = cv2.VideoCapture(1)
    if not cap1.isOpened():
        print("Warning: Could not open Ceiling Camera 1")

    # Compute undistort maps for ceiling camera
    ret1, frame1 = cap1.read()
    if ret1:
        h, w = frame1.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)
    else:
        print("Error reading from Ceiling Camera 1. Cannot compute undistort maps.")
        sys.exit(1)

    # Initialize Dobot
    api = dType.load()
    COM_PORT = "COM5" # Change if needed
    print(f"Connecting to Dobot on {COM_PORT} ...")
    dobotArm.initialize_robot(api, COM_PORT)
    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)

    robot = RobotController(api)
    control_logic = ControlModeLogic()
    collab_logic = CollabModeLogic(map1, map2, H_matrix)

    mode = "CONTROL" # Start mode
    print("\n--- System Ready ---")
    print("Modes:")
    print("  'CONTROL': Laptop camera gestures control the robot.")
    print("  'COLLAB': Ceiling camera automates pick & place.")
    print("Press 'm' to manually toggle modes.")
    print("Press 'q' to quit.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                        min_detection_confidence=0.6, min_tracking_confidence=0.5) as detector:

        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            display_frame1 = frame1.copy() if ret1 else None
            display_frame0 = frame0.copy() if ret0 else None

            # --- CEILING CAMERA HAND DETECTION ---
            hand_in_ceiling = False
            if ret1:
                frame1_rgb = cv2.cvtColor(cv2.flip(frame1, 1), cv2.COLOR_BGR2RGB)
                results1 = detector.process(frame1_rgb)
                if results1.multi_hand_landmarks:
                    hand_in_ceiling = True
                    for hand_lm in results1.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame1, hand_lm, mp_hands.HAND_CONNECTIONS)

            # --- MODE LOGIC ---
            if mode == "CONTROL":
                # Automatically shift to COLLAB mode if human hand detected on ceiling camera
                if hand_in_ceiling:
                    print("\n[ALERT] Hand detected on ceiling camera! Switching to COLLAB mode.")
                    mode = "COLLAB"
                    continue # Skip the rest of the frame processing for this tick
                
                # Process Control Mode
                if ret0:
                    frame0_flip = cv2.flip(frame0, 1)
                    frame0_rgb = cv2.cvtColor(frame0_flip, cv2.COLOR_BGR2RGB)
                    results0 = detector.process(frame0_rgb)

                    if results0.multi_hand_landmarks:
                        hand_lm = results0.multi_hand_landmarks[0]
                        handedness = results0.multi_handedness[0].classification[0].label
                        
                        mp_drawing.draw_landmarks(display_frame0, hand_lm, mp_hands.HAND_CONNECTIONS)

                        # Project wrist to robot coords (assuming same H_matrix logic)
                        wrist = hand_lm.landmark[0]
                        h, w = frame0.shape[:2]
                        px, py = int(wrist.x * w), int(wrist.y * h)
                        p = np.array([px, py, 1.0], dtype=np.float64)
                        xy = H_matrix @ p
                        xy /= xy[2]
                        rx, ry = float(xy[0]), float(xy[1])

                        raw, conf, arm_x, arm_y, arm_z, safe = control_logic.update_control_state(hand_lm, handedness, rx, ry, robot)
                        
                        # HUD
                        cv2.putText(display_frame0, f"RAW: {raw} | CONF: {conf}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame0, f"Pos: {arm_x:.0f}, {arm_y:.0f}, {arm_z:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if safe:
                            cv2.putText(display_frame0, "SAFETY STOP", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if display_frame0 is not None:
                    cv2.putText(display_frame0, "MODE: CONTROL", (10, display_frame0.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.imshow("Laptop Camera", display_frame0)
                if display_frame1 is not None:
                    cv2.imshow("Ceiling Camera", display_frame1)

            elif mode == "COLLAB":
                # Process Collab Mode
                robot.set_speed(slow=hand_in_ceiling)
                
                if ret1:
                    collab_logic.update_collab_state(frame1, display_frame1, robot)
                    
                    if hand_in_ceiling:
                        cv2.putText(display_frame1, "HAND DETECTED! SPEED 50%", (10, display_frame1.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    cv2.putText(display_frame1, "MODE: COLLAB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.imshow("Ceiling Camera", display_frame1)
                    
                if display_frame0 is not None:
                    cv2.imshow("Laptop Camera", display_frame0)

            # --- KEYBOARD INPUT ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mode = "CONTROL" if mode == "COLLAB" else "COLLAB"
                print(f"\n[INFO] Manually switched to {mode} mode.")
                if mode == "CONTROL":
                    robot.set_speed(slow=False) # Restore speed

    # Cleanup
    robot.stop()
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    dType.DisconnectDobot(api)
    print("Shutdown complete.")

if __name__ == "__main__":
    main()
