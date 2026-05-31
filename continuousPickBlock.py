#This code is a modified implementation based on pickCVBlock.py.
#It continuously tracks a light green block while scanning for plates and red targets,
#and allows continuous picking by pressing the SPACE bar after a batch is completed.

import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
from collections import defaultdict, deque

"""CONSTANTS"""

Z_SAFE = 40
Z_PICK = -30
STABILITY_LIMIT = 60
PIXEL_TOLERANCE = 10

coord_histories = defaultdict(lambda: deque(maxlen=5))

def get_stable_target(idx, new_x, new_y):
    # Avoid averaging based on contour index, as contour sorting order fluctuates
    # causing targets to jump wildly between frames.
    return new_x, new_y

machine_state = "scanning plate" 

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
api = dType.load()
cap = cv2.VideoCapture(1)
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)
else:
    print("Failed to read from camera. Ensure it is connected.")
    exit()

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

# ---------------------------------------------------------
# TRACK GREEN BLOCK HELPER FUNCTION
# ---------------------------------------------------------
def track_green_block(frame, display_frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5,5), 0), cv2.COLOR_BGR2HSV)
    # Light green (연두색) HSV range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 150]
    
    for cnt in valid_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            rx, ry = pixel_to_robot(cx, cy, H_matrix)
            
            cv2.drawContours(display_frame, [cnt], -1, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Green: ({rx:.1f}, {ry:.1f})", (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ---------------------------------------------------------
# ENSURE VALID POSITIONING HELPER FUNCTION
# ---------------------------------------------------------
def is_valid_position(x, y, z):
    # Dobot Magician safe workspace (mm)
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


# ---------------------------------------------------------
# PHASE 1: DETECT Part Drop Zones (Plates)
# ---------------------------------------------------------
def phase_detect_plates():
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=15, maxRadius=65)

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        track_green_block(frame, display_frame)

        # --- AUTO-LOCK LOGIC ---
        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Q pressed. Exiting...")
            exit(0)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list

# ---------------------------------------------------------
# PHASE 2: DETECT Red Targets
# ---------------------------------------------------------
def phase_detect_targets():
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5,5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255])) | \
               cv2.inRange(hsv, np.array([155, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_list = []
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 150]
        
        for idx, cnt in enumerate(valid_contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                rx, ry = pixel_to_robot(cx, cy, H_matrix)
                
                smooth_x, smooth_y = get_stable_target(idx, rx, ry)
                current_list.append((smooth_x, smooth_y))
        
                cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
                    
        track_green_block(frame, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Q pressed. Exiting...")
            exit(0)

        # --- STABILITY LOGIC ---
        if len(current_list) != 0:
            if len(current_list) > 0 and len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Detection", display_frame)
        
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            return current_list

# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP
# ---------------------------------------------------------
def phase_execute_batch(api, pick_list, drop_list):
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api)        # clears queue between movement
    time.sleep(0.5)
    
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("missing targets, aborting")
        return False
    
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y = pick_list[i]
        drop_x, drop_y = drop_list[i]

        print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y}")

        if not is_valid_position(pick_x, pick_y, Z_PICK):
            print("Skipping invalid pick position")
            continue
        if not is_valid_position(drop_x, drop_y, Z_SAFE):
            print("Skipping invalid drop position")
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

    if len(pick_list) > len(drop_list):
         drop_x, drop_y = drop_list[0]
         for i in range(batch_size, len(pick_list)):
             pick_x, pick_y = pick_list[i]
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

    print("\nBatch Complete.")
    return True

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

# Pre-initialized to None so they are always defined, even if an exception
# occurs before the scanning phases complete (prevents NameError in finally
# and guards against the pick place state running with undefined variables)
drop_zone = None
pick_target = None

try:
    while machine_state != "exit":
        if cv2.waitKey(1) & 0xFF == ord('q'):  # outer safety exit between states
            break

        if machine_state == "scanning plate":
            drop_zone = phase_detect_plates()
            if drop_zone:                           # non-empty list check ([] is falsy)
                machine_state = "scanning target"

        elif machine_state == "scanning target":
            pick_target = phase_detect_targets()
            if pick_target:                         # non-empty list check ([] is falsy)
                machine_state = "pick place"

        elif machine_state == "pick place":
            # Guard against reaching this state before scans complete
            if pick_target is None or drop_zone is None:
                print("[WARN] Missing scan data, returning to plate scan...")
                machine_state = "scanning plate"
            else:
                completed = phase_execute_batch(api, pick_target, drop_zone)
                if completed:
                    machine_state = "wait_for_input"
                else:
                    print("[WARN] Batch failed, re-scanning targets...")
                    machine_state = "scanning target"

        elif machine_state == "wait_for_input":
            while True:
                ret, frame = cap.read()
                if not ret: continue

                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                display_frame = frame.copy()

                track_green_block(frame, display_frame)

                cv2.putText(display_frame, "Press SPACE to pick again, Q to quit", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Detection", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    print("\n[INFO] SPACE pressed. Scanning for targets again...")
                    coord_histories.clear()         # reset for a fresh scan
                    machine_state = "scanning target"
                    break
                elif key == ord('q'):
                    print("\n[INFO] Q pressed. Exiting...")
                    machine_state = "exit"
                    break                           # breaks inner loop; outer checks "exit"

finally:
    # Always runs: clean exit on q, break, Ctrl+C, or unhandled exception
    cap.release()
    cv2.destroyAllWindows()
    print("Cleaned up.")