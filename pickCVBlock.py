#This code is a simplified implementation of a collaborative robotics system that detects plates and targets using computer vision, 
#and then commands a Dobot robotic arm to pick and place objects accordingly. The system operates in four phases: scanning for plates,
#scanning for targets, scanning for hands, and executing the pick/place operations.
#Stability checks are implemented to ensure reliable detection before proceeding to the next phase.

# Note: there are parameters that are useful to the successful operation of the robot arm. Read through the code before running the program.

# How to use: 
# 1. Ensure you have the Dobot robotic arm set up and connected to your computer.
# 2. Place the plates (drop zones) and targets (red blocks) within the camera's
# field of view.
# 3. Run the script. The system will scan for plates, then targets, then check for hands, and finally execute the pick/place operations based on the detected positions.
# 4. Monitor the console output and the video feed for feedback on the system's status and operations

#Other Useful Codes you can use:
#dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead): moves the robot to the specified (x, y, z) coordinates with a specified rotation for the end effector (rHead). Z_SAFE is a predefined constant that ensures the robot maintains a safe height to avoid collisions when moving horizontally.



import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time

try:
    import mediapipe as mp
except ImportError:
    mp = None


"""CONSTANTS"""

Z_SAFE = 40 #what is the clearance distance for the robot arm to avoid collisions when moving horizontally?
Z_PICK = -25 #what is the  height for the robot claw to successfully pick up the target?
Z_APPROACH = -8 #staging height above the target before the final pick descent
Z_PLACE = 0 #slower release height above the plate for a more consistent drop-off
STABILITY_LIMIT = 60  #how many consecutive frames of stable detection before we "lock in" the positions and move to the next phase? (at 30fps, 60 frames is about 2 seconds)
PIXEL_TOLERANCE = 10  #object can move at most this # of pixels to be considered stationary
TARGET_MIN_AREA = 250 #smaller velcro tags need a lower contour threshold than the starter code used
RETRY_LIMIT = 2
SHOW_MASK_WINDOW = True
HAND_CLEAR_LIMIT = 45
MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE = 0.5

LOWER_RED_1 = np.array([0, 80, 50])
UPPER_RED_1 = np.array([15, 255, 255])
LOWER_RED_2 = np.array([165, 80, 50])
UPPER_RED_2 = np.array([180, 255, 255])

machine_state = "scanning plate" 

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
# MAKE SURE THAT YOU HAVE RAN calibrateCamera.py FIRST TO GENERATE THE camera_params.npz FILE
api = dType.load()
cap = cv2.VideoCapture(0)
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

if mp is not None:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )
else:
    mp_hands = None
    mp_drawing = None
    hand_detector = None

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]


def sort_points(points):
    return sorted(points, key=lambda point: (point[0], point[1]))


def points_are_stable(current_points, last_points, tolerance):
    if len(current_points) == 0 or len(current_points) != len(last_points):
        return False

    sorted_current = sort_points(current_points)
    sorted_last = sort_points(last_points)

    for current_point, last_point in zip(sorted_current, sorted_last):
        if np.hypot(current_point[0] - last_point[0], current_point[1] - last_point[1]) > tolerance:
            return False

    return True


def build_red_mask(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask += cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def detect_targets(frame):
    mask = build_red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TARGET_MIN_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        rx, ry = pixel_to_robot(cx, cy, H_matrix)
        detections.append({
            "pixel": (cx, cy),
            "robot": (rx, ry),
            "area": area,
            "contour": cnt,
        })

    detections.sort(key=lambda detection: (detection["robot"][0], detection["robot"][1]))
    return detections, mask


def detect_hands(frame):
    if hand_detector is None or mp_hands is None:
        raise RuntimeError(
            "MediaPipe is not installed. Install it with 'pip install mediapipe' before running hand detection."
        )

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_frame)

    hand_regions = []
    if not results.multi_hand_landmarks:
        return hand_regions

    frame_height, frame_width = frame.shape[:2]
    handedness_list = results.multi_handedness or []

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]

        x_min = max(0, int(min(x_coords) * frame_width))
        y_min = max(0, int(min(y_coords) * frame_height))
        x_max = min(frame_width, int(max(x_coords) * frame_width))
        y_max = min(frame_height, int(max(y_coords) * frame_height))

        label = "Hand"
        score = 0.0
        if index < len(handedness_list) and handedness_list[index].classification:
            label = handedness_list[index].classification[0].label
            score = handedness_list[index].classification[0].score

        hand_regions.append({
            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "landmarks": hand_landmarks,
            "label": label,
            "score": score,
        })

    return hand_regions


# State machine logic to control the flow of the program through the three phases: scanning for plates, scanning for targets, and executing pick/place operations.
# THIS STATE MACHINE IS TOO SIMPLE. Can you think of logics that should change the robot's sequnece of actions?
# Ex: what if the robot fails to pick up a target? should it retry? should it go back to scanning for targets in case the target was moved? what if a new plate is added during the pick/place phase?
# What if a human's hand is in sight during pick/place phase? (safety first!)

def next_state():
    global machine_state
    if machine_state == "scanning plate":
        machine_state = "scanning target"
    elif machine_state == "scanning target":
        machine_state = "scanning hand"
    elif machine_state == "scanning hand":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning plate"
    else:
        machine_state = "scanning plate"



# ---------------------------------------------------------
# PHASE 1: DETECT Part Drop Zones (Plates)
# this script assumes a metallic circular plate as the drop zone, but you can modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_plates():
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=25, maxRadius=55)

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        # --- AUTO-LOCK LOGIC ---
        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list
 

# ---------------------------------------------------------
# PHASE 2: DETECT Red velcros to pick up (Red Blocks)
# this script assumes the targets to be picked up are red blocks
# be aware your target maynot be red, and they may not be rectangular! You will need to modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_targets():
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_positions = []
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        detections, mask = detect_targets(frame)

        current_pixels = [detection["pixel"] for detection in detections]
        current_robots = [detection["robot"] for detection in detections]

        for detection in detections:
            cx, cy = detection["pixel"]
            cv2.drawContours(display_frame, [detection["contour"]], -1, (0, 255, 0), 2)
            cv2.circle(display_frame, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(
                display_frame,
                f"A={int(detection['area'])}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        # --- STABILITY LOGIC ---
        if points_are_stable(current_pixels, last_positions, PIXEL_TOLERANCE):
            stability_counter += 1
        else:
            stability_counter = 0
        last_positions = current_pixels

        # Visual Feedback
        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Detection", display_frame)
        if SHOW_MASK_WINDOW:
            cv2.imshow("Red Mask", mask)
        cv2.waitKey(1)
        
        # --- EXIT CONDITION ---
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_robots)} targets.")
            return current_robots


# ---------------------------------------------------------
# PHASE 3: DETECT HUMAN HANDS IN THE WORKSPACE
# This phase pauses robot motion until the workspace has been clear of hands for a short period.
# ---------------------------------------------------------
def phase_detect_hands():
    print("\n[PHASE 3] Checking workspace for hands before robot motion...")
    clear_counter = 0
    hand_present_last_frame = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        hand_regions = detect_hands(frame)

        if hand_regions:
            if not hand_present_last_frame:
                print("[ALERT] Hand detected in workspace. Robot is waiting.")
            hand_present_last_frame = True
            clear_counter = 0

            for region in hand_regions:
                x, y, w_box, h_box = region["bbox"]
                cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                mp_drawing.draw_landmarks(
                    display_frame,
                    region["landmarks"],
                    mp_hands.HAND_CONNECTIONS,
                )
                cv2.putText(
                    display_frame,
                    f"{region['label']} {region['score']:.2f}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
        else:
            if hand_present_last_frame:
                print("[INFO] Workspace looks clear. Verifying before motion resumes.")
            hand_present_last_frame = False
            clear_counter += 1

        progress = int((clear_counter / HAND_CLEAR_LIMIT) * 100)
        status_text = f"WORKSPACE CLEAR: {min(progress, 100)}%"
        status_color = (0, 255, 0) if not hand_regions else (0, 0, 255)
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.imshow("Hand Detection", display_frame)
        cv2.waitKey(1)

        if clear_counter >= HAND_CLEAR_LIMIT:
            print("[SUCCESS] Workspace clear. Proceeding to robot motion.")
            return True


# ---------------------------------------------------------
# PHASE 4: PICK/PLACE LOOP
# This function assumes 1 drop zone only has 1 part, and executes the pick/place operations in batches.
# if you are picking up rigid car parts, would you still be able to move directly to the object and to the drop zone? 
# Do you need collision avoidance? Think about if the robot gripper accidentally hits the plate or other parts on the way to the target, what would happen? How would you modify the robot's movement logic to avoid collisions?
# ---------------------------------------------------------
def phase_execute_batch(api, pick_list, drop_list):
    cv2.VideoCapture(0)
    time.sleep(0.5)
    
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("missing targets, aborting")
        return False
    
    # Match 1 part to 1 drop zone (uses the smaller count)
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 4] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y = pick_list[i]
        drop_x, drop_y = drop_list[i]

        print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y}")

        if not phase_detect_hands():
            return False

        picked = False
        for attempt in range(1, RETRY_LIMIT + 1):
            print(f"  Pick attempt {attempt}/{RETRY_LIMIT}")

            # Use a staged descent so the arm does not dive straight into the workspace.
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_APPROACH)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
            dobotArm.close_gripper(api)
            time.sleep(0.25)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_APPROACH)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            picked = True
            break

        if not picked:
            print("  Unable to secure the target, rescanning is required.")
            return False

        # --- PLACE SEQUENCE ---
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_PLACE)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        time.sleep(0.2)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    # irl, it is ok for 1 dish to contain multiple parts
    # if len(pick_list) > len(drop_list):
    #     for i in range(len(pick_list)):
    #         pick_x, pick_y = pick_list[i]
    #         drop_x, drop_y = drop_list[0]
    #         # --- PICK SEQUENCE ---
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
    #         dobotArm.close_gripper(api)
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

    #     # --- PLACE SEQUENCE ---
    #         dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
    #         dobotArm.open_gripper(api)
    #         dobotArm.stop_pump(api)
    #         dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("\nBatch Complete.")
    return True
 

# ---------------------------------------------------------
# MAIN EXECUTION
# contains an oversimplified state machine that runs the three phases sequentially. You can modify the logic to fit your specific use case.
# ---------------------------------------------------------
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

while True:
    if machine_state == "scanning plate":
        drop_zone = phase_detect_plates()
        if drop_zone is not None:
            next_state()
        continue

    if machine_state == "scanning target":
        pick_target = phase_detect_targets()
        if pick_target is not None:
            next_state()
        continue

    if machine_state == "scanning hand":
        workspace_clear = phase_detect_hands()
        if workspace_clear:
            next_state()
        continue

    if machine_state == "pick place":
        completed = phase_execute_batch(api, pick_target, drop_zone)
        if completed:
            next_state()
        else:
            machine_state = "scanning target"
        continue

    break


cap.release()
cv2.destroyAllWindows()
