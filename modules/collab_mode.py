import cv2
import numpy as np

STABILITY_LIMIT = 60
Z_SAFE = 40
Z_PICK = -35

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

class CollabModeLogic:
    def __init__(self, map1, map2, H_matrix):
        self.map1 = map1
        self.map2 = map2
        self.H_matrix = H_matrix
        self.state = "scanning plate"
        self.stability_counter = 0
        self.last_count = 0
        
        self.drop_zone = []
        self.pick_target = []

    def update_collab_state(self, frame, display_frame, robot):
        frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        display_frame[:] = frame.copy() # output to display
        
        if self.state == "scanning plate":
            self._scan_plates(frame, display_frame)
        elif self.state == "scanning target":
            self._scan_targets(frame, display_frame)
        elif self.state == "pick place":
            if not robot.busy.is_set():
                print(f"[COLLAB] Queuing batch execution. Targets: {len(self.pick_target)}, Plates: {len(self.drop_zone)}")
                robot.send("batch", self.pick_target, self.drop_zone, Z_SAFE, Z_PICK)
                self.state = "scanning plate"
                self.stability_counter = 0
                self.last_count = 0
            
        # HUD for Collab Mode
        cv2.putText(display_frame, f"COLLAB STATE: {self.state.upper()}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 0), 2)

    def _scan_plates(self, frame, display_frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=25, maxRadius=55)

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], self.H_matrix)
                current_list.append((rx, ry))

        if len(current_list) > 0 and len(current_list) == self.last_count:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            self.last_count = len(current_list)

        progress = int((self.stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if self.stability_counter >= STABILITY_LIMIT:
            self.drop_zone = current_list
            print(f"[SUCCESS] Locked {len(current_list)} plates.")
            self.state = "scanning target"
            self.stability_counter = 0
            self.last_count = 0

    def get_stable_target(self, idx, new_x, new_y):
        from collections import deque
        if not hasattr(self, 'coord_histories'):
            from collections import defaultdict
            self.coord_histories = defaultdict(lambda: deque(maxlen=5))
        self.coord_histories[idx].append((new_x, new_y))
        return np.mean(self.coord_histories[idx], axis=0)

    def _scan_targets(self, frame, display_frame):
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5,5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,100,50]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([160,120,70]), np.array([180,255,255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_list = []
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 150]
        
        for idx, cnt in enumerate(valid_contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                rx, ry = pixel_to_robot(cx, cy, self.H_matrix)
                smooth_x, smooth_y = self.get_stable_target(idx, rx, ry)
                current_list.append((smooth_x, smooth_y))
                cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)

        if len(current_list) != 0:
            if len(current_list) == self.last_count:
                self.stability_counter += 1
            else:
                self.stability_counter = 0
                self.last_count = len(current_list)

        progress = int((self.stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.stability_counter >= STABILITY_LIMIT:
            self.pick_target = current_list
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            self.state = "pick place"
            self.stability_counter = 0
            self.last_count = 0
