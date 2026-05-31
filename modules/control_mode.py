import time
import dobotArm

# Safe workspace bounds (mm in Dobot frame). Arm will not be commanded outside these.
WS_X = (150, 310)
WS_Y = (-140, 140)
WS_Z = (-25,  100)

Z_HOVER            = 50       # mm
Z_STEP             = 15       # mm
TRACK_DEADZONE_MM  = 8        # mm
MOVE_HOLD_INTERVAL = 0.7      # seconds

def _fingers_extended(lm, handedness):
    """Return (thumb_up, index_up, middle_up, ring_up, pinky_up)."""
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
    return "HOLD"

class GestureDebouncer:
    def __init__(self, required=18):
        self.required = required
        self.current_gesture = "NONE"
        self.count = 0
        self.confirmed = "NONE"

    def update(self, raw_gesture):
        if raw_gesture == self.current_gesture:
            self.count += 1
        else:
            self.current_gesture = raw_gesture
            self.count = 1

        if self.count >= self.required:
            self.confirmed = self.current_gesture
            self.count = self.required
        return self.confirmed

def clamp(x, y, z):
    x = max(WS_X[0], min(WS_X[1], x))
    y = max(WS_Y[0], min(WS_Y[1], y))
    z = max(WS_Z[0], min(WS_Z[1], z))
    return x, y, z

def hand_in_workspace(rx, ry):
    return WS_X[0] < rx < WS_X[1] and WS_Y[0] < ry < WS_Y[1]

class ControlModeLogic:
    def __init__(self, debounce_frames=18):
        self.dbnc = GestureDebouncer(required=debounce_frames)
        self.arm_x = float(dobotArm.home_pos[0])
        self.arm_y = float(dobotArm.home_pos[1])
        self.arm_z = float(Z_HOVER)
        self.last_move_time = 0.0

    def process(self, hand_landmarks, handedness_label, H_matrix, robot):
        raw = classify_gesture(hand_landmarks, handedness_label)
        
        lm = hand_landmarks.landmark
        wrist = lm[0]
        
        # Calculate wrist position in mm (robot coordinates)
        # Note: wrist.x and wrist.y are normalized [0, 1]
        # We assume H_matrix expects pixel coordinates, so we'll pass (wrist.x * W, wrist.y * H) in main.py
        # Actually, let's keep it clean. H_matrix is passed in, but we need W and H.
        # We will pass rx, ry directly instead.
        pass

    def update_control_state(self, hand_landmarks, handedness_label, rx, ry, robot):
        raw = classify_gesture(hand_landmarks, handedness_label)
        
        safety_stop = hand_in_workspace(rx, ry)
        if safety_stop:
            raw = "HOLD"

        confirmed = self.dbnc.update(raw)
        
        if confirmed == "TRACK":
            dx = rx - self.arm_x
            dy = ry - self.arm_y
            if (dx*dx + dy*dy) ** 0.5 > TRACK_DEADZONE_MM:
                self.arm_x, self.arm_y, self.arm_z = clamp(rx, ry, Z_HOVER)
                robot.send("xyz", self.arm_x, self.arm_y, self.arm_z, drop_if_busy=True)

        elif confirmed in ("MOVE_UP", "MOVE_DN"):
            now = time.time()
            if now - self.last_move_time > MOVE_HOLD_INTERVAL:
                dz = Z_STEP if confirmed == "MOVE_UP" else -Z_STEP
                self.arm_x, self.arm_y, self.arm_z = clamp(self.arm_x, self.arm_y, self.arm_z + dz)
                robot.send("xyz", self.arm_x, self.arm_y, self.arm_z, drop_if_busy=True)
                self.last_move_time = now

        elif confirmed == "GRIP_TOGGLE":
            if not robot.gripper_closed:
                robot.send("close")
            else:
                robot.send("open")
            self.dbnc.confirmed = "HOLD"
            self.dbnc.count = 0

        elif confirmed == "OPEN_GRIP":
            if robot.gripper_closed:
                robot.send("open")
            self.dbnc.confirmed = "HOLD"
            self.dbnc.count = 0

        elif confirmed == "HOME":
            robot.send("home")
            self.arm_x = float(dobotArm.home_pos[0])
            self.arm_y = float(dobotArm.home_pos[1])
            self.arm_z = float(Z_HOVER)
            self.dbnc.confirmed = "HOLD"
            self.dbnc.count = 0

        return raw, confirmed, self.arm_x, self.arm_y, self.arm_z, safety_stop
