import os
import sys
import math
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "mediapipe is required for real hand detection. "
        "Run this script with the project's virtualenv Python."
    ) from exc

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "The installed mediapipe package does not expose mp.solutions. "
        "Use the standard mediapipe package in the project virtualenv."
    )


CAMERA_CANDIDATES       = (0,)   # laptop built-in camera
MAX_HANDS               = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
DEBOUNCE_FRAMES         = 18
BOMB_TOTAL_FRAMES       = 45   # how long the full explosion animation plays

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Gesture map ───────────────────────────────────────────────────────────────
GESTURE_INFO = {
    "TRACK":        ("1 - TRACK XY",       (0, 220, 0)),
    "MOVE_UP":      ("2 - MOVE UP",        (0, 200, 255)),
    "MOVE_DN":      ("3 - MOVE DOWN",      (0, 140, 255)),
    "GRIP_TOGGLE":  ("4 - GRIP TOGGLE",    (255, 180, 0)),
    "OPEN_GRIP":    ("5 - OPEN GRIP",      (255, 220, 0)),
    "HOME":         ("6 - HOME",           (200, 0, 255)),
    "MOVE_LEFT":    ("7 - MOVE LEFT",      (0, 255, 200)),
    "MOVE_RIGHT":   ("8 - MOVE RIGHT",     (255, 100, 200)),
    "MIDDLE_FINGER":("💣 BOOM!",           (0, 0, 255)),
    "NONE":         ("-- HOLD (no gesture)","grey"),
}

_LEGEND = [
    "1  Open palm    -> TRACK XY",
    "2  Index only   -> MOVE UP",
    "3  Peace sign   -> MOVE DOWN",
    "4  Fist         -> GRIP TOGGLE",
    "5  3 fingers    -> OPEN GRIP",
    "6  Thumb up     -> HOME",
    "7  Pinky only   -> MOVE LEFT",
    "8  Ring only    -> MOVE RIGHT",
    "   Middle only  -> BOOM!",
    "   Other        -> HOLD",
]


# ── Gesture Recognition ───────────────────────────────────────────────────────

def _fingers_extended(lm, handedness):
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    # Require a larger gap (0.02 normalised units) so thumb must be clearly extended
    THUMB_THRESHOLD = 0.03
    if handedness == "Right":
        thumb_up = (lm[3].x - lm[4].x) > THUMB_THRESHOLD
    else:
        thumb_up = (lm[4].x - lm[3].x) > THUMB_THRESHOLD
    return thumb_up, index_up, middle_up, ring_up, pinky_up


def classify_gesture(hand_landmarks, handedness="Right"):
    lm = hand_landmarks.landmark
    thumb, idx, mid, ring, pinky = _fingers_extended(lm, handedness)

    if thumb and not idx and not mid and not ring and not pinky:
        return "HOME"

    # Middle finger only — bomb!
    if mid and not idx and not ring and not pinky and not thumb:
        return "MIDDLE_FINGER"

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
    def __init__(self, required=DEBOUNCE_FRAMES):
        self.required = required
        self.current  = "NONE"
        self.count    = 0

    def update(self, gesture):
        if gesture == self.current:
            self.count = min(self.count + 1, self.required)
        else:
            self.current = gesture
            self.count   = 1
        return self.current if self.count >= self.required else None


# ── Bomb Animation ────────────────────────────────────────────────────────────

class BombAnimation:
    """
    Full-screen explosion animation triggered by the middle-finger gesture.
    Plays for BOMB_TOTAL_FRAMES frames then resets.

    Phases:
      0 – 14  : white flash that fades
      5 – 44  : shockwave ring expanding from centre
      0 – 44  : sparks flying outward
      10 – 44 : "BOOM!" text slamming in from top
    """

    def __init__(self):
        self.frame_idx = -1          # -1 = not playing
        self._sparks   = []          # list of (x, y, vx, vy, color, size)

    def trigger(self, w, h):
        """Start the animation."""
        self.frame_idx = 0
        cx, cy = w // 2, h // 2
        # Generate sparks in random directions
        self._sparks = []
        colors = [
            (0, 80, 255), (0, 140, 255), (0, 200, 255),
            (0, 255, 255), (0, 255, 180), (255, 255, 255),
        ]
        for _ in range(80):
            angle  = random.uniform(0, 2 * math.pi)
            speed  = random.uniform(8, 28)
            vx     = math.cos(angle) * speed
            vy     = math.sin(angle) * speed
            color  = random.choice(colors)
            size   = random.randint(3, 9)
            self._sparks.append([cx, cy, vx, vy, color, size])

    @property
    def active(self):
        return self.frame_idx >= 0

    def draw(self, frame):
        if not self.active:
            return

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        t = self.frame_idx
        total = BOMB_TOTAL_FRAMES

        # ── White flash (frames 0-14) ─────────────────────────────────────────
        if t < 15:
            alpha = 1.0 - (t / 15.0)
            flash = np.full_like(frame, 255)
            cv2.addWeighted(flash, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)

        # ── Expanding shockwave ring ──────────────────────────────────────────
        if t >= 3:
            ring_t     = t - 3
            max_radius = int(math.sqrt(w**2 + h**2))
            radius     = int((ring_t / (total - 3)) * max_radius)
            thickness  = max(2, 18 - ring_t // 2)
            alpha_ring = max(0.0, 1.0 - ring_t / (total * 0.7))
            overlay    = frame.copy()
            cv2.circle(overlay, (cx, cy), radius, (0, 180, 255), thickness)
            cv2.addWeighted(overlay, alpha_ring, frame, 1 - alpha_ring, 0, frame)

            # Second inner ring
            if radius > 40:
                cv2.circle(overlay, (cx, cy), radius - 40, (0, 80, 255), max(1, thickness - 6))
                cv2.addWeighted(overlay, alpha_ring * 0.5, frame, 1 - alpha_ring * 0.5, 0, frame)

        # ── Sparks ────────────────────────────────────────────────────────────
        for spark in self._sparks:
            sx, sy, vx, vy, color, size = spark
            # Update position
            spark[0] += vx
            spark[1] += vy
            spark[3] += 1.5   # gravity
            sx, sy = int(spark[0]), int(spark[1])
            fade = max(0.0, 1.0 - t / total)
            r = max(1, int(size * fade))
            if 0 <= sx < w and 0 <= sy < h:
                cv2.circle(frame, (sx, sy), r, color, -1)
                # Spark trail
                trail_x = int(sx - vx * 0.5)
                trail_y = int(sy - vy * 0.5)
                if 0 <= trail_x < w and 0 <= trail_y < h:
                    cv2.line(frame, (sx, sy), (trail_x, trail_y), color, max(1, r - 1))

        # ── BOOM! text slamming in ────────────────────────────────────────────
        if t >= 8:
            boom_t     = t - 8
            font       = cv2.FONT_HERSHEY_DUPLEX
            # Scale: starts huge, settles to normal
            scale      = max(3.0, 9.0 - boom_t * 0.25)
            thickness  = max(3, int(scale * 2))
            text       = "BOOM!"
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx = (w - tw) // 2
            # Slam from top
            target_y   = h // 2 + th // 2
            start_y    = -th
            progress   = min(1.0, boom_t / 10.0)
            # Bounce easing
            if progress < 0.8:
                ease = progress / 0.8
            else:
                ease = 1.0 - (progress - 0.8) / 0.2 * 0.15
            ty = int(start_y + (target_y - start_y) * ease)

            fade_out = max(0.0, 1.0 - (t - 20) / 20.0) if t > 20 else 1.0

            # Black outline
            cv2.putText(frame, text, (tx + 4, ty + 4), font, scale, (0, 0, 0), thickness + 4)
            # Colour cycling: red → orange → yellow
            hue     = int((t * 8) % 180)
            hsv_col = np.uint8([[[hue, 255, 255]]])
            bgr_col = cv2.cvtColor(hsv_col, cv2.COLOR_HSV2BGR)[0][0]
            color   = (int(bgr_col[0]), int(bgr_col[1]), int(bgr_col[2]))
            if fade_out < 1.0:
                txt_overlay = frame.copy()
                cv2.putText(txt_overlay, text, (tx, ty), font, scale, color, thickness)
                cv2.addWeighted(txt_overlay, fade_out, frame, 1 - fade_out, 0, frame)
            else:
                cv2.putText(frame, text, (tx, ty), font, scale, color, thickness)

        # Advance frame counter
        self.frame_idx += 1
        if self.frame_idx >= total:
            self.frame_idx = -1


# ── Camera helpers ────────────────────────────────────────────────────────────

def open_camera(camera_candidates=CAMERA_CANDIDATES):
    for camera_index in camera_candidates:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue
        print(f"Opened camera index {camera_index}")
        return cap, camera_index

    tried = ", ".join(str(idx) for idx in camera_candidates)
    raise RuntimeError(
        "Could not open any camera. "
        f"Tried indices: {tried}. "
        "Close any app that may be using the camera, then allow camera access for your terminal "
        "or editor in macOS Settings > Privacy & Security > Camera."
    )


def detect_hands(frame, hands_detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands_detector.process(rgb_frame)
    return results


def landmark_bounds(hand_landmarks, frame_shape):
    height, width = frame_shape[:2]
    xs = [int(landmark.x * width) for landmark in hand_landmarks.landmark]
    ys = [int(landmark.y * height) for landmark in hand_landmarks.landmark]
    return max(min(xs), 0), max(min(ys), 0), min(max(xs), width - 1), min(max(ys), height - 1)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_legend(frame):
    h, w = frame.shape[:2]
    for i, line in enumerate(_LEGEND):
        cv2.putText(
            frame, line,
            (w - 285, h - len(_LEGEND) * 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1,
        )


def draw_hud(frame, raw_gesture, confirmed_gesture, detected_hands, bomb_active):
    h, w = frame.shape[:2]

    GREY  = (180, 180, 180)
    GREEN = (0, 220, 0)
    RED   = (0, 0, 220)
    AMBER = (0, 170, 255)

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    status_text  = f"HANDS DETECTED: {detected_hands}" if detected_hands else "NO HANDS DETECTED"
    status_color = GREEN if detected_hands else RED
    cv2.putText(frame, status_text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Raw: {raw_gesture}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREY, 1)

    # Bottom command banner — hide during bomb animation
    if not bomb_active:
        if confirmed_gesture and confirmed_gesture != "NONE":
            label, color = GESTURE_INFO.get(confirmed_gesture, (confirmed_gesture, AMBER))
            if color == "grey":
                color = GREY
        else:
            label, color = "HOLD -- no gesture", AMBER

        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.6
        thickness  = 3
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        banner_pad = 18
        bx1 = (w - tw) // 2 - banner_pad
        by1 = h - th - baseline - banner_pad * 2 - 10
        bx2 = (w + tw) // 2 + banner_pad
        by2 = h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
        tx = (w - tw) // 2
        ty = by2 - banner_pad - baseline
        cv2.putText(frame, label, (tx, ty), font, font_scale, color, thickness)

    draw_legend(frame)


def draw_finger_debug(frame, hand_landmarks, handedness, x1, y2):
    """Show T/I/M/R/P on-screen so we can see exactly which fingers are detected."""
    lm = hand_landmarks.landmark
    thumb, idx, mid, ring, pinky = _fingers_extended(lm, handedness)
    fingers = [
        ("T", thumb),
        ("I", idx),
        ("M", mid),
        ("R", ring),
        ("P", pinky),
    ]
    ox = x1
    for name, up in fingers:
        color = (0, 255, 0) if up else (0, 0, 200)
        cv2.putText(frame, name, (ox, y2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ox += 22


def draw_hand_annotations(frame, results, debouncers):
    detected_hands = 0
    raw_gesture    = "NONE"
    confirmed      = None

    if not results.multi_hand_landmarks:
        for d in debouncers:
            d.update("NONE")
        return detected_hands, raw_gesture, confirmed

    handedness_list = results.multi_handedness or []

    while len(debouncers) < len(results.multi_hand_landmarks):
        debouncers.append(GestureDebouncer())
    while len(debouncers) > len(results.multi_hand_landmarks):
        debouncers.pop()

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        x1, y1, x2, y2 = landmark_bounds(hand_landmarks, frame.shape)

        side = "Right"
        label_text = "Hand"
        score = None
        if idx < len(handedness_list) and handedness_list[idx].classification:
            classification = handedness_list[idx].classification[0]
            side       = classification.label
            label_text = classification.label
            score      = classification.score

        gesture   = classify_gesture(hand_landmarks, side)
        confirmed = debouncers[idx].update(gesture)

        if idx == 0:
            raw_gesture = gesture

        box_color = (0, 255, 0)
        if confirmed and confirmed != "NONE":
            _, box_color = GESTURE_INFO.get(confirmed, ("", (0, 255, 0)))
            if box_color == "grey":
                box_color = (180, 180, 180)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        hand_label = f"{label_text} Hand"
        if score is not None:
            hand_label = f"{hand_label} {score:.2f}"

        cv2.putText(
            frame, hand_label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2,
        )

        # Debug: show T I M R P finger states below bounding box
        draw_finger_debug(frame, hand_landmarks, side, x1, y2)

        detected_hands += 1

    return detected_hands, raw_gesture, confirmed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap, camera_index = open_camera()
    print(f"Press 'q' to quit. Using camera index {camera_index}.")

    debouncers = []
    bomb       = BombAnimation()
    bomb_armed = True   # prevents re-triggering while gesture is held

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read a frame from the camera.")
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            results       = detect_hands(frame, hands_detector)
            detected_hands, raw_gesture, confirmed = draw_hand_annotations(
                frame, results, debouncers
            )

            # Trigger bomb on confirmed MIDDLE_FINGER (fire once per hold)
            if confirmed == "MIDDLE_FINGER" and bomb_armed and not bomb.active:
                bomb.trigger(w, h)
                bomb_armed = False
            elif confirmed != "MIDDLE_FINGER":
                bomb_armed = True   # re-arm when gesture released

            # Draw HUD then bomb on top
            draw_hud(frame, raw_gesture, confirmed, detected_hands, bomb.active)
            bomb.draw(frame)

            cv2.imshow("Hand Detection — Gesture Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def print_runtime_help(error):
    print(f"Error: {error}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Run with the project virtual environment:", file=sys.stderr)
    print("  source .venv/bin/activate", file=sys.stderr)
    print("  python hand_detection.py", file=sys.stderr)
    print("", file=sys.stderr)
    print("Or run it directly without activating:", file=sys.stderr)
    print("  ./.venv/bin/python hand_detection.py", file=sys.stderr)
    print("", file=sys.stderr)
    print("If macOS blocks the camera, enable access for the app you launched from:", file=sys.stderr)
    print("  System Settings > Privacy & Security > Camera", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print_runtime_help(error)
        raise SystemExit(1) from error
