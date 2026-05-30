import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))

import cv2

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


CAMERA_CANDIDATES = (1, 0, 2)
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


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


def draw_hand_annotations(frame, results):
    detected_hands = 0

    if not results.multi_hand_landmarks:
        return detected_hands

    handedness_list = results.multi_handedness or []

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        x1, y1, x2, y2 = landmark_bounds(hand_landmarks, frame.shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = "Hand"
        score = None
        if idx < len(handedness_list) and handedness_list[idx].classification:
            classification = handedness_list[idx].classification[0]
            label = classification.label
            score = classification.score

        text = f"{label} Hand"
        if score is not None:
            text = f"{text} {score:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        detected_hands += 1

    return detected_hands


def main():
    cap, camera_index = open_camera()

    print(f"Press 'q' to quit. Using camera index {camera_index}.")

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
            results = detect_hands(frame, hands_detector)
            detected_hands = draw_hand_annotations(frame, results)

            status_text = f"HANDS DETECTED: {detected_hands}" if detected_hands else "NO HANDS DETECTED"
            status_color = (0, 255, 0) if detected_hands else (0, 0, 255)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.imshow("Hand Detection", frame)

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
