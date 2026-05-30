"""
Human Safety Monitor — vision-based hand detection for robot collaboration.

Uses MediaPipe Hands to detect when a human hand enters the robot workspace.
Runs in a background thread so the main program isn't blocked by camera reads.

Install dependency: pip install mediapipe
"""

import cv2
import numpy as np
import threading
import time

try:
    import mediapipe as mp
    _MEDIAPIPE_OK = True
except ImportError:
    _MEDIAPIPE_OK = False
    print("[SAFETY] mediapipe not found. Run:  pip install mediapipe")
    print("[SAFETY] Falling back to skin-colour detection (less reliable).")


class HumanSafetyMonitor:
    """
    Monitors the camera for human hands.  Call start() before the robot moves
    and stop() when it is done.  The robot checks is_human_detected() or calls
    wait_for_clear() to block until the workspace is empty.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Shared camera object.  The monitor should be the ONLY reader while it
        is running (don't read from cap in the main thread simultaneously).
    undistort_maps : (map1, map2) or None
        Pre-computed undistort maps from calibrateCamera.py.  Pass None to skip.
    workspace_pixels : (x1, y1, x2, y2) or None
        Pixel bounding box of the robot workspace in the camera frame.
        Detections outside this box are ignored.  None = full frame.
    """

    def __init__(self, cap, undistort_maps=None, workspace_pixels=None):
        self.cap = cap
        self.undistort_maps = undistort_maps
        self.workspace_pixels = workspace_pixels

        self.human_detected = False
        self._running = False
        self._thread = None
        self._last_display_frame = None

        if _MEDIAPIPE_OK:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
        else:
            self._hands = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the background monitoring thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[SAFETY] Human safety monitor started.")

    def stop(self):
        """Stop the background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[SAFETY] Human safety monitor stopped.")

    def is_human_detected(self):
        """Return True if a hand is currently visible in the workspace."""
        return self.human_detected

    def wait_for_clear(self):
        """Block the calling thread until no human is detected."""
        if self.human_detected:
            print("[SAFETY] Human in workspace — robot paused. Remove hand to resume.")
            while self.human_detected and self._running:
                time.sleep(0.05)
            print("[SAFETY] Workspace clear — resuming.")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.033)
                continue

            if self.undistort_maps is not None:
                map1, map2 = self.undistort_maps
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            # Crop to the declared workspace region before detection
            if self.workspace_pixels is not None:
                x1, y1, x2, y2 = self.workspace_pixels
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame

            if _MEDIAPIPE_OK:
                self.human_detected = self._detect_mediapipe(roi)
            else:
                self.human_detected = self._detect_skin_colour(roi)

            self._draw_overlay(frame)
            time.sleep(0.033)  # ~30 fps is enough for safety

    def _detect_mediapipe(self, roi):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        return results.multi_hand_landmarks is not None

    def _detect_skin_colour(self, roi):
        """
        Fallback: detect large skin-coloured regions as a hand proxy.
        Less accurate than MediaPipe — tune MIN_SKIN_PIXELS for your lighting.
        """
        MIN_SKIN_PIXELS = 4000
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,  20,  60]), np.array([20, 255, 255]))
        mask |= cv2.inRange(hsv, np.array([170, 20, 60]), np.array([180, 255, 255]))
        return int(cv2.countNonZero(mask)) > MIN_SKIN_PIXELS

    def _draw_overlay(self, frame):
        """Annotate the frame and show it; also keep a reference for callers."""
        display = frame.copy()
        if self.workspace_pixels is not None:
            x1, y1, x2, y2 = self.workspace_pixels
            colour = (0, 0, 255) if self.human_detected else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)

        label = "!! HUMAN DETECTED — ROBOT STOPPED !!" if self.human_detected else "Workspace Clear"
        colour = (0, 0, 255) if self.human_detected else (0, 200, 0)
        cv2.putText(display, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.imshow("Safety Monitor", display)
        cv2.waitKey(1)
        self._last_display_frame = display
