# AI Virtual Mouse - Hand Gesture Control
# Author: Asik Dial Kuffer
# Date: January 2025

# --------------------------- IMPORTS ---------------------------
import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol

# --------------------------- SETUP ---------------------------
pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --------------------------- ENUMERATIONS ---------------------------
class Gesture(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MIDDLE = 4
    LAST_THREE = 7
    INDEX = 8
    FIRST_TWO = 12
    LAST_FOUR = 15
    THUMB = 16
    PALM = 31
    V_SIGN = 33
    TWO_FINGERS_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

class HandLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# --------------------------- HAND RECOGNITION CLASS ---------------------------
class HandRecognition:
    def __init__(self, hand_label):
        self.finger_state = 0
        self.current_gesture = Gesture.PALM
        self.previous_gesture = Gesture.PALM
        self.frame_count = 0
        self.hand_landmarks = None
        self.hand_label = hand_label

    def update_landmarks(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks

    def calculate_signed_distance(self, points):
        """Calculate signed distance between two landmarks."""
        sign = -1 if self.hand_landmarks.landmark[points[0]].y < self.hand_landmarks.landmark[points[1]].y else 1
        distance = math.sqrt(
            (self.hand_landmarks.landmark[points[0]].x - self.hand_landmarks.landmark[points[1]].x) ** 2 +
            (self.hand_landmarks.landmark[points[0]].y - self.hand_landmarks.landmark[points[1]].y) ** 2
        )
        return distance * sign

    def calculate_distance(self, points):
        """Calculate distance between two landmarks."""
        return math.sqrt(
            (self.hand_landmarks.landmark[points[0]].x - self.hand_landmarks.landmark[points[1]].x) ** 2 +
            (self.hand_landmarks.landmark[points[0]].y - self.hand_landmarks.landmark[points[1]].y) ** 2
        )

    def calculate_depth_difference(self, points):
        """Calculate absolute z-axis difference."""
        return abs(self.hand_landmarks.landmark[points[0]].z - self.hand_landmarks.landmark[points[1]].z)

    def set_finger_state(self):
        """Determine the state of fingers."""
        if not self.hand_landmarks:
            return
        self.finger_state = 0
        finger_points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        for points in finger_points:
            ratio = round(
                self.calculate_signed_distance(points[:2]) / self.calculate_signed_distance(points[1:]), 1
            )
            self.finger_state = (self.finger_state << 1) | (1 if ratio > 0.5 else 0)

    def determine_gesture(self):
        """Determine the gesture based on finger state."""
        if not self.hand_landmarks:
            return Gesture.PALM

        if self.finger_state in [Gesture.LAST_THREE, Gesture.LAST_FOUR] and self.calculate_distance([8, 4]) < 0.05:
            return Gesture.PINCH_MAJOR if self.hand_label == HandLabel.MAJOR else Gesture.PINCH_MINOR
        elif self.finger_state == Gesture.FIRST_TWO:
            ratio = self.calculate_distance([8, 12]) / self.calculate_distance([5, 9])
            if ratio > 1.7:
                return Gesture.V_SIGN
            elif self.calculate_depth_difference([8, 12]) < 0.1:
                return Gesture.TWO_FINGERS_CLOSED
        else:
            return self.finger_state

        return Gesture.PALM

# --------------------------- CONTROLLER CLASS ---------------------------
class GestureController:
    def __init__(self):
        self.capture_device = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hand_major = HandRecognition(HandLabel.MAJOR)
        self.hand_minor = HandRecognition(HandLabel.MINOR)

    def classify_hands(self, results):
        """Classify hands into major and minor based on handedness."""
        left, right = None, None
        try:
            for hand in results.multi_handedness:
                label = MessageToDict(hand)['classification'][0]['label']
                if label == "Right":
                    right = hand
                else:
                    left = hand
        except:
            pass

        self.hand_major.update_landmarks(right)
        self.hand_minor.update_landmarks(left)

    def start(self):
        """Start the gesture control loop."""
        while self.capture_device.isOpened():
            success, frame = self.capture_device.read()
            if not success:
                print("Failed to capture frame.")
                continue

            # Process frame
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame)
            if results.multi_hand_landmarks:
                self.classify_hands(results)

                # Gesture detection and control
                major_gesture = self.hand_major.determine_gesture()
                minor_gesture = self.hand_minor.determine_gesture()
                print(f"Major Gesture: {major_gesture}, Minor Gesture: {minor_gesture}")

                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("AI Virtual Mouse", frame)

            # Exit on 'a' key
            if cv2.waitKey(10) & 0xFF == ord('a'):
                break

        # Cleanup
        self.capture_device.release()
        cv2.destroyAllWindows()


# Run the program
if __name__ == "__main__":
    gesture_controller = GestureController()
    gesture_controller.start()
