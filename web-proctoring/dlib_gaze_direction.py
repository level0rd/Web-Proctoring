import numpy as np
import cv2
from enum import Enum
from face_detectors import Ultralight320Detector

LEFT_GAZE_THRESHOLD = 0.85
RIGHT_GAZE_THRESHOLD = 1.6
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]


class GazeDirection(Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


def get_gaze_ratio(
        eye_points: [int],
        facial_landmarks: np.ndarray,
        height: int,
        width: int,
        gray: np.ndarray
) -> float:
    """
        Calculate the ratio of white pixels in eye regions.

        Args:
            eye_points (list[int]): List of indices representing eye points.
            facial_landmarks (np.ndarray): Array of facial landmarks.
            height (int): Height of the image.
            width (int): Width of the image.
            gray (np.ndarray): Grayscale image.

        Returns:
            float: The gaze ratio calculated based on the white pixel ratio.
        """
    eye_region = np.array([(facial_landmarks[eye_points[0]][0], facial_landmarks[eye_points[0]][1]),
                           (facial_landmarks[eye_points[1]][0], facial_landmarks[eye_points[1]][1]),
                           (facial_landmarks[eye_points[2]][0], facial_landmarks[eye_points[2]][1]),
                           (facial_landmarks[eye_points[3]][0], facial_landmarks[eye_points[3]][1]),
                           (facial_landmarks[eye_points[4]][0], facial_landmarks[eye_points[4]][1]),
                           (facial_landmarks[eye_points[5]][0], facial_landmarks[eye_points[5]][1])], np.int32)

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0: height, 0: round(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, round(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ration = 1
    elif right_side_white == 0:
        gaze_ration = 5
    else:
        gaze_ration = left_side_white / right_side_white

    return gaze_ration


def get_gaze_direction(
        img: np.ndarray,
        img_height: int,
        img_width: int,
        face_detector: Ultralight320Detector
) -> str:
    """
    Calculate the direction of the eye (left, center, right).

    Args:
        img (np.ndarray): Input image.
        img_height (int): Height of the image.
        img_width (int): Width of the image.

    Returns:
        str: The direction of the gaze.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_keypoints = face_detector.detect_faces_keypoints(gray, get_all=True)

    landmarks = face_keypoints[0]['keypoints']

    gaze_ration_left_eye = get_gaze_ratio(LEFT_EYE_POINTS, landmarks, img_height, img_width, gray)
    gaze_ration_right_eye = get_gaze_ratio(RIGHT_EYE_POINTS, landmarks, img_height, img_width, gray)
    gaze_ratio = (gaze_ration_right_eye + gaze_ration_left_eye) / 2

    if gaze_ratio <= LEFT_GAZE_THRESHOLD:
        gaze_direction = GazeDirection.LEFT.value
    elif LEFT_GAZE_THRESHOLD < gaze_ratio < RIGHT_GAZE_THRESHOLD:
        gaze_direction = GazeDirection.CENTER.value
    else:
        gaze_direction = GazeDirection.RIGHT.value

    return gaze_direction
