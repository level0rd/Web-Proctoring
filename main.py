import cv2
import numpy as np
import math
import av
import time
from enum import Enum
from typing import Tuple

import streamlit as st
from streamlit_webrtc import WebRtcMode, RTCConfiguration, webrtc_streamer

from face_detectors import Ultralight320Detector
from openvino.runtime import Core
import tensorflow as tf

import pvcobra
from pvrecorder import PvRecorder


class GazeDirection(Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class HandsPosition(Enum):
    BOTTOM = "BOTTOM"
    TOP = "TOP"


POSE_MODEL = 'models/move_net_openvino.xml'
MOVENET_INPUT_SIZE = 256
COBRA_ACCESS_KEY = 'your_access_key'

FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 2
FONT_THICKNESS = 3

LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

LEFT_GAZE_THRESHOLD = 0.85
RIGHT_GAZE_THRESHOLD = 1.6
MOVE_NET_THRESHOLD = 0.1

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

cobra = pvcobra.create(access_key=COBRA_ACCESS_KEY)
recorder = PvRecorder(frame_length=512, device_index=3)

core = Core()
ov_model = core.read_model(POSE_MODEL)
pose_model = core.compile_model(ov_model)

face_detector = Ultralight320Detector()


def keep_aspect_ratio_resizer(image: tf.Tensor, target_size: int) -> Tuple[tf.Tensor, Tuple[int, int]]:
    """Resizes the image.

    The function resizes the image such that its longer side matches the required
    target_size while keeping the image aspect ratio.

    Args:
        image (tf.Tensor): Input image tensor.
        target_size (int): Target size for the longer side of the image.

    Returns:
        tf.Tensor: Resized and padded image tensor.
        Tuple[int, int]: Tuple containing the target height and width of the image.
    """
    _, height, width, _ = image.shape
    if height > width:
        scale = float(target_size / height)
        target_height = target_size
        scaled_width = math.ceil(width * scale)
        image = tf.image.resize(image, [target_height, scaled_width])
        target_width = round(math.ceil(scaled_width / 32) * 32)
    else:
        scale = float(target_size / width)
        target_width = target_size
        scaled_height = math.ceil(height * scale)
        image = tf.image.resize(image, [scaled_height, target_width])
        target_height = round(math.ceil(scaled_height / 32) * 32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
    return image,  (target_height, target_width)


def get_gaze_ratio(
	eye_points: list[int],
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


def get_gaze_direction(img: np.ndarray, img_height: int, img_width: int) -> str:
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


def second_person_detection(keypoints: np.ndarray) -> bool:
    """
    Detect the presence of a second person in the keypoints.

    Args:
        keypoints (np.ndarray): Array of keypoints.

    Returns:
        bool: True if a second person is detected, False otherwise.
    """
    confidences_per_instance = keypoints[0, :, -1]
    second_person = True if confidences_per_instance[1] >= MOVE_NET_THRESHOLD else False

    return second_person


def get_movenet_keypoints(image_rgb: np.ndarray) -> np.ndarray:
    """
     Process an input image using the MoveNet network to detect human poses.

    Args:
        image_rgb (np.ndarray): RGB image array.

    Returns:
        np.ndarray: Array containing MoveNet keypoints.
    """
    _, image_encoded = cv2.imencode('.jpg', image_rgb)
    image_bytes = image_encoded.tobytes()
    image = tf.compat.v1.image.decode_jpeg(image_bytes)
    image = tf.expand_dims(image, axis=0)

    resized_image, image_shape = keep_aspect_ratio_resizer(image, MOVENET_INPUT_SIZE)
    input_tensor = tf.cast(resized_image, dtype=tf.uint8)

    return pose_model(input_tensor)[0]


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Process each frame of the video stream.

    Args:
        frame (av.VideoFrame): Input video frame.

    Returns:
        av.VideoFrame: Processed video frame.
    """
    image_rgb = frame.to_ndarray(format="rgb24")
    image_height, image_width, _ = image_rgb.shape

    keypoints = get_movenet_keypoints(image_rgb)

    if second_person_detection(keypoints):
        cv2.putText(image_rgb, 'Second person detected!', (10, 260), FONT, FONT_SIZE, RED,  FONT_THICKNESS)

    p10 = (round(keypoints[0][0][28] * image_width), round(keypoints[0][0][27] * image_height))
    p11 = (round(keypoints[0][0][31] * image_width), round(keypoints[0][0][30] * image_height))
    hand_border = (image_height / 100) * 85

    if p11[1] > hand_border or p10[1] > hand_border:
        hands_positions = HandsPosition.BOTTOM.value
    else:
        hands_positions = HandsPosition.TOP.value
    cv2.putText(image_rgb, 'Hands: ' + hands_positions, (10, 90), FONT, FONT_SIZE, RED, FONT_THICKNESS)

    try:
        cv2.putText(image_rgb, 'Eyes: ' + get_gaze_direction(image_rgb, image_height, image_width), (10, 120), FONT,
                    FONT_SIZE, RED, FONT_THICKNESS)
    except IndexError:
        cv2.putText(image_rgb, 'The examinee is absent!', (10, 200), FONT, FONT_SIZE + 1, RED, FONT_THICKNESS)
        print("Ошибка: индекс списка вне диапазона")

    return av.VideoFrame.from_ndarray(image_rgb, format="rgb24")


def process_audio(frame: av.AudioFrame) -> av.AudioFrame:
    """
    Detect voice presence in each audio frame of the stream using Cobra Voice Activity Detection.

    Args:
        frame (av.AudioFrame): Input audio frame.

    Returns:
        av.AudioFrame: Processed audio frame.
    """
    recorder.start()
    while True:
        pcm = recorder.read()
        is_voiced = cobra.process(pcm)
        if is_voiced:
            print("Voice detecte!")


def main():
	streaming_placeholder = st.empty()
	
	with streaming_placeholder.container():
	    webrtc_ctx = webrtc_streamer(
	        key="object-detection",
	        mode=WebRtcMode.SENDRECV,
	        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
	        video_frame_callback=callback,
	        audio_frame_callback=process_audio,
	        media_stream_constraints={"video": True, "audio": False}, #
	        async_processing=True,
	    )


if __name__ == "__main__":
    main()
