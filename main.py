import cv2
import numpy as np
import av

import streamlit as st
from streamlit_webrtc import WebRtcMode, RTCConfiguration, webrtc_streamer

from face_detectors import Ultralight320Detector
import mediapipe as mp

FONT = cv2.FONT_HERSHEY_PLAIN
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

pose = mp.solutions.pose.Pose(model_complexity=0)
detector = Ultralight320Detector()

streaming_placeholder = st.empty()


def det_gaze_ratio(eye_points, facial_landmarks, height, width, gray):
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

    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ration = 1
    elif right_side_white == 0:
        gaze_ration = 5
    else:
        gaze_ration = left_side_white / right_side_white

    return gaze_ration


def get_gaze_direction(img, img_height, img_width):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bbox = detector.detect_faces_keypoints(gray, get_all=True)
    landmarks = bbox[0]['keypoints']

    gaze_ration_left_eye = det_gaze_ratio(LEFT_EYE_POINTS, landmarks, img_height, img_width, gray)
    gaze_ration_right_eye = det_gaze_ratio(RIGHT_EYE_POINTS, landmarks, img_height, img_width, gray)
    gaze_ratio = (gaze_ration_right_eye + gaze_ration_left_eye) / 2

    if gaze_ratio <= 0.85:
        gaze_direction = "LEFT"
    elif 0.85 < gaze_ratio < 1.6:
        gaze_direction = "CENTER"
    else:
        gaze_direction = "RIGHT"

    return gaze_direction


def get_coordinates(number, body, img_height, img_width):
    x = int(body.pose_landmarks.landmark[number].x * img_width)
    y = int(body.pose_landmarks.landmark[number].y * img_height)

    return x, y


def callback(frame: av.VideoFrame) -> av.VideoFrame:

    img_rgb = frame.to_ndarray(format="rgb24")

    image_height, image_width, _ = img_rgb.shape

    cv2.putText(img_rgb, get_gaze_direction(img_rgb,  image_height, image_width), (50, 100), FONT, 2, (255, 0, 0), 3)

    body = pose.process(img_rgb)

    if body.pose_landmarks:

        # shoulders
        p11 = get_coordinates(11, body, image_height, image_width)
        cv2.circle(img_rgb, p11, 3, BLUE, 2)
        p12 = get_coordinates(12, body, image_height, image_width)
        cv2.circle(img_rgb, p12, 3, BLUE, 2)

        cv2.line(img_rgb, p12, p11, WHITE, 1)

        # right hand
        p13 = get_coordinates(13, body, image_height, image_width)
        cv2.circle(img_rgb, p13, 3, BLUE, 2)
        p15 = get_coordinates(15, body, image_height, image_width)
        cv2.circle(img_rgb, p15, 3, BLUE, 2)
        p17 = get_coordinates(17, body, image_height, image_width)
        cv2.circle(img_rgb, p17, 3, BLUE, 2)
        p19 = get_coordinates(19, body, image_height, image_width)
        cv2.circle(img_rgb, p19, 3, BLUE, 2)
        p21 = get_coordinates(21, body, image_height, image_width)
        cv2.circle(img_rgb, p21, 3, BLUE, 2)

        cv2.line(img_rgb, p11, p13, WHITE, 1)
        cv2.line(img_rgb, p13, p15, WHITE, 1)
        cv2.line(img_rgb, p15, p21, WHITE, 1)
        cv2.line(img_rgb, p15, p17, WHITE, 1)
        cv2.line(img_rgb, p15, p19, WHITE, 1)

        # left hand
        p14 = get_coordinates(14, body, image_height, image_width)
        cv2.circle(img_rgb, p14, 3, BLUE, 2)
        p16 = get_coordinates(16, body, image_height, image_width)
        cv2.circle(img_rgb, p16, 3, BLUE, 2)
        p18 = get_coordinates(18, body, image_height, image_width)
        cv2.circle(img_rgb, p18, 3, BLUE, 2)
        p20 = get_coordinates(20, body, image_height, image_width)
        cv2.circle(img_rgb, p20, 3, BLUE, 2)
        p22 = get_coordinates(22, body, image_height, image_width)
        cv2.circle(img_rgb, p22, 3, BLUE, 2)

        cv2.line(img_rgb, p12, p14, WHITE, 1)
        cv2.line(img_rgb, p14, p16, WHITE, 1)
        cv2.line(img_rgb, p16, p22, WHITE, 1)
        cv2.line(img_rgb, p16, p20, WHITE, 1)
        cv2.line(img_rgb, p16, p18, WHITE, 1)

    return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")


with streaming_placeholder.container():
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
