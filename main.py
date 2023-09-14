import cv2
import time
import numpy as np

from face_detectors import Ultralight320Detector
import mediapipe as mp

FONT = cv2.FONT_HERSHEY_PLAIN
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

vid = cv2.VideoCapture("Good_no_voice.mp4")

pose = mp.solutions.pose.Pose(model_complexity=0)
detector = Ultralight320Detector()


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


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


def get_gaze_ratio(img, img_height, img_width):

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


pTime = 0

while True:
    rect, frame = vid.read()
    if not rect:
        break

    cTime = time.time()

    image_height, image_width, _ = frame.shape

    cv2.putText(frame, get_gaze_ratio(frame,  image_height, image_width), (50, 100), FONT, 2, (255, 0, 0), 3)

    body = pose.process(frame)

    if body.pose_landmarks:
        x11 = int(body.pose_landmarks.landmark[11].x * image_width)
        y11 = int(body.pose_landmarks.landmark[11].y * image_height)
        cv2.circle(frame, (x11, y11), 3, BLUE, 2)
        x12 = int(body.pose_landmarks.landmark[12].x * image_width)
        y12 = int(body.pose_landmarks.landmark[12].y * image_height)
        cv2.circle(frame, (x12, y12), 3, BLUE, 2)

        x13 = int(body.pose_landmarks.landmark[13].x * image_width)
        y13 = int(body.pose_landmarks.landmark[13].y * image_height)
        cv2.circle(frame, (x13, y13), 3, BLUE, 2)
        x15 = int(body.pose_landmarks.landmark[15].x * image_width)
        y15 = int(body.pose_landmarks.landmark[15].y * image_height)
        cv2.circle(frame, (x15, y15), 3, BLUE, 2)
        x17 = int(body.pose_landmarks.landmark[17].x * image_width)
        y17 = int(body.pose_landmarks.landmark[17].y * image_height)
        cv2.circle(frame, (x17, y17), 3, BLUE, 2)
        x19 = int(body.pose_landmarks.landmark[19].x * image_width)
        y19 = int(body.pose_landmarks.landmark[19].y * image_height)
        cv2.circle(frame, (x19, y19), 3, BLUE, 2)
        x21 = int(body.pose_landmarks.landmark[21].x * image_width)
        y21 = int(body.pose_landmarks.landmark[21].y * image_height)
        cv2.circle(frame, (x21, y21), 3, BLUE, 2)

        x14 = int(body.pose_landmarks.landmark[14].x * image_width)
        y14 = int(body.pose_landmarks.landmark[14].y * image_height)
        cv2.circle(frame, (x14, y14), 3, BLUE, 2)
        x16 = int(body.pose_landmarks.landmark[16].x * image_width)
        y16 = int(body.pose_landmarks.landmark[16].y * image_height)
        cv2.circle(frame, (x16, y16), 3, BLUE, 2)
        x18 = int(body.pose_landmarks.landmark[18].x * image_width)
        y18 = int(body.pose_landmarks.landmark[18].y * image_height)
        cv2.circle(frame, (x18, y18), 3, BLUE, 2)
        x20 = int(body.pose_landmarks.landmark[20].x * image_width)
        y20 = int(body.pose_landmarks.landmark[20].y * image_height)
        cv2.circle(frame, (x20, y20), 3, BLUE, 2)
        x22 = int(body.pose_landmarks.landmark[22].x * image_width)
        y22 = int(body.pose_landmarks.landmark[22].y * image_height)
        cv2.circle(frame, (x22, y22), 3, BLUE, 2)

        cv2.line(frame, (x12, y12), (x11, y11), WHITE, 1)

        cv2.line(frame, (x11, y11), (x13, y13), WHITE, 1)
        cv2.line(frame, (x13, y13), (x15, y15), WHITE, 1)
        cv2.line(frame, (x15, y15), (x21, y21), WHITE, 1)
        cv2.line(frame, (x15, y15), (x17, y17), WHITE, 1)
        cv2.line(frame, (x15, y15), (x19, y19), WHITE, 1)

        cv2.line(frame, (x12, y12), (x14, y14), WHITE, 1)
        cv2.line(frame, (x14, y14), (x16, y16), WHITE, 1)
        cv2.line(frame, (x16, y16), (x22, y22), WHITE, 1)
        cv2.line(frame, (x16, y16), (x20, y20), WHITE, 1)
        cv2.line(frame, (x16, y16), (x18, y18), WHITE, 1)

    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (70, 50), FONT, 3, BLUE, 3)

    cv2.imshow("Pose detection and gaze direction", frame)

    cv2.waitKey(1)

