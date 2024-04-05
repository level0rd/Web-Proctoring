import cv2
import tensorflow as tf
import numpy as np
import math
from typing import Tuple

LINE_THICKNESS = 2
INPUT_SIZE = 256
MOVE_NET_THRESHOLD = 0.15
BLUE = (0, 0, 255)


def get_movenet_keypoints(image_rgb: np.ndarray, pose_model) -> np.ndarray:
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

    resized_image, image_shape = keep_aspect_ratio_resizer(image, INPUT_SIZE)
    input_tensor = tf.cast(resized_image, dtype=tf.uint8)

    return pose_model(input_tensor)[0]


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


def draw_pose(i: int, keypoints: np.ndarray,  image_rgb: np.ndarray, lines_color: tuple):
    """
    Draw a pose based on keypoints on the input image.

    Args:
        i (int): Index of the pose to draw.
        keypoints (np.ndarray): Array of keypoints.
        image_rgb (np.ndarray): RGB image to draw the pose on.
        lines_color (tuple): Color of the lines for drawing the pose.

    Returns:
        None
    """
    image_height, image_width, _ = image_rgb.shape

    p6 = (round(keypoints[0][i][16] * image_width), round(keypoints[0][i][15] * image_height))
    p7 = (round(keypoints[0][i][19] * image_width), round(keypoints[0][i][18] * image_height))

    p8 = (round(keypoints[0][i][22] * image_width), round(keypoints[0][i][21] * image_height))
    p9 = (round(keypoints[0][i][25] * image_width), round(keypoints[0][i][24] * image_height))

    p10 = (round(keypoints[0][i][28] * image_width), round(keypoints[0][i][27] * image_height))
    p11 = (round(keypoints[0][i][31] * image_width), round(keypoints[0][i][30] * image_height))

    points = [p6, p7, p8, p9, p10, p11]
    for i, coord in enumerate(points):
        cv2.circle(image_rgb, coord, 5, BLUE, -1)

    cv2.line(image_rgb, p6, p7, lines_color, LINE_THICKNESS)
    cv2.line(image_rgb, p6, p8, lines_color, LINE_THICKNESS)  # left hand
    cv2.line(image_rgb, p7, p9, lines_color, LINE_THICKNESS)  # right hand
    cv2.line(image_rgb, p8, p10, lines_color, LINE_THICKNESS)  # left hand
    cv2.line(image_rgb, p9, p11, lines_color, LINE_THICKNESS)  # right hand
