import av
from openvino.runtime import Core
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from face_detectors import Ultralight320Detector

import pvcobra
from pvrecorder import PvRecorder

from exam import Exam, Violation
from movenet_utils import get_movenet_keypoints, second_person_detection, draw_pose
from gaze_direction import get_gaze_direction


POSE_MODEL = 'models/move_net_openvino.xml'

BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

MOVE_NET_THRESHOLD = 0.15


core = Core()
ov_model = core.read_model(POSE_MODEL)
pose_model = core.compile_model(ov_model)

face_detector = Ultralight320Detector()

cobra = pvcobra.create(access_key='your_access_key')
recorder = PvRecorder(frame_length=512, device_index=3)


exam = Exam()


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Analyze each frame using MoveNet and DLIB 68 landmarks neural networks.
    Detects the presence of a second person, the absence of the examinee,
    raised hands, and gaze direction.

    Args:
        frame (av.VideoFrame): Video frame to be analyzed.

    Returns:
        av.VideoFrame: Analyzed video frame.
    """
    img_rgb = frame.to_ndarray(format="rgb24")
    image_height, image_width, _ = img_rgb.shape

    keypoints = get_movenet_keypoints(img_rgb, pose_model)

    # Second person
    if second_person_detection(keypoints):
        exam.violation = True
        exam.violation_type = Violation.SECOND_PERSON.value
        draw_pose(1, keypoints, img_rgb, RED)
    else:
        exam.violation = False

    draw_pose(0, keypoints, img_rgb, WHITE)

    # The absence of the examinee, raised hands, and gaze direction.
    confidences_per_instance = keypoints[0, :, -1]
    if confidences_per_instance[0] >= MOVE_NET_THRESHOLD:

        exam.violation = False
        p10 = (round(keypoints[0][0][28] * image_width), round(keypoints[0][0][27] * image_height))
        p11 = (round(keypoints[0][0][31] * image_width), round(keypoints[0][0][30] * image_height))
        hand_border = (image_height / 100) * 85

        if p11[1] > hand_border or p10[1] > hand_border:
            exam.violation = False
        else:
            exam.violation = True
            exam.violation_type = Violation.HANDS_RAISED.value

        try:
            if get_gaze_direction(img_rgb, image_height, image_width, face_detector) != "CENTER":
                exam.violation = True
                exam.violation_type = Violation.GAZE_AVERSION.value
            else:
                exam.violation = False
        except IndexError:
            print("Error: list index out of range")

    else:
        exam.violation = True
        exam.violation_type = Violation.LEAVING_PERSON.value

    exam.record(av.VideoFrame.from_ndarray(img_rgb, format="rgb24"))

    return frame


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
            exam.violation = True
            exam.violation_type = Violation.VOICE_DETECTED.value
        else:
            exam.violation = False


def main():
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        audio_frame_callback=process_audio,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


if __name__ == "__main__":
    main()
