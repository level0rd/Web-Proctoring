import datetime
from enum import Enum
import queue as Q
import av
import threading
import cv2


class Violation(Enum):
    SECOND_PERSON = "Second_person"
    LEAVING_PERSON = "Leaving_person"
    HANDS_RAISED = "Hands_raised"
    GAZE_AVERSION = "Gaze_aversion"
    VOICE_DETECTED = "Voice_detected"


class Exam:

    def __init__(self):
        self.violation = False
        self.violation_fixed = False
        self.frame_queue = Q.Queue()
        self.should_stop_writing = False
        self.video_writer = None
        self.no_violation_counter = 0
        self.violation_type = None

    def generate_filename(self):
        """
        Generate filename for the recorded video based on the type of detected violation.

        Returns:
            str: Generated filename.
        """
        now = datetime.datetime.now()
        return now.strftime(f"Captured/%H_%M_%S_%d_%m_%Y_{self.violation_type}.mp4")


    def record(self, copied_frame: av.VideoFrame):
        """
        Record the exam session, filtering out false positives
        and starting recording only if the violation lasts at least 10 frames.

        Args:
            copied_frame: Copied frame from the exam video stream.
        """
        if self.violation:
            self.frame_queue.put(copied_frame)
            if not self.video_writer and self.frame_queue.qsize() >= 15:
                self.violation_fixed = True
                self.video_writer = VideoFileWriter(640, 480, self)
                self.video_writer.start_writing(self.generate_filename())
        elif self.violation_fixed and not self.violation and self.no_violation_counter < 10:
            self.no_violation_counter += 1
        else:
            self.frame_queue.queue.clear()

        if not self.violation and self.violation_fixed and self.no_violation_counter >= 10:
            self.should_stop_writing = True
            self.video_writer.close()
            self.video_writer = None
            self.should_stop_writing = False
            self.violation_fixed = False
            self.no_violation_counter = 0


class VideoFileWriter:
    def __init__(self, width: int, height: int, exam_state: Exam):
        """
        Initialize VideoFileWriter object.

        Args:
            width (int): Width of the video frames.
            height (int): Height of the video frames.
            exam_state (Exam): Exam object representing the current exam session.
        """
        self.width = width
        self.height = height
        self.exam_state = exam_state
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.out = None
        self.file_writer_task = None
        self.filename = None

    def start_writing(self, filename: str):
        """
        Start writing frames to the video file in a separate thread.

        Args:
            filename (str): Name of the output video file.
        """
        self.filename = filename
        self.out = cv2.VideoWriter(filename, self.fourcc, 24.0, (self.width, self.height))
        self.file_writer_task = threading.Thread(target=self.write_frames_to_file)
        self.file_writer_task.start()

    def write_frames_to_file(self):
        """
        Write frames to the video file.
        """
        while not self.exam_state.should_stop_writing:
            try:
                frame = self.exam_state.frame_queue.get(timeout=1)
                img_bgr = frame.to_ndarray(format="bgr24")
                if self.out is None:
                    self.start_writing(self.filename)
                self.out.write(img_bgr)
                self.exam_state.frame_queue.task_done()
            except Q.Empty:
                pass

    def close(self):
        """
        Close the video file writer.
        """
        if self.out is not None:
            self.out.release()
        if self.file_writer_task is not None:
            self.file_writer_task.join()
