import datetime
from enum import Enum
import queue as Q
import av
from video_file_writer import VideoFileWriter


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