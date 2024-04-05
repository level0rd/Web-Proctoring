import cv2
import threading
import queue as Q
from exam import Exam

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
