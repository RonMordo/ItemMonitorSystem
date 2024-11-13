from picamera2 import Picamera2
import cv2

class Camera:
    def __init__(self, camera_id=0, resolution=(640, 480), frame_rate=30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.frame_rate = frame_rate

        # Initialize Picamera2 object
        self.picam2 = Picamera2()

        # Configure the camera
        camera_config = self.picam2.create_preview_configuration(main={"size": self.resolution})
        self.picam2.configure(camera_config)

        # Start the camera
        self.picam2.start()

    def capture_video_frame(self):
        # Capture an image as NumPy array
        frame = self.picam2.capture_array()
        return frame

    def stop(self):
        # Stop the camera
        self.picam2.stop()