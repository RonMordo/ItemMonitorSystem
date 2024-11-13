import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import zmq
from datetime import datetime
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example

    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    """if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        print("Inside get numpy from buffer.")
        frame = get_numpy_from_buffer(buffer, format, width, height)"""
    try:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        print("Frame extraction succeeded.", flush=True)
    except Exception as e:
        print(f"Error in get_numpy_from_buffer: {e}", flush=True)
        frame = None

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)

    # Set up ZMQ publisher
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    # Test prints
    print(f"ROI: {roi}, Frame available: {frame is not None}", flush=True)
    if buffer is None:
        print("Empty buffer", flush=true)
    print(f"Format: {format}, Width: {width}, Height: {height}", flush=True)
    if format is None or width is None or height is None:
        print("Invalid format or dimensions from pad", flush=True)
    print(f"Use frame: {user_data.use_frame is not None}", flush=True)
    
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
            detection_count += 1
            x1, y1, x2, y2 = int(bbox.xmin() * width), int(bbox.ymin() * height), int(bbox.xmax() * width), int(bbox.ymax() * height)
            print(f" {x1, x2, y1, y2}", flush=True)
            print(f" {frame.shape}", flush=True)
            if (x2-x1) < 100 or (y2-y1) < 120:
                continue
            # Crop the frame to the bounding box region
            cropped_frame = frame[y1:y2, x1:x2]
            if cropped_frame.size == 0:
                print("Error with cropped frame", flush=True)

            # Encode cropped frame as JPEG and send over ZMQ
            _, jpg_encoded_frame = cv2.imencode('.png', cropped_frame)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sending frame over ZMQ")
            publisher.send(jpg_encoded_frame.tobytes())
    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
