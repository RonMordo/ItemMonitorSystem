import zmq
import cv2
import numpy as np
import os
from datetime import datetime
import subprocess
import threading
import sys
import logging
import signal

class DetectionModule:
    def __init__(self, input_path):
        """
        Initializes the Detection Module.

        Parameters:
        - env_script (str): Path to the environment setup script (e.g., setup_env.sh).
        - detection_script (str): Path to the Detection Application script (e.g., detection_application.py).
        - input_path (str): Path to the input video file.
        - output_dir (str): Directory to save annotated frames.
        - zmq_address (str): ZeroMQ address to subscribe to (e.g., tcp://localhost:5555).
        """
        self.env_script = '/home/ron/hailo-rpi5-examples/setup_env.sh'
        self.detection_script = '/home/ron/hailo-rpi5-examples/basic_pipelines/detection.py'
        self.input_path = input_path
        self.output_dir = '/home/ron/PycharmProjects/ItemMointorSystem/output'
        self.zmq_address = 'tcp://localhost:5555'
        self.process = None
        self.monitor_thread = None
        self.receiver_thread = None
        self.stop_event = threading.Event()

        # Set up logging
        logging.basicConfig(
            filename='detection_module.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def activate_environment(self):
        """
        Sources the environment setup script to activate the virtual environment.
        """
        command = f"source {self.env_script}"
        process = subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logging.error(f"Failed to source environment script: {stderr.decode().strip()}")
            raise RuntimeError(f"Environment activation failed: {stderr.decode().strip()}")
        else:
            logging.info("Virtual environment activated successfully.")

    def launch_detection_app(self):
        """
        Launches the Detection Application as a subprocess with the specified input.
        """
        command = f"source {self.env_script} && python {self.detection_script} --input '{self.input_path}'"
        self.process = subprocess.Popen(
            ['bash', '-c', command],
            preexec_fn=os.setsid  # To allow killing the whole process group
        )
        logging.info(f"Detection Application launched with input: {self.input_path}")

    def monitor_process_output(self):
        """
        Monitors the Detection Application's subprocess output and logs it.
        """
        try:
            for line in self.process.stdout:
                logging.info(f"Detection App: {line.strip()}")
        except Exception as e:
            logging.error(f"Error monitoring Detection Application output: {e}")

    def receive_and_save_frames(self):
        """
        Receives annotated frames from the Detection Application via ZeroMQ and saves them.
        """
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        try:
            subscriber.connect(self.zmq_address)
            subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
            logging.info(f"Connected to ZeroMQ publisher at {self.zmq_address}")
        except Exception as e:
            logging.error(f"Failed to connect to ZeroMQ publisher: {e}")
            return

        frame_count = 0

        try:
            while not self.stop_event.is_set():
                try:
                    # Receive the JPEG-encoded frame with a timeout
                    if subscriber.poll(1000):  # Timeout in milliseconds
                        jpg_as_bytes = subscriber.recv()
                        # Decode the JPEG bytes to an image
                        np_arr = np.frombuffer(jpg_as_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            # Generate a unique filename using timestamp and frame count
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"person_detected_{timestamp}_{frame_count}.jpg"
                            filepath = os.path.join(self.output_dir, filename)

                            # Save the annotated frame
                            cv2.imwrite(filepath, frame)
                            logging.info(f"Saved annotated frame to {filepath}")

                            frame_count += 1

                            # Optional: Display the frame for verification
                            # cv2.imshow('Annotated Frame', frame)
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     self.stop()
                except zmq.Again:
                    # No message received within the timeout
                    continue
                except Exception as e:
                    logging.error(f"Error receiving or saving frame: {e}")
        finally:
            subscriber.close()
            context.term()
            # cv2.destroyAllWindows()
            logging.info("ZeroMQ subscriber closed.")

    def start(self):
        """
        Starts the Detection Module by launching the Detection Application and setting up IPC.
        """
        try:
            # Start receiving and saving frames in a separate thread
            self.receiver_thread = threading.Thread(target=self.receive_and_save_frames, daemon=True)
            self.receiver_thread.start()

            # Start monitoring the Detection Application's output in a separate thread
            self.monitor_thread = threading.Thread(target=self.monitor_process_output, daemon=True)
            self.monitor_thread.start()

            # Activate the virtual environment
            self.activate_environment()

            # Launch the Detection Application
            self.launch_detection_app()

            logging.info("Detection Module started successfully.")

            # Wait for the subprocess to finish
            self.process.wait()
            logging.info("Detection Application subprocess has terminated.")

        except Exception as e:
            logging.error(f"An error occurred in Detection Module: {e}")
            self.stop()

    def stop(self):
        """
        Stops the Detection Module by terminating subprocesses and threads.
        """
        self.stop_event.set()

        if self.process:
            try:
                # Terminate the whole process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
                logging.info("Detection Application subprocess terminated.")
            except Exception as e:
                logging.error(f"Failed to terminate Detection Application subprocess: {e}")

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)

        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1)

        logging.info("Detection Module stopped gracefully.")
        sys.exit(0)