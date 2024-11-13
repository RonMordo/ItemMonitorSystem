from HumenDetection import DetectionModule
import logging
import signal


def main():
    detection_module = DetectionModule(
        input_path='/home/ron/Downloads/test_vid.mp4'
    )

    # Handle graceful shutdown on SIGINT and SIGTERM
    def signal_handler(sig, frame):
        logging.info("Received termination signal. Shutting down...")
        detection_module.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the Detection Module
    detection_module.start()

if __name__ == "__main__":
    main()
