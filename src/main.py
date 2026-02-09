import os
import cv2
import torch
from ultralytics import YOLO

from utils import get_files, get_video_files
from config import Config
from processors.video_processor import VideoProcessor

MODE = "VIDEO"  # Options: "IMAGE", "VIDEO", "REALTIME"


def process_image_mode():
    """Process images with YOLO detection (legacy mode)."""
    print("\n=== IMAGE MODE ===\n")

    model = YOLO(str(Config.MODEL_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_path = "images"
    output_path = str(Config.OUTPUT_DIR)
    Config.ensure_output_dir()

    images = get_files(image_path)
    print(f"Processing {len(images)} images from '{image_path}'")

    for img_path in images:
        results = model(img_path, classes=Config.CLASSES, conf=Config.CONFIDENCE_THRESHOLD)

        for result in results:
            annotated_img = result.plot()
            output_file = f"{output_path}/annotated_{os.path.basename(img_path)}"
            cv2.imwrite(output_file, annotated_img)
            print(f"Saved: {output_file}")

    print(f"\nImage processing complete. Output saved to '{output_path}'\n")


def process_video_mode():
    """Process videos with object tracking and counting."""
    print("\n=== VIDEO MODE (with Tracking & Counting) ===\n")

    # Initialize video processor
    processor = VideoProcessor()

    video_path = "videos"
    output_path = str(Config.OUTPUT_DIR)

    video_files = get_video_files(video_path)

    if not video_files:
        print(f"No video files found in '{video_path}'")
        return

    print(f"Found {len(video_files)} video file(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(vf)}")

    # Ask user to select video
    if len(video_files) == 1:
        selected_video = video_files[0]
        print(f"\nProcessing: {os.path.basename(selected_video)}")
    else:
        print("\nEnter video number to process (or 'all' for all videos): ", end="")
        try:
            user_input = input().strip()

            if user_input.lower() == 'all':
                print("\nProcessing all videos...")
                processor.process_multiple_videos(
                    video_files,
                    display=True,
                    output_dir=output_path
                )
                return
            else:
                video_idx = int(user_input) - 1
                if 0 <= video_idx < len(video_files):
                    selected_video = video_files[video_idx]
                else:
                    print("Invalid selection. Processing first video.")
                    selected_video = video_files[0]
        except (ValueError, EOFError):
            print("Invalid input. Processing first video.")
            selected_video = video_files[0]

    # Process selected video
    print(f"\nProcessing: {os.path.basename(selected_video)}")
    print("Press 'q' to quit, 'p' to pause\n")

    output_file = os.path.join(output_path, f"annotated_{os.path.basename(selected_video)}")

    processor.process_video(
        selected_video,
        display=True,
        output_path=output_file
    )


def process_realtime_mode():
    """Process real-time video stream (legacy mode)."""
    print("\n=== REALTIME MODE ===\n")

    model = YOLO(str(Config.MODEL_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    stream_url = "http://192.168.0.53:4747/video"
    print(f"Connecting to stream: {stream_url}")

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream at {stream_url}")
        return

    print("Stream connected. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or connection lost.")
            break

        results = model(frame, classes=Config.CLASSES, conf=Config.CONFIDENCE_THRESHOLD)

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv12 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nRealtime processing stopped.\n")


def main():
    """Main entry point for application."""
    print("="*60)
    print("YOLO Object Detection & Tracking System")
    print("="*60)
    print(f"Mode: {MODE}")
    print(f"Model: {Config.MODEL_PATH.name}")
    print(f"Device: {Config.DEVICE}")
    print(f"Classes: {Config.get_class_names()}")
    print(f"Confidence: {Config.CONFIDENCE_THRESHOLD}")
    print("="*60)

    if MODE == "IMAGE":
        process_image_mode()
    elif MODE == "VIDEO":
        process_video_mode()
    elif MODE == "REALTIME":
        process_realtime_mode()
    else:
        print(f"Error: Unknown mode '{MODE}'")
        print("Valid modes: IMAGE, VIDEO, REALTIME")


if __name__ == "__main__":
    main()
