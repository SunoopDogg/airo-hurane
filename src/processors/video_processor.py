import cv2
import time
from pathlib import Path
from typing import Optional
from config import Config
from tracking.object_tracker import ObjectTracker
from visualization.frame_renderer import FrameRenderer


class VideoProcessor:
    """
    Processes video files with object tracking and real-time visualization.
    Integrates tracking and rendering modules for complete video processing pipeline.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize video processor.

        Args:
            model_path: Path to YOLO model weights (default: from config)
        """
        self.tracker = ObjectTracker(model_path)
        self.renderer = FrameRenderer()
        self.config = Config

    def process_video(
        self,
        video_path: str,
        display: bool = True,
        output_path: Optional[str] = None,
        skip_frames: int = None
    ) -> dict:
        """
        Process a video file with object tracking and visualization.

        Args:
            video_path: Path to input video file
            display: Whether to display video in real-time (default: True)
            output_path: Optional path to save annotated video
            skip_frames: Number of frames to skip (0 = process all)

        Returns:
            Dictionary with processing statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nProcessing video: {video_path.name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.2f}s\n")

        # Setup video writer if output path is provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

        # Reset tracker for new video
        self.tracker.reset()

        # Processing variables
        skip_frames = skip_frames if skip_frames is not None else Config.SKIP_FRAMES
        frame_count = 0
        processed_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames if configured
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue

                # Track objects in frame
                tracked_objects, current_count, total_unique_count = \
                    self.tracker.track_frame(frame)

                # Calculate real-time FPS
                elapsed_time = time.time() - start_time
                processing_fps = processed_count / elapsed_time if elapsed_time > 0 else 0

                # Render frame with tracking results
                annotated_frame = self.renderer.render_frame(
                    frame,
                    tracked_objects,
                    current_count,
                    total_unique_count,
                    processing_fps
                )

                # Save to output video if configured
                if video_writer:
                    video_writer.write(annotated_frame)

                # Display frame if enabled
                if display:
                    # Resize for display if configured
                    display_frame = annotated_frame
                    if Config.DISPLAY_WIDTH and width != Config.DISPLAY_WIDTH:
                        display_height = int(height * Config.DISPLAY_WIDTH / width)
                        display_frame = cv2.resize(
                            annotated_frame,
                            (Config.DISPLAY_WIDTH, display_height)
                        )

                    cv2.imshow(Config.DISPLAY_WINDOW_NAME, display_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nProcessing interrupted by user")
                        break
                    elif key == ord('p'):  # Pause
                        print("Paused - Press any key to continue")
                        cv2.waitKey(0)

                processed_count += 1

                # Print progress
                if processed_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | "
                          f"Frame: {frame_count}/{total_frames} | "
                          f"Current: {current_count} | "
                          f"Total Unique: {total_unique_count} | "
                          f"FPS: {processing_fps:.1f}")

        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()

        # Get final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        stats = self.tracker.get_statistics()

        # Print summary
        print(f"\n{'='*50}")
        print(f"Processing Complete")
        print(f"{'='*50}")
        print(f"Total frames processed: {processed_count}")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Average FPS: {processed_count/processing_time:.2f}")
        print(f"Total unique persons detected: {stats['total_unique']}")
        print(f"Unique IDs: {stats['unique_ids']}")
        print(f"{'='*50}\n")

        return {
            'video_path': str(video_path),
            'frames_processed': processed_count,
            'processing_time': processing_time,
            'average_fps': processed_count / processing_time,
            'statistics': stats
        }

    def process_multiple_videos(
        self,
        video_paths: list,
        display: bool = True,
        output_dir: Optional[str] = None
    ) -> list:
        """
        Process multiple video files.

        Args:
            video_paths: List of video file paths
            display: Whether to display videos in real-time
            output_dir: Optional directory to save annotated videos

        Returns:
            List of processing statistics for each video
        """
        results = []

        for i, video_path in enumerate(video_paths, 1):
            print(f"\n{'='*50}")
            print(f"Processing video {i}/{len(video_paths)}")
            print(f"{'='*50}")

            output_path = None
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                video_name = Path(video_path).stem
                output_path = str(output_dir_path / f"tracked_{video_name}.mp4")

            result = self.process_video(
                video_path,
                display=display,
                output_path=output_path
            )
            results.append(result)

        return results
