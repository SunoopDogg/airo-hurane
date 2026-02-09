from typing import Dict, Set, Tuple, List
import numpy as np
from ultralytics import YOLO
from config import Config


class ObjectTracker:
    """
    Tracks objects across video frames using BoT-SORT algorithm.
    Maintains statistics for current and cumulative object counts.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize object tracker with YOLO model.

        Args:
            model_path: Path to YOLO model weights (default: from config)
        """
        self.model_path = model_path or str(Config.MODEL_PATH)
        self.model = YOLO(self.model_path)

        # Tracking statistics
        self.unique_ids: Set[int] = set()
        self.current_frame_count = 0
        self.total_unique_count = 0

        # Frame counter
        self.frame_number = 0

    def track_frame(self, frame: np.ndarray) -> Tuple[List, int, int]:
        """
        Track objects in a single frame using BoT-SORT.

        Args:
            frame: Input frame (numpy array)

        Returns:
            Tuple of (tracking_results, current_count, total_unique_count)
            - tracking_results: List of tracked objects with IDs
            - current_count: Number of objects in current frame
            - total_unique_count: Total unique objects seen across all frames
        """
        self.frame_number += 1

        # Run YOLO tracking with BoT-SORT
        results = self.model.track(
            frame,
            persist=True,  # Persist tracks across frames
            tracker=Config.TRACKER_TYPE,
            conf=Config.CONFIDENCE_THRESHOLD,
            classes=Config.CLASSES,
            device=Config.DEVICE,
            verbose=False  # Suppress console output
        )

        # Extract tracking information
        tracked_objects = []
        current_ids = set()

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Extract box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                # Extract tracking ID
                track_id = int(boxes.id[i].cpu().numpy())

                # Extract confidence
                confidence = float(boxes.conf[i].cpu().numpy())

                # Extract class
                cls = int(boxes.cls[i].cpu().numpy())

                tracked_objects.append({
                    'id': track_id,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class': cls
                })

                current_ids.add(track_id)

        # Update statistics
        self.current_frame_count = len(current_ids)
        self.unique_ids.update(current_ids)
        self.total_unique_count = len(self.unique_ids)

        return tracked_objects, self.current_frame_count, self.total_unique_count

    def reset(self):
        """Reset tracking statistics (useful for processing new video)."""
        self.unique_ids.clear()
        self.current_frame_count = 0
        self.total_unique_count = 0
        self.frame_number = 0

    def get_statistics(self) -> Dict:
        """
        Get current tracking statistics.

        Returns:
            Dictionary containing tracking statistics
        """
        return {
            'frames_processed': self.frame_number,
            'current_count': self.current_frame_count,
            'total_unique': self.total_unique_count,
            'unique_ids': sorted(list(self.unique_ids))
        }
