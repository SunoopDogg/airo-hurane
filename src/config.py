import torch
from pathlib import Path


class Config:
    """Centralized configuration for video processing and object detection."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PATH = PROJECT_ROOT / "yolo12m.pt"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.25
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    CLASSES = [0]  # 0 = person in COCO dataset

    # Tracking settings (BoT-SORT)
    TRACKER_TYPE = "botsort.yaml"
    TRACK_BUFFER = 30  # Number of frames to keep lost tracks
    MATCH_THRESH = 0.8  # Matching threshold for BoT-SORT

    # Counting settings
    COUNT_MODE = "dual"  # "current", "cumulative", or "dual"

    # Visualization settings
    BOX_THICKNESS = 2
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    TEXT_PADDING = 10

    # Colors (BGR format for OpenCV)
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_BG_COLOR = (0, 0, 0)  # Black
    PANEL_BG_COLOR = (50, 50, 50)  # Dark gray
    PANEL_TEXT_COLOR = (255, 255, 255)  # White

    # Display settings
    DISPLAY_WINDOW_NAME = "YOLO Video Tracking"
    DISPLAY_WIDTH = 1280  # Resize width for display (None for original)
    FPS_DISPLAY = True

    # Video processing settings
    SKIP_FRAMES = 0  # Number of frames to skip (0 = process all frames)

    @classmethod
    def ensure_output_dir(cls):
        """Ensure output directory exists."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_class_names(cls):
        """Get human-readable class names."""
        coco_classes = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light"
            # Add more as needed
        }
        return [coco_classes.get(c, f"class_{c}") for c in cls.CLASSES]
