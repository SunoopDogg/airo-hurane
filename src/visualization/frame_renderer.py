"""
Frame rendering module for visualizing tracking results.
Draws bounding boxes, IDs, and count statistics on video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple
from config import Config


class FrameRenderer:
    """
    Renders tracking results on video frames.
    Displays bounding boxes, object IDs, and count statistics.
    """

    def __init__(self):
        """Initialize frame renderer with configuration."""
        self.box_color = Config.BOX_COLOR
        self.text_color = Config.TEXT_COLOR
        self.text_bg_color = Config.TEXT_BG_COLOR
        self.panel_bg_color = Config.PANEL_BG_COLOR
        self.panel_text_color = Config.PANEL_TEXT_COLOR

        self.box_thickness = Config.BOX_THICKNESS
        self.font_scale = Config.FONT_SCALE
        self.font_thickness = Config.FONT_THICKNESS
        self.text_padding = Config.TEXT_PADDING

    def render_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[dict],
        current_count: int,
        total_unique_count: int,
        fps: float = None
    ) -> np.ndarray:
        """
        Render tracking results on a frame.

        Args:
            frame: Input frame (numpy array)
            tracked_objects: List of tracked objects with IDs and bboxes
            current_count: Number of objects in current frame
            total_unique_count: Total unique objects seen
            fps: Optional FPS value to display

        Returns:
            Annotated frame with visualizations
        """
        # Create a copy to avoid modifying original
        annotated_frame = frame.copy()

        # Draw bounding boxes and IDs for each tracked object
        for obj in tracked_objects:
            track_id = obj['id']
            bbox = obj['bbox']
            confidence = obj['confidence']

            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                self.box_color,
                self.box_thickness
            )

            # # Prepare label text
            # label = f"ID:{track_id} ({confidence:.2f})"

            # # Calculate text size for background
            # (text_width, text_height), baseline = cv2.getTextSize(
            #     label,
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     self.font_scale,
            #     self.font_thickness
            # )

            # # Draw text background
            # cv2.rectangle(
            #     annotated_frame,
            #     (x1, y1 - text_height - baseline - 10),
            #     (x1 + text_width + 10, y1),
            #     self.text_bg_color,
            #     -1
            # )

            # # Draw label text
            # cv2.putText(
            #     annotated_frame,
            #     label,
            #     (x1 + 5, y1 - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     self.font_scale,
            #     self.text_color,
            #     self.font_thickness
            # )

        # Draw count statistics panel
        # annotated_frame = self._draw_statistics_panel(
        #     annotated_frame,
        #     current_count,
        #     total_unique_count,
        #     fps
        # )

        return annotated_frame

    def _draw_statistics_panel(
        self,
        frame: np.ndarray,
        current_count: int,
        total_unique_count: int,
        fps: float = None
    ) -> np.ndarray:
        """
        Draw statistics panel on top of the frame.

        Args:
            frame: Input frame
            current_count: Current frame object count
            total_unique_count: Total unique object count
            fps: Optional FPS value

        Returns:
            Frame with statistics panel
        """
        height, width = frame.shape[:2]

        # Panel dimensions
        panel_height = 80
        panel_width = width

        # Create semi-transparent panel overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (panel_width, panel_height),
            self.panel_bg_color,
            -1
        )

        # Blend overlay with original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Prepare statistics text
        stats_lines = [
            f"Current Frame: {current_count} person(s)",
            f"Total Unique: {total_unique_count} person(s)"
        ]

        if fps is not None and Config.FPS_DISPLAY:
            stats_lines.append(f"FPS: {fps:.1f}")

        # Draw statistics text
        y_offset = 25
        for line in stats_lines:
            cv2.putText(
                frame,
                line,
                (self.text_padding, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.panel_text_color,
                self.font_thickness
            )
            y_offset += 25

        return frame

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Draw a single bounding box with label.

        Args:
            frame: Input frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            label: Label text
            color: Box color (default: from config)

        Returns:
            Frame with bounding box
        """
        color = color or self.box_color
        x1, y1, x2, y2 = bbox

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        # Draw label
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.font_thickness
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            self.text_bg_color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.text_color,
            self.font_thickness
        )

        return frame
