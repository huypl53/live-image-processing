from ultralytics import YOLO
from typing import TypedDict, List
import cv2
from image_analysis.models import Box, BoxComponent, SegResult
import numpy as np
from image_analysis.config import load_segmenter_config

class YOLOBoxDetector:
    def __init__(self, device: str = "cpu"):
        """
        Initialize YOLOv8 model.

        Args:
            model_path (str): Path to YOLO model weights.
            device (str): Device to use ("cuda" or "cpu").
        """
        self.config = load_segmenter_config()
        model_path = self.config.get("model_path", "package/image_analysis/src/image_analysis/weights/040825_seg_s.pt")
        self.model = YOLO(model_path)
        self.device = device

    def segment(self, image: np.ndarray) -> SegResult:
        """
        Run YOLO inference on an image and return formatted results.

        Args:
            image_path (str): Path to image file.

        Returns:
            List[BoxComponent]: List of detection results in custom format.
        """
        results = self.model(image, device=self.device, conf = 0.6)[0]

        components: List[BoxComponent] = []

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls.item())
            class_name = self.model.names[cls_id]

            # xyxy format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            area = w * h

            component: BoxComponent = {
                "id": i,
                "type": class_name,
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": w,
                    "height": h
                },
                "area": area
            }

            components.append(component)

        return {
            "steps": {},
            "components": components,
            "total_components": len(components),
        }

