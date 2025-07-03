import cv2
import numpy as np
from typing import Dict, List, Tuple, TypedDict

class Box(TypedDict):
    x: int
    y: int
    width: int
    height: int


class BoxComponent(TypedDict):
    id: int
    type: str
    bbox: Box
    area: int


class SegResult(TypedDict):
    steps: Dict[str, np.ndarray]
    components: List[BoxComponent]
    total_components: int

# --- Unified Segmenter Class ---
class UnifiedSegmenter:
    def __init__(self):
        # Default parameters
        self.min_component_area = 500
        self.max_component_area = 1_000_000
        self.merge_threshold = 20
        self.group_x = 40
        self.group_y = 30
        self.table_aspect_ratio_threshold = 2.0
        self.blur_kernel = 3
        self.adaptive_block_size = 11
        self.adaptive_c = 2
        self.adaptive_method = "gaussian"  # 'mean' or 'gaussian'
        self.morph_op = "close"  # 'dilate', 'erode', 'open', 'close'
        self.morph_kernel = 3
        self.morph_iter = 2
        self.color_ranges = [
            ([40, 50, 50], [80, 255, 255]),  # green
            ([100, 50, 50], [130, 255, 255]),  # blue
            ([0, 0, 100], [180, 30, 200]),  # gray/white
        ]
        self.min_aspect_ratio = 0.2
        self.max_aspect_ratio = 30.0

    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        steps = {}
        # CLAHE
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        steps["clahe"] = enhanced.copy()
        # Blur
        if self.blur_kernel > 1:
            blurred = cv2.GaussianBlur(
                enhanced, (self.blur_kernel, self.blur_kernel), 0
            )
        else:
            blurred = enhanced.copy()
        steps["blurred"] = blurred.copy()
        # Adaptive Threshold
        block_size = self.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        if self.adaptive_method == "mean":
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.adaptive_c,
            )
        else:
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.adaptive_c,
            )
        steps["adaptive_thresh"] = thresh.copy()
        return steps

    def edge_and_morph(
        self, image: np.ndarray, thresh: np.ndarray
    ) -> Dict[str, np.ndarray]:
        steps = {}
        # Canny Edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        steps["canny"] = edges.copy()
        # Morphological Operation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel)
        )
        op_map = {
            "dilate": cv2.dilate,
            "erode": cv2.erode,
            "open": lambda img, k, iterations=1: cv2.morphologyEx(
                img, cv2.MORPH_OPEN, k, iterations=iterations
            ),
            "close": lambda img, k, iterations=1: cv2.morphologyEx(
                img, cv2.MORPH_CLOSE, k, iterations=iterations
            ),
        }
        morph = op_map[self.morph_op](
            thresh, kernel, iterations=self.morph_iter)
        steps["morph"] = morph.copy()
        # Combine edges and morph
        combined = cv2.bitwise_or(morph, edges)
        steps["combined"] = combined.copy()

        edge_boxes = self.detect_edge_boxes(steps["combined"])
        edge_components = self.box2component(edge_boxes, steps["combined"])
        vis_box_edge = self.draw_segmentation(
            np.zeros_like(steps["combined"]), edge_components
        )

        steps["boxed_combined"] = vis_box_edge.copy()

        return steps

    def detect_color_regions(
        self, image: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int, int, int]]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        all_regions = []
        for lower, upper in self.color_ranges:
            lower, upper = np.array(lower, dtype=np.uint8), np.array(
                upper, dtype=np.uint8
            )
            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel)
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iter
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iter
            )
            mask_total = cv2.bitwise_or(mask_total, mask)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_component_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 10:
                        all_regions.append((x, y, w, h))

        steps = {}
        color_components = self.box2component(all_regions, image)
        vis_box_color = self.draw_segmentation(
            np.zeros_like(image), color_components)

        steps["color_mask"] = mask_total
        steps["boxed_color"] = vis_box_color.copy()

        return steps, all_regions

    def detect_edge_boxes(self, im: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(
            im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_boxes = [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) > self.min_component_area
        ]
        return edge_boxes

    def find_components(
        self,
        all_boxes: List[Tuple[int, int, int, int]],
        image_shape,
    ) -> List[Tuple[int, int, int, int]]:
        # Filter by area and aspect ratio
        filtered = []
        for x, y, w, h in all_boxes:
            area = w * h
            aspect = w / h if h > 0 else 0
            if (
                self.min_component_area <= area <= self.max_component_area
                and self.min_aspect_ratio <= aspect <= self.max_aspect_ratio
                and w > 10
                and h > 10
                and x >= 0
                and y >= 0
                and x + w <= image_shape[1]
                and y + h <= image_shape[0]
            ):
                filtered.append((x, y, w, h))
        return self.merge_boxes(filtered)

    def merge_boxes(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []
        merged = [box for box in boxes]
        changed = True
        while changed:
            changed = False
            new_merged = []
            skip = set()
            for i, box1 in enumerate(merged):
                if i in skip:
                    continue
                x1, y1, w1, h1 = box1
                merged_this = False
                for j, box2 in enumerate(merged):
                    if i >= j or j in skip:
                        continue
                    x2, y2, w2, h2 = box2
                    # Check if one box is inside the other
                    box1_inside_box2 = (x1 >= x2 and y1 >= y2 and 
                                       x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2)
                    box2_inside_box1 = (x2 >= x1 and y2 >= y1 and 
                                       x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1)
                    
                    # Single merge threshold or one box inside another
                    if (
                        abs(x1 - x2) < self.merge_threshold
                        and abs(y1 - y2) < self.merge_threshold
                    ) or (abs(x1 - x2) < self.group_x and abs(y1 - y2) < self.group_y) or box1_inside_box2 or box2_inside_box1:
                        nx1, ny1 = min(x1, x2), min(y1, y2)
                        nx2, ny2 = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
                        new_merged.append((nx1, ny1, nx2 - nx1, ny2 - ny1))
                        skip.add(j)
                        merged_this = True
                        changed = True
                        break
                if not merged_this:
                    new_merged.append(box1)
            merged = new_merged
        return merged

    def classify_component(
        self, box: Tuple[int, int, int, int], image: np.ndarray
    ) -> str:
        x, y, w, h = box
        aspect = w / h if h > 0 else 0
        if aspect > self.table_aspect_ratio_threshold and h > 100:
            return "table"
        elif aspect > 3 and h < 50:
            return "toolbar"
        elif w > 200 and h > 150:
            return "form_panel"
        elif aspect < 0.5 and w < 100:
            return "sidebar"
        elif h < 100 and w > 100:
            return "input_group"
        else:
            return "component"

    def segment(self, image: np.ndarray) -> SegResult:
        steps = {}
        steps_pre = self.preprocess(image)
        steps.update(steps_pre)
        steps_edge = self.edge_and_morph(
            steps_pre["clahe"], steps_pre["adaptive_thresh"]
        )
        steps.update(steps_edge)
        steps_color, color_boxes = self.detect_color_regions(image)
        steps.update(steps_color)

        edge_boxes = self.detect_edge_boxes(steps_edge["combined"])
        all_boxes = edge_boxes + color_boxes
        boxes = self.find_components(all_boxes, image.shape)

        components = self.box2component(boxes, image)
        components.sort(
            key=lambda c: (-c["area"], c["bbox"]["y"], c["bbox"]["x"]))

        return {
            "steps": steps,
            "components": components,
            "total_components": len(components),
        }

    def box2component(
        self, boxes: List[Tuple[int, int, int, int]], image: np.ndarray
    ) -> List[BoxComponent]:
        # Compose metadata
        components: List[BoxComponent] = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            comp_type = self.classify_component(box, image)
            components.append(
                {
                    "id": i,
                    "type": comp_type,
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                    },
                    "area": int(w * h),
                }
            )
        return components

    def draw_segmentation(
        self, image: np.ndarray, components: List[BoxComponent]
    ) -> np.ndarray:
        vis = image.copy()
        is_gray = len(vis.shape) == 2
        type_colors = {
            "table": (0, 255, 0),
            "toolbar": (0, 125, 125),
            "form_panel": (0, 0, 255),
            "sidebar": (255, 255, 0),
            "input_group": (255, 0, 255),
            "component": (0, 255, 255),
        }
        for comp in components:
            bbox = comp["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            if is_gray:
                color = 255
            else:
                color = type_colors.get(comp["type"], (128, 128, 128))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label = f"{comp['id']}: {comp['type']}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y - 10 if y - 10 > 10 else y + 20
            cv2.rectangle(
                vis,
                (x, label_y - label_size[1] - 5),
                (x + label_size[0] + 5, label_y + 5),
                color,
                -1,
            )
            cv2.putText(
                vis,
                label,
                (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return vis
    
    
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    segmenter = UnifiedSegmenter()
    input_image_rgb = cv2.imread(image_path)    
    image_bgr = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)
    seg_result = segmenter.segment(image_bgr)
    vis_image_bgr = segmenter.draw_segmentation(
        image_bgr, seg_result["components"])
    vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_vis.png", vis_image_rgb)