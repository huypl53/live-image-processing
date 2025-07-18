import cv2
import numpy as np
from typing import Dict, List, Literal, Tuple, TypedDict

# Use matplotlib's colormap for a diverse color palette (BGR)
import matplotlib.pyplot as plt
import numpy as np

num_colors = 64
cmap = plt.get_cmap("tab20", num_colors)
palette = []
for i in range(num_colors):
    rgb = np.array(cmap(i)[:3]) * 255
    bgr = tuple(int(x) for x in rgb[::-1])
    palette.append(bgr)


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
        self.min_component_area = 100
        self.max_component_area = 1_000_000
        self.max_image_area_ratio = 0.95  # Maximum area as ratio of image area
        self.merge_threshold = 10
        self.group_x = 15
        self.group_y = 5
        self.table_aspect_ratio_threshold = 2.0
        self.blur_kernel = 3
        self.adaptive_block_size = 11
        self.adaptive_c = 2
        self.adaptive_method = "gaussian"  # 'mean' or 'gaussian'
        self.morph_op = "close"  # 'dilate', 'erode', 'open', 'close'
        self.morph_kernel = 5
        self.morph_iter = 1

        # Edge detection options
        self.use_canny = True
        self.use_morph = True
        self.canny_low = 50
        self.canny_high = 150
        self.canny_aperture = 3

        # Color detection options
        self.color_ranges = [
            ([40, 50, 50], [80, 255, 255]),  # green
            ([100, 50, 50], [130, 255, 255]),  # blue
            ([0, 0, 100], [180, 30, 200]),  # gray/white
        ]
        self.color_morph_kernel = 1
        self.color_morph_iter = 1
        self.use_color_close = True
        self.use_color_open = True

        # Box processing options
        self.enable_merge = True
        self.iou_threshold = 0.9  # IoU threshold for keep_detail strategy

        self.min_aspect_ratio = 0.05
        self.max_aspect_ratio = 70.0

        self.merge_strategy: Literal["ignore_detail", "keep_detail"] | None = None
        self.nms_iou_threshold = 0.8  # High threshold for NMS after keep-detail merging

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
        combined = np.zeros_like(thresh)

        # Canny Edges (optional)
        if self.use_canny:
            edges = cv2.Canny(
                image, self.canny_low, self.canny_high, apertureSize=self.canny_aperture
            )
            steps["canny"] = edges.copy()
            combined = cv2.bitwise_or(combined, edges)

        # Morphological Operation (optional)
        if self.use_morph:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.morph_kernel, 1)
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
            morph = op_map[self.morph_op](thresh, kernel, iterations=self.morph_iter)
            steps["morph"] = morph.copy()
            combined = cv2.bitwise_or(combined, morph)

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
        image_area = image.shape[0] * image.shape[1]
        max_allowed_area = image_area * self.max_image_area_ratio

        for lower, upper in self.color_ranges:
            lower, upper = np.array(lower, dtype=np.uint8), np.array(
                upper, dtype=np.uint8
            )
            mask = cv2.inRange(hsv, lower, upper)

            # Apply morphological operations if enabled
            if self.color_morph_kernel > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (self.color_morph_kernel, self.color_morph_kernel)
                )

                if self.use_color_close:
                    mask = cv2.morphologyEx(
                        mask, cv2.MORPH_CLOSE, kernel, iterations=self.color_morph_iter
                    )

                if self.use_color_open:
                    mask = cv2.morphologyEx(
                        mask, cv2.MORPH_OPEN, kernel, iterations=self.color_morph_iter
                    )

            mask_total = cv2.bitwise_or(mask_total, mask)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_component_area and area < max_allowed_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 10:
                        all_regions.append((x, y, w, h))

        steps = {}
        color_components = self.box2component(all_regions, image)
        vis_box_color = self.draw_segmentation(np.zeros_like(image), color_components)

        steps["color_mask"] = mask_total
        steps["boxed_color"] = vis_box_color.copy()

        return steps, all_regions

    def detect_edge_boxes(self, im: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = im.shape[0] * im.shape[1]
        max_allowed_area = image_area * self.max_image_area_ratio
        
        edge_boxes = []
        for c in contours:
            if cv2.contourArea(c) > self.min_component_area and cv2.contourArea(c) < max_allowed_area:
                x, y, w, h = cv2.boundingRect(c)
                edge_boxes.append((x, y, w, h))
        return edge_boxes

    def find_components(
        self,
        all_boxes: List[Tuple[int, int, int, int]],
        image_shape,
    ) -> List[Tuple[int, int, int, int]]:
        # Filter by area and aspect ratio
        filtered = []
        image_area = image_shape[0] * image_shape[1]
        max_allowed_area = image_area * self.max_image_area_ratio
        
        for x, y, w, h in all_boxes:
            area = w * h
            aspect = w / h if h > 0 else 0
            if (
                self.min_component_area <= area <= self.max_component_area
                and area < max_allowed_area  # Additional check for large boxes
                and self.min_aspect_ratio <= aspect <= self.max_aspect_ratio
                and w > 10
                and h > 10
                and x >= 0
                and y >= 0
                and x + w <= image_shape[1]
                and y + h <= image_shape[0]
            ):
                filtered.append((x, y, w, h))

        # Conditionally merge boxes
        if self.enable_merge:
            return self.merge_boxes(filtered)
        else:
            return filtered

    def calculate_iosa(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Smaller Area between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        smaller_area = min(box1_area, box2_area)

        return intersection_area / smaller_area if smaller_area > 0 else 0.0

    def is_smaller_box_inside_bigger(
        self,
        smaller_box: Tuple[int, int, int, int],
        bigger_box: Tuple[int, int, int, int],
    ) -> bool:
        """Check if smaller box is nearly inside bigger box using IoSA."""
        # Calculate IoSA (Intersection over Smaller Area)
        iosa = self.calculate_iosa(smaller_box, bigger_box)
        
        # IoSA threshold for containment detection
        return iosa >= self.iou_threshold  # You might want to rename this parameter

    def non_maximum_suppression(
        self, 
        boxes: List[Tuple[int, int, int, int]], 
        iou_threshold: float = 0.8
    ) -> List[Tuple[int, int, int, int]]:
        """
        Apply Non-Maximum Suppression to remove nearly identical overlapping boxes.
        
        Args:
            boxes: List of boxes in (x, y, w, h) format
            iou_threshold: High threshold for NMS since boxes should be nearly identical
        """
        if not boxes:
            return []
        
        # Convert to (x1, y1, x2, y2) format for easier processing
        boxes_xyxy = []
        for x, y, w, h in boxes:
            boxes_xyxy.append((x, y, x + w, y + h))
        
        # Sort by area (largest first) - keep larger boxes
        areas = [(i, (x2 - x1) * (y2 - y1)) for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy)]
        areas.sort(key=lambda x: x[1], reverse=True)
        
        kept_indices = []
        
        for idx, _ in areas:
            should_keep = True
            
            # Check against already kept boxes
            for kept_idx in kept_indices:
                kept_box = boxes_xyxy[kept_idx]
                current_box = boxes_xyxy[idx]
                
                # Calculate IoU between current box and kept box
                iou = self._calculate_iou_xyxy(current_box, kept_box)
                
                if iou >= iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept_indices.append(idx)
        
        # Convert back to (x, y, w, h) format
        result = []
        for idx in kept_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            result.append((x1, y1, x2 - x1, y2 - y1))
        
        return result
    
    def _calculate_iou_xyxy(
        self, 
        box1: Tuple[int, int, int, int], 
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between boxes in (x1, y1, x2, y2) format."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

    def merge_boxes_ignore_detail(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Original merge strategy - ignore detail."""
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
                    box1_inside_box2 = (
                        x1 >= x2
                        and y1 >= y2
                        and x1 + w1 <= x2 + w2
                        and y1 + h1 <= y2 + h2
                    )
                    box2_inside_box1 = (
                        x2 >= x1
                        and y2 >= y1
                        and x2 + w2 <= x1 + w1
                        and y2 + h2 <= y1 + h1
                    )
                    
                    iosa = self.calculate_iosa(box1, box2)

                    # Single merge threshold or one box inside another
                    if (
                        (
                            abs(x1 - x2) < self.merge_threshold
                            and abs(y1 - y2) < self.merge_threshold
                        )
                        or (abs(x1 - x2) < self.group_x and abs(y1 - y2) < self.group_y)
                        or box1_inside_box2
                        or box2_inside_box1
                        or iosa >= self.iou_threshold
                    ):
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

    def merge_boxes_keep_detail(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Keep detail strategy - preserve small boxes that would be merged."""
        if not boxes:
            return []

        # Sort boxes by area (smallest first) for easier processing
        sorted_boxes = sorted(boxes, key=lambda box: box[2] * box[3])
        
        final_boxes = []
        boxes_to_skip = set()
        
        # Iterate through all boxes
        for i, current_box in enumerate(sorted_boxes):
            if i in boxes_to_skip:
                continue
                
            current_area = current_box[2] * current_box[3]
            contained_small_boxes = []
            
            # Check if this box contains any smaller boxes
            for j, other_box in enumerate(sorted_boxes):
                if i == j or j in boxes_to_skip:
                    continue
                    
                other_area = other_box[2] * other_box[3]
                
                # Only consider boxes that are smaller than current box
                if other_area >= current_area:
                    continue
                
                # Check if smaller box is contained in current box using IoSA
                iosa = self.calculate_iosa(current_box, other_box)
                if iosa >= self.iou_threshold:
                    contained_small_boxes.append(other_box)
                    boxes_to_skip.add(j)
            
            # Decision logic:
            # - If current box contains multiple small boxes, keep the small ones
            # - If current box contains only one or no small boxes, keep the current box
            if len(contained_small_boxes) > 1:
                # Keep all the small boxes, skip the big box
                final_boxes.extend(contained_small_boxes)
                boxes_to_skip.add(i)
            else:
                # Keep the current box (either it contains no small boxes or only one)
                final_boxes.append(current_box)
        
        # Apply NMS to remove nearly identical overlapping boxes
        final_boxes = self.non_maximum_suppression(final_boxes, iou_threshold=self.nms_iou_threshold)
        
        return final_boxes

    def merge_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        strategy: Literal["ignore_detail", "keep_detail"] | None = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Merge boxes using specified strategy. If strategy is None, use the default strategy."""
        if strategy is None:
            strategy = self.merge_strategy or "ignore_detail"

        if strategy == "ignore_detail":
            return self.merge_boxes_ignore_detail(boxes)
        elif strategy == "keep_detail":
            return self.merge_boxes_keep_detail(boxes)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

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
        color_boxes.sort(key=lambda x: (x[1], x[0]))
        steps.update(steps_color)

        edge_boxes = self.detect_edge_boxes(steps_edge["combined"])

        edge_boxes.sort(key=lambda x: (x[1], x[0]))
        # edge_boxes = self.merge_boxes(edge_boxes, strategy="keep_detail")
        color_boxes = self.merge_boxes(color_boxes, strategy="ignore_detail")

        # Add type prefixes to boxes
        edge_boxes_with_type = [(x, y, w, h, "edge") for x, y, w, h in edge_boxes]
        color_boxes_with_type = [(x, y, w, h, "color") for x, y, w, h in color_boxes]

        all_boxes_with_type = edge_boxes_with_type + color_boxes_with_type

        # Extract just the box coordinates for processing
        all_boxes = [(x, y, w, h) for x, y, w, h, _ in all_boxes_with_type]
        boxes = self.find_components(all_boxes, image.shape)

        # Reconstruct boxes with their types
        final_boxes = []
        for x, y, w, h in boxes:
            # Find the original type for this box (or merged box)
            box_type = "edge"  # default
            for orig_x, orig_y, orig_w, orig_h, orig_type in all_boxes_with_type:
                if (
                    x <= orig_x
                    and y <= orig_y
                    and x + w >= orig_x + orig_w
                    and y + h >= orig_y + orig_h
                ):
                    box_type = orig_type
                    break
            final_boxes.append((x, y, w, h, box_type))
        

        components = self.box2component_with_type(final_boxes, image)
        components.sort(key=lambda c: (c["bbox"]["y"], c["bbox"]["x"], -c["area"]))

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

    def box2component_with_type(
        self, boxes_with_type: List[Tuple[int, int, int, int, str]], image: np.ndarray
    ) -> List[BoxComponent]:
        # Compose metadata
        components: List[BoxComponent] = []
        for i, (x, y, w, h, box_type) in enumerate(boxes_with_type):
            comp_type = self.classify_component((x, y, w, h), image)
            # Add prefix to component type
            prefixed_type = f"{box_type}_{comp_type}"
            components.append(
                {
                    "id": i,
                    "type": prefixed_type,
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
        import random

        vis = image.copy()
        is_gray = len(vis.shape) == 2

        for comp in components:
            bbox = comp["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            if is_gray:
                color = 255
            else:
                color = palette[comp["id"] % len(palette)]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            # label = f"{comp['id']}: {comp['type']}"
            label = f"{comp['id']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            label_y = y - 10 if y - 10 > 10 else y + 20
            cv2.rectangle(
                vis,
                (x, label_y - label_size[1] - 2),
                (x + label_size[0] + 2, label_y + 2),
                color,
                -1,
            )
            cv2.putText(
                vis,
                label,
                (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
        return vis


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1]
    segmenter = UnifiedSegmenter()
    input_image_rgb = cv2.imread(image_path)
    image_bgr = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)
    seg_result = segmenter.segment(image_bgr)
    vis_image_bgr = segmenter.draw_segmentation(image_bgr, seg_result["components"])
    vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_vis.png", vis_image_rgb)
