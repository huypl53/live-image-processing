import cv2
import numpy as np
import gradio as gr
import json
import tempfile
import shutil
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from typing import TypedDict


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
        self.max_aspect_ratio = 10.0

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
        # Merge using both single threshold and X/Y grouping
        if not boxes:
            return []
        merged = []
        for box in boxes:
            x, y, w, h = box
            found = False
            for i, (mx, my, mw, mh) in enumerate(merged):
                # Single merge threshold
                if (
                    abs(x - mx) < self.merge_threshold
                    and abs(y - my) < self.merge_threshold
                ):
                    nx1, ny1 = min(x, mx), min(y, my)
                    nx2, ny2 = max(x + w, mx + mw), max(y + h, my + mh)
                    merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                    found = True
                    break
                # X/Y grouping
                if abs(x - mx) < self.group_x and abs(y - my) < self.group_y:
                    nx1, ny1 = min(x, mx), min(y, my)
                    nx2, ny2 = max(x + w, mx + mw), max(y + h, my + mh)
                    merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                    found = True
                    break
            if not found:
                merged.append(box)
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
        type_colors = {
            "table": (0, 255, 0),
            "toolbar": (125, 0, 0),
            "form_panel": (0, 0, 255),
            "sidebar": (255, 255, 0),
            "input_group": (255, 0, 255),
            "component": (0, 255, 255),
        }
        for comp in components:
            bbox = comp["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
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


# --- Gradio UI ---
segmenter = UnifiedSegmenter()
temp_dirs_to_cleanup = set()


def process_and_update(
    input_image_rgb,
    min_area,
    max_area,
    merge_dist,
    group_x,
    group_y,
    table_ratio,
    blur_kernel,
    block_size,
    c_val,
    adaptive_method,
    morph_op,
    morph_kernel,
    morph_iter,
    min_aspect,
    max_aspect,
    previous_temp_dir,
):
    # Cleanup previous run's directory to prevent resource leaks
    if previous_temp_dir and Path(previous_temp_dir).exists():
        shutil.rmtree(previous_temp_dir, ignore_errors=True)
        if previous_temp_dir in temp_dirs_to_cleanup:
            temp_dirs_to_cleanup.remove(previous_temp_dir)

    if input_image_rgb is None:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            "Status: Please upload an image.",
            None,
        )

    run_temp_dir = Path(tempfile.mkdtemp())
    temp_dirs_to_cleanup.add(str(run_temp_dir))

    try:
        # Update segmenter parameters from UI
        segmenter.min_component_area = min_area
        segmenter.max_component_area = max_area
        segmenter.merge_threshold = merge_dist
        segmenter.group_x = group_x
        segmenter.group_y = group_y
        segmenter.table_aspect_ratio_threshold = table_ratio
        segmenter.blur_kernel = blur_kernel
        segmenter.adaptive_block_size = block_size
        segmenter.adaptive_c = c_val
        segmenter.adaptive_method = adaptive_method
        segmenter.morph_op = morph_op
        segmenter.morph_kernel = morph_kernel
        segmenter.morph_iter = morph_iter
        segmenter.min_aspect_ratio = min_aspect
        segmenter.max_aspect_ratio = max_aspect

        image_bgr = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)
        seg_result = segmenter.segment(image_bgr)
        vis_image_bgr = segmenter.draw_segmentation(
            image_bgr, seg_result["components"])
        vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)

        # Intermediate steps
        steps = seg_result["steps"]
        step_imgs = []
        for key in steps:
            img = steps[key]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # step_imgs.append((f"{key}", img))
            step_imgs.append((img, f"{key}"))

        # Save patches
        components = seg_result["components"]
        patches_dir = run_temp_dir / "patches"
        patches_dir.mkdir()
        patch_files = []
        for comp in components:
            bbox = comp["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            patch = image_bgr[y: y + h, x: x + w]
            patch_filename = f"patch_{comp['id']:03d}_{comp['type']}.png"
            patch_path = patches_dir / patch_filename
            cv2.imwrite(str(patch_path), patch)
            patch_files.append(str(patch_path))

        metadata_path = patches_dir / "segmentation_metadata.json"
        metadata = {
            "components": seg_result["components"],
            "total_components": seg_result["total_components"],
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        zip_path_base = str(run_temp_dir / "segmented_patches")
        zip_path = shutil.make_archive(zip_path_base, "zip", patches_dir)

        status = f"Status: Success! Found and extracted {len(components)} components."

        # Prepare intermediate images for gallery
        step_gallery_labels = [[name] for _, name in step_imgs]

        return (
            vis_image_rgb,
            patch_files,
            metadata,
            zip_path,
            step_imgs,
            # step_gallery_imgs_labels,
            step_gallery_labels,
            status,
            str(run_temp_dir),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_message = f"Status: An error occurred: {e}"
        # Return 8 values to match the number of outputs
        return None, None, None, None, None, None, error_message, str(run_temp_dir)


def cleanup_temp_dirs():
    for d in list(temp_dirs_to_cleanup):
        shutil.rmtree(d, ignore_errors=True)


css = """
#horizontal_gallery > .grid-container {
    flex-wrap: nowrap;
    overflow-x: auto;
}

#param-col {
  max-height: 400px;
  overflow-y: auto;
}
"""

with gr.Blocks(title="Unified UI Image Segmenter", css=css, fill_height=True) as demo:
    gr.Markdown("# Unified UI Image Segmenter")
    gr.Markdown(
        "Upload a screenshot and adjust parameters. See intermediate steps and final segmentation."
    )

    # Hidden state to manage the temporary directory from the previous run
    previous_temp_dir_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Screenshot")
            with gr.Accordion(
                "Segmentation Parameters", open=True, elem_id="param-col"
            ):
                min_area_slider = gr.Slider(
                    100,
                    10000,
                    value=500,
                    step=100,
                    label="Min Component Area",
                    info="Minimum pixel area for a detected region to be considered valid. Increase to filter out noise.",
                )
                max_area_slider = gr.Slider(
                    10000,
                    2_000_000,
                    value=1_000_000,
                    step=10000,
                    label="Max Component Area",
                    info="Maximum pixel area for a detected region. Decrease to ignore very large (background) regions.",
                )
                merge_thresh_slider = gr.Slider(
                    0,
                    100,
                    value=20,
                    step=1,
                    label="Merge Distance Threshold",
                    info="How close (in pixels) two components can be before being merged into one.",
                )
                group_x_slider = gr.Slider(
                    0,
                    200,
                    value=40,
                    step=1,
                    label="Group X Threshold",
                    info="Horizontal distance for grouping/merging components in the same row.",
                )
                group_y_slider = gr.Slider(
                    0,
                    200,
                    value=30,
                    step=1,
                    label="Group Y Threshold",
                    info="Vertical distance for grouping/merging components in the same column.",
                )
                table_ratio_slider = gr.Slider(
                    1.0,
                    10.0,
                    value=2.0,
                    step=0.1,
                    label="Table Aspect Ratio",
                    info="Width-to-height ratio above which a component is classified as a table.",
                )
                blur_kernel_slider = gr.Slider(
                    1,
                    15,
                    value=3,
                    step=2,
                    label="Gaussian Blur Kernel Size (odd)",
                    info="Size of the Gaussian blur kernel (must be odd); helps reduce noise before thresholding.",
                )
                block_size_slider = gr.Slider(
                    3,
                    51,
                    value=11,
                    step=2,
                    label="Adaptive Block Size (odd)",
                    info="Size of the local window for adaptive thresholding (must be odd). Smaller values detect finer details.",
                )
                c_val_slider = gr.Slider(
                    0,
                    20,
                    value=2,
                    step=1,
                    label="Adaptive Threshold C",
                    info="Constant subtracted from the mean/weighted mean in adaptive thresholding. Lower values are more sensitive.",
                )
                adaptive_method_radio = gr.Radio(
                    ["mean", "gaussian"],
                    value="gaussian",
                    label="Adaptive Threshold Method",
                    info="Method for adaptive thresholding: 'mean' or 'gaussian'.",
                )
                morph_op_radio = gr.Radio(
                    ["dilate", "erode", "open", "close"],
                    value="close",
                    label="Morphological Operation",
                    info="Type of morphological operation: dilate, erode, open, or close. 'Close' joins gaps, 'open' removes noise.",
                )
                morph_kernel_slider = gr.Slider(
                    1,
                    15,
                    value=3,
                    step=2,
                    label="Morph Kernel Size (odd)",
                    info="Size of the structuring element for morphological operations (must be odd).",
                )
                morph_iter_slider = gr.Slider(
                    1,
                    10,
                    value=2,
                    step=1,
                    label="Morph Iterations",
                    info="Number of times the morphological operation is applied.",
                )
                min_aspect_slider = gr.Slider(
                    0.05,
                    2.0,
                    value=0.2,
                    step=0.05,
                    label="Min Aspect Ratio",
                    info="Minimum width/height ratio for a component to be considered valid.",
                )
                max_aspect_slider = gr.Slider(
                    2.0,
                    20.0,
                    value=10.0,
                    step=0.1,
                    label="Max Aspect Ratio",
                    info="Maximum width/height ratio for a component to be considered valid.",
                )
            status_text = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                submit_btn = gr.Button("Segment Image", variant="primary")
                save_params_btn = gr.Button("Save Params", variant="secondary")
                params_file = gr.File(label="Download Params", visible=True)
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Segmentation Visualization"):
                    output_visualization = gr.Image(label="Segmented Image")
                with gr.TabItem("Extracted Patches"):
                    output_gallery = gr.Gallery(
                        label="Component Patches",
                        show_label=False,
                        columns=4,
                        height="auto",
                        object_fit="contain",
                    )
                with gr.TabItem("Segmentation Data (JSON)"):
                    output_json = gr.JSON(label="Component Metadata")
                with gr.TabItem("Download All"):
                    gr.Markdown(
                        "Click below to download a .zip file containing all patches and metadata."
                    )
                    output_zip = gr.File(
                        label="Download Patches and Metadata (.zip)")
            # Place Intermediate Steps right below the tab bar, in the same column
            gr.Markdown("## Intermediate Steps")
            with gr.Row():
                step_gallery = gr.Gallery(
                    label="Intermediate Steps",
                    show_label=True,
                    columns=10,
                    height="auto",
                    object_fit="contain",
                    elem_id="horizontal_gallery",
                )
            step_labels = gr.Dataframe(
                headers=["Step"], label="Step Names", interactive=False, type="array"
            )

    # Save Params logic
    def save_params_fn(
        min_area,
        max_area,
        merge_dist,
        group_x,
        group_y,
        table_ratio,
        blur_kernel,
        block_size,
        c_val,
        adaptive_method,
        morph_op,
        morph_kernel,
        morph_iter,
        min_aspect,
        max_aspect,
    ):
        import tempfile
        import json
        import os

        params = {
            "min_area": min_area,
            "max_area": max_area,
            "merge_dist": merge_dist,
            "group_x": group_x,
            "group_y": group_y,
            "table_ratio": table_ratio,
            "blur_kernel": blur_kernel,
            "block_size": block_size,
            "c_val": c_val,
            "adaptive_method": adaptive_method,
            "morph_op": morph_op,
            "morph_kernel": morph_kernel,
            "morph_iter": morph_iter,
            "min_aspect": min_aspect,
            "max_aspect": max_aspect,
        }
        fd, path = tempfile.mkstemp(suffix="_params.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        os.close(fd)
        return path

    # Define the lists of inputs and outputs for the event handlers
    all_sliders = [
        min_area_slider,
        max_area_slider,
        merge_thresh_slider,
        group_x_slider,
        group_y_slider,
        table_ratio_slider,
        blur_kernel_slider,
        block_size_slider,
        c_val_slider,
        morph_kernel_slider,
        morph_iter_slider,
        min_aspect_slider,
        max_aspect_slider,
    ]
    all_radios = [adaptive_method_radio, morph_op_radio]

    all_inputs = [
        input_image,
        min_area_slider,
        max_area_slider,
        merge_thresh_slider,
        group_x_slider,
        group_y_slider,
        table_ratio_slider,
        blur_kernel_slider,
        block_size_slider,
        c_val_slider,
        adaptive_method_radio,
        morph_op_radio,
        morph_kernel_slider,
        morph_iter_slider,
        min_aspect_slider,
        max_aspect_slider,
        previous_temp_dir_state,
    ]
    all_outputs = [
        output_visualization,
        output_gallery,
        output_json,
        output_zip,
        step_gallery,
        step_labels,
        status_text,
        previous_temp_dir_state,
    ]

    # Wire up the events
    submit_btn.click(
        fn=process_and_update,
        inputs=all_inputs,
        outputs=all_outputs,
        api_name="segment",
    )
    save_params_btn.click(
        fn=save_params_fn,
        inputs=[
            min_area_slider,
            max_area_slider,
            merge_thresh_slider,
            group_x_slider,
            group_y_slider,
            table_ratio_slider,
            blur_kernel_slider,
            block_size_slider,
            c_val_slider,
            adaptive_method_radio,
            morph_op_radio,
            morph_kernel_slider,
            morph_iter_slider,
            min_aspect_slider,
            max_aspect_slider,
        ],
        outputs=params_file,
    )

    # Use .release() for sliders to avoid excessive updates while dragging
    for slider in all_sliders:
        slider.release(fn=process_and_update,
                       inputs=all_inputs, outputs=all_outputs)

    # .change() is fine for radio buttons as it's a single action
    for radio in all_radios:
        radio.change(fn=process_and_update,
                     inputs=all_inputs, outputs=all_outputs)

    input_image.upload(fn=process_and_update,
                       inputs=all_inputs, outputs=all_outputs)

    # Register the cleanup function to run when the app session is closed (as a fallback)
    # demo.unload(cleanup_temp_dirs)

if __name__ == "__main__":
    demo.launch()
