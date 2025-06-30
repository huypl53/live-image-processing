import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
from pathlib import Path
import gradio as gr
import tempfile
import shutil
import os

# --- Your Original WinFormSegmenter Class (No changes needed here) ---

class WinFormSegmenter:
    def __init__(self):
        self.min_component_area = 1000
        self.merge_threshold = 20
        self.table_aspect_ratio_threshold = 2.0
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced
    
    def detect_gui_elements(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = self.preprocess_image(image)
        edges_canny = cv2.Canny(gray, 50, 150, apertureSize=3)
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_morph = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel_rect)
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        horizontal_lines = cv2.morphologyEx(edges_canny, cv2.MORPH_OPEN, kernel_horizontal)
        vertical_lines = cv2.morphologyEx(edges_canny, cv2.MORPH_OPEN, kernel_vertical)
        combined_edges = cv2.bitwise_or(edges_morph, horizontal_lines)
        combined_edges = cv2.bitwise_or(combined_edges, vertical_lines)
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_component_area:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 30:
                    bounding_boxes.append((x, y, w, h))
        return bounding_boxes
    
    def detect_color_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_ranges = [
            ([40, 50, 50], [80, 255, 255]),
            ([100, 50, 50], [130, 255, 255]),
            ([0, 0, 100], [180, 30, 200]),
        ]
        all_regions = []
        for lower, upper in color_ranges:
            lower, upper = np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_component_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 100 and h > 20:
                        all_regions.append((x, y, w, h))
        return all_regions
    
    def merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if not boxes: return []
        boxes = sorted(boxes, key=lambda x: x[0])
        merged = []
        for current in boxes:
            if not merged:
                merged.append(list(current))
                continue
            merged_with_existing = False
            for i in range(len(merged)):
                if self.should_merge_boxes(tuple(merged[i]), current):
                    x1 = min(merged[i][0], current[0])
                    y1 = min(merged[i][1], current[1])
                    x2 = max(merged[i][0] + merged[i][2], current[0] + current[2])
                    y2 = max(merged[i][1] + merged[i][3], current[1] + current[3])
                    merged[i] = [x1, y1, x2 - x1, y2 - y1]
                    merged_with_existing = True
                    break
            if not merged_with_existing:
                merged.append(list(current))
        return [tuple(m) for m in merged]

    def should_merge_boxes(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        intersects = not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
        prox_x = (x1 <= x2 + w2 + self.merge_threshold) and (x2 <= x1 + w1 + self.merge_threshold)
        prox_y = (y1 <= y2 + h2 + self.merge_threshold) and (y2 <= y1 + h1 + self.merge_threshold)
        return intersects or (prox_x and prox_y)

    def classify_component_type(self, box: Tuple[int, int, int, int], image: np.ndarray) -> str:
        x, y, w, h = box
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > self.table_aspect_ratio_threshold and h > 100: return "table"
        elif aspect_ratio > 3 and h < 50: return "toolbar"
        elif w > 200 and h > 150: return "form_panel"
        elif aspect_ratio < 0.5 and w < 100: return "sidebar"
        elif h < 100 and w > 100: return "input_group"
        else: return "component"
    
    def segment_image(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path}")
        height, width = image.shape[:2]
        
        edge_boxes = self.detect_gui_elements(image)
        color_boxes = self.detect_color_regions(image)
        
        all_boxes = self.merge_overlapping_boxes(edge_boxes + color_boxes)
        
        filtered_boxes = []
        for box in all_boxes:
            x, y, w, h = box
            if (w * h > self.min_component_area and w < width * 0.95 and h < height * 0.95 and
                x >= 0 and y >= 0 and x + w <= width and y + h <= height):
                filtered_boxes.append(box)
        
        components = []
        for i, box in enumerate(filtered_boxes):
            x, y, w, h = box
            component_type = self.classify_component_type(box, image)
            components.append({
                'id': i, 'type': component_type,
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'area': int(w * h)
            })
        
        components.sort(key=lambda c: (-c['area'], c['bbox']['y'], c['bbox']['x']))
        
        return {
            'image_path': image_path,
            'image_size': {'width': int(width), 'height': int(height)},
            'components': components, 'total_components': len(components)
        }

    def draw_segmentation_on_image(self, image: np.ndarray, segmentation_result: Dict) -> np.ndarray:
        vis_image = image.copy()
        type_colors = {
            'table': (0, 255, 0), 'toolbar': (255, 0, 0), 'form_panel': (0, 0, 255),
            'sidebar': (255, 255, 0), 'input_group': (255, 0, 255), 'component': (0, 255, 255)
        }
        for component in segmentation_result.get('components', []):
            bbox = component['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            color = type_colors.get(component['type'], (128, 128, 128))
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3) # Thicker rectangle
            label = f"{component['id']}: {component['type']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y - 10 if y - 10 > 10 else y + 20
            cv2.rectangle(vis_image, (x, label_y - label_size[1] - 5), (x + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(vis_image, label, (x + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return vis_image


# --- GRADIO APPLICATION LOGIC ---

# Global instance of the segmenter
segmenter = WinFormSegmenter()
temp_dirs_to_cleanup = set()

def process_and_update(input_image_rgb: np.ndarray, min_area: int, merge_dist: int, table_ratio: float):
    """
    Main processing function for the Gradio interface.
    Accepts image and hyperparameters, performs segmentation, and returns results.
    """
    if input_image_rgb is None:
        return None, None, None, None, "Status: Please upload an image first."

    # Update the global segmenter instance with values from the UI
    segmenter.min_component_area = min_area
    segmenter.merge_threshold = merge_dist
    segmenter.table_aspect_ratio_threshold = table_ratio
    
    run_temp_dir = Path(tempfile.mkdtemp())
    temp_dirs_to_cleanup.add(run_temp_dir)

    try:
        image_bgr = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)
        temp_image_path = run_temp_dir / "input_image.png"
        cv2.imwrite(str(temp_image_path), image_bgr)

        segmentation_result = segmenter.segment_image(str(temp_image_path))
        
        vis_image_bgr = segmenter.draw_segmentation_on_image(image_bgr, segmentation_result)
        vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)

        components = segmentation_result.get('components', [])
        if not components:
            return vis_image_rgb, None, segmentation_result, None, "Status: Segmentation complete. No components found."

        patches_dir = run_temp_dir / "patches"
        patches_dir.mkdir()
        
        patch_files_for_gallery = []
        for component in components:
            bbox = component['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            patch = image_bgr[y:y+h, x:x+w]
            patch_filename = f"patch_{component['id']:03d}_{component['type']}.png"
            patch_path = patches_dir / patch_filename
            cv2.imwrite(str(patch_path), patch)
            patch_files_for_gallery.append(str(patch_path))

        metadata_path = patches_dir / "segmentation_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(segmentation_result, f, indent=2, ensure_ascii=False)
        
        zip_path_base = str(run_temp_dir / "segmented_patches")
        zip_path = shutil.make_archive(zip_path_base, 'zip', patches_dir)

        status_message = f"Status: Success! Found and extracted {len(components)} components."
        return vis_image_rgb, patch_files_for_gallery, segmentation_result, zip_path, status_message

    except Exception as e:
        error_message = f"Status: An error occurred: {e}"
        print(f"Error: {e}") # Log to console for debugging
        return None, None, None, None, error_message

def cleanup_temp_dirs():
    for d in list(temp_dirs_to_cleanup):
        shutil.rmtree(d, ignore_errors=True)
        temp_dirs_to_cleanup.remove(d)

def create_examples():
    example_dir = "examples"
    os.makedirs(example_dir, exist_ok=True)
    ex1_path = os.path.join(example_dir, "example1.png")
    if not os.path.exists(ex1_path):
        img1 = np.full((400, 600, 3), 220, dtype=np.uint8)
        cv2.rectangle(img1, (20, 20), (580, 80), (100, 150, 100), -1)
        cv2.rectangle(img1, (20, 100), (200, 380), (180, 180, 180), -1)
        cv2.rectangle(img1, (220, 100), (580, 380), (200, 200, 200), -1)
        cv2.imwrite(ex1_path, img1)
    return [ex1_path]

# --- GRADIO UI DEFINITION ---
with gr.Blocks(theme=gr.themes.Soft(), title="Interactive Image Patcher") as demo:
    gr.Markdown("# Interactive UI Image Patcher")
    gr.Markdown("Upload a screenshot and adjust the hyperparameters to see the segmentation results update in real-time.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Screenshot")
            
            with gr.Accordion("Hyperparameters", open=True):
                min_area_slider = gr.Slider(
                    label="Min Component Area",
                    minimum=100, maximum=10000, step=100, value=1000,
                    info="The smallest area (in pixels) for a detected component to be considered valid."
                )
                merge_thresh_slider = gr.Slider(
                    label="Merge Distance Threshold",
                    minimum=0, maximum=100, step=1, value=20,
                    info="How close (in pixels) components can be before they are merged together."
                )
                table_ratio_slider = gr.Slider(
                    label="Table Aspect Ratio",
                    minimum=1.0, maximum=10.0, step=0.1, value=2.0,
                    info="The width-to-height ratio above which a component is classified as a 'table'."
                )
                
            status_text = gr.Textbox(label="Status", interactive=False)
            submit_btn = gr.Button("Segment Image", variant="primary")
            
            gr.Examples(
                examples=create_examples(),
                inputs=input_image,
                label="Example Screenshots"
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Segmentation Visualization"):
                    output_visualization = gr.Image(label="Segmented Image")
                with gr.TabItem("Extracted Patches"):
                    output_gallery = gr.Gallery(label="Component Patches", show_label=False, columns=[3, 4], height="auto", object_fit="contain")
                with gr.TabItem("Segmentation Data (JSON)"):
                    output_json = gr.JSON(label="Component Metadata")
                with gr.TabItem("Download All"):
                    gr.Markdown("Click the link below to download a .zip file containing all extracted patches and the metadata.json file.")
                    output_zip = gr.File(label="Download Patches and Metadata (.zip)")

    # Define the lists of inputs and outputs for the event handlers
    all_inputs = [input_image, min_area_slider, merge_thresh_slider, table_ratio_slider]
    all_outputs = [output_visualization, output_gallery, output_json, output_zip, status_text]
    
    # Wire up the events
    # The button click will trigger the processing
    submit_btn.click(fn=process_and_update, inputs=all_inputs, outputs=all_outputs, api_name="segment")
    
    # Any change to a slider will also trigger the processing
    for slider in [min_area_slider, merge_thresh_slider, table_ratio_slider]:
        slider.change(fn=process_and_update, inputs=all_inputs, outputs=all_outputs)
        
    # When a new image is uploaded, clear old outputs and then run processing
    def on_upload_clear_and_process(img, min_area, merge_dist, table_ratio):
        # First, return cleared outputs
        yield None, None, None, None, "Processing new image..."
        # Then, yield the result of the full processing
        yield process_and_update(img, min_area, merge_dist, table_ratio)

    input_image.upload(fn=on_upload_clear_and_process, inputs=all_inputs, outputs=all_outputs)

    # Register the cleanup function to run when the app session is closed
    demo.unload(cleanup_temp_dirs)

if __name__ == "__main__":
    demo.launch()
