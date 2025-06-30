import gradio as gr
import cv2
import numpy as np
import os
import shutil
from PIL import Image
import click

# --- 1. Core Image Processing Logic (Unchanged) ---
# These functions are self-contained and work perfectly here.

def get_enclosing_box(boxes):
    """ Calculates the single bounding box that encloses a list of boxes. """
    if not boxes: return None
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[0] + b[2] for b in boxes)
    max_y = max(b[1] + b[3] for b in boxes)
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def boxes_are_close(box1, box2, x_thresh, y_thresh):
    """ Checks if two boxes are close enough to be considered part of the same group. """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ex1, ey1, ew1, eh1 = x1 - x_thresh, y1 - y_thresh, w1 + 2*x_thresh, h1 + 2*y_thresh
    ex2, ey2, ew2, eh2 = x2, y2, w2, h2
    return ex1 < ex2 + ew2 and ex1 + ew1 > ex2 and ey1 < ey2 + eh2 and ey1 + eh1 > ey2

def refine_and_filter_patches(patches, image_shape):
    """ Removes patches that are fully contained within another, larger patch. """
    patches = sorted(patches, key=lambda p: p[2] * p[3], reverse=True)
    final_patches = []
    image_area = image_shape[0] * image_shape[1]
    for p in patches:
        if p[2] * p[3] > image_area * 0.95: continue
        is_contained = False
        for fp in final_patches:
            if (p[0] >= fp[0] and p[1] >= fp[1] and 
                p[0] + p[2] <= fp[0] + fp[2] and 
                p[1] + p[3] <= fp[1] + fp[3]):
                is_contained = True
                break
        if not is_contained:
            final_patches.append(p)
    return final_patches

def process_image_with_params(image, params):
    """
    Processes the image using a dictionary of parameters and returns the
    visualization image and the list of patch coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = int(params['block_size'])
    if block_size % 2 == 0: block_size += 1
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, int(params['threshold_c']))
    kernel_size = int(params['dilate_kernel'])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=int(params['dilate_iter']))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    initial_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > params['min_area']]
    num_boxes = len(initial_boxes)
    adj_matrix = [[0] * num_boxes for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            if boxes_are_close(initial_boxes[i], initial_boxes[j], params['group_x'], params['group_y']):
                adj_matrix[i][j] = adj_matrix[j][i] = 1
    visited = [False] * num_boxes
    groups = []
    for i in range(num_boxes):
        if not visited[i]:
            component = []
            q = [i]
            visited[i] = True
            while q:
                u = q.pop(0)
                component.append(initial_boxes[u])
                for v in range(num_boxes):
                    if adj_matrix[u][v] and not visited[v]:
                        visited[v] = True
                        q.append(v)
            groups.append(component)
    grouped_patches = [get_enclosing_box(g) for g in groups if g]
    final_patches = refine_and_filter_patches(grouped_patches, image.shape)
    visualization_image = image.copy()
    for i, (x, y, w, h) in enumerate(final_patches):
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(visualization_image, f"{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return visualization_image, final_patches

# --- 2. Gradio Application Logic ---

# Global variable for the image folder
IMAGE_FOLDER = "sample_images"

def get_image_files():
    """Finds image files in the specified folder."""
    if not os.path.isdir(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"Created folder '{IMAGE_FOLDER}'. Please place screenshots inside and restart.")
        return []
    return [f for f in sorted(os.listdir(IMAGE_FOLDER)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def generate_visualization(image_filename, block_size, threshold_c, dilate_kernel, dilate_iter, min_area, group_x, group_y):
    """Main function that Gradio will call to update the UI."""
    if not image_filename:
        # Create a blank placeholder image if no file is selected
        return np.zeros((400, 600, 3), dtype=np.uint8), "Please select an image.", None, None

    # Load the original image
    full_path = os.path.join(IMAGE_FOLDER, image_filename)
    original_image = cv2.imread(full_path)
    if original_image is None:
        return None, f"Error loading {image_filename}", None, None

    # Bundle parameters
    params = {
        'block_size': block_size, 'threshold_c': threshold_c, 'dilate_kernel': dilate_kernel,
        'dilate_iter': dilate_iter, 'min_area': min_area, 'group_x': group_x, 'group_y': group_y
    }
    
    # Process the image
    vis_image, final_patches = process_image_with_params(original_image, params)
    
    # Convert from BGR (OpenCV) to RGB (Gradio/PIL) for correct color display
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    # Return the visualization, a status message, and the state for saving
    status = f"Processed {image_filename}. Found {len(final_patches)} patches."
    return vis_image_rgb, status, full_path, final_patches

def save_patches_handler(original_image_path, patches_to_save):
    """Function to handle the save button click."""
    if not original_image_path or not patches_to_save:
        return "Error: No valid patches to save. Process an image first."

    original_image = cv2.imread(original_image_path)
    if original_image is None:
        return f"Error: Could not re-read original image at {original_image_path}"

    base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
    output_dir = "tuned_patches_output"
    patch_output_dir = os.path.join(output_dir, base_filename)
    
    if os.path.exists(patch_output_dir):
        shutil.rmtree(patch_output_dir)
    os.makedirs(patch_output_dir, exist_ok=True)
    
    padding = 5
    for i, (x, y, w, h) in enumerate(patches_to_save):
        px1, py1 = max(0, x - padding), max(0, y - padding)
        px2, py2 = min(original_image.shape[1], x + w + padding), min(original_image.shape[0], y + h + padding)
        patch = original_image[py1:py2, px1:px2]
        patch_filename = os.path.join(patch_output_dir, f"patch_{i+1:02d}.png")
        cv2.imwrite(patch_filename, patch)
    
    return f"Success! Saved {len(patches_to_save)} patches to '{patch_output_dir}'"

# --- 3. Build the Gradio Interface ---

def start(im_dir: str):
    global IMAGE_FOLDER
    IMAGE_FOLDER = im_dir
    image_files = get_image_files()

    with gr.Blocks(theme=gr.themes.Soft(), title="Interactive UI Patcher") as demo:
        gr.Markdown("# Interactive UI Patcher\nAdjust parameters to segment UI components and save the results.")
        
        # Hidden state components to store data between function calls
        state_original_path = gr.State(value=None)
        state_patches = gr.State(value=None)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("## Controls")
                image_selector = gr.Dropdown(
                    choices=image_files, 
                    label="Select Screenshot", 
                    value=image_files[0] if image_files else None
                )
                
                with gr.Accordion("Segmentation Parameters", open=True):
                    s_block = gr.Slider(3, 51, value=11, step=2, label="Adaptive Block Size")
                    s_thresh_c = gr.Slider(1, 15, value=2, step=1, label="Adaptive Threshold C")
                    s_dilate_kernel = gr.Slider(1, 15, value=3, step=1, label="Dilation Kernel Size")
                    s_dilate_iter = gr.Slider(1, 10, value=2, step=1, label="Dilation Iterations")
                    s_min_area = gr.Slider(50, 1000, value=200, step=10, label="Min Contour Area")

                with gr.Accordion("Grouping Parameters", open=True):
                    s_group_x = gr.Slider(10, 100, value=40, step=1, label="Group X Threshold")
                    s_group_y = gr.Slider(10, 100, value=30, step=1, label="Group Y Threshold")
                
                save_button = gr.Button("Save Patches", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=3):
                image_output = gr.Image(label="Processed Image", type="numpy")

        # List of all input components that trigger a re-run
        inputs = [image_selector, s_block, s_thresh_c, s_dilate_kernel, s_dilate_iter, s_min_area, s_group_x, s_group_y]
        
        # List of all output components
        outputs = [image_output, status_box, state_original_path, state_patches]

        # Wire up the event handlers
        for component in inputs:
            component.change(fn=generate_visualization, inputs=inputs, outputs=outputs)
            
        save_button.click(fn=save_patches_handler, inputs=[state_original_path, state_patches], outputs=status_box)

        # Trigger initial processing when the app loads
        demo.load(fn=generate_visualization, inputs=inputs, outputs=outputs)

    if not image_files:
        print("="*60)
        print("WARNING: No images found in the 'sample_images' folder.")
        print("Please create the folder, add your screenshots, and run again.")
        print("The Gradio app will launch, but the dropdown will be empty.")
        print("="*60)
    
    # Launch the Gradio web server
    demo.launch()

@click.command()
@click.option('--image-dir', default='./yolo-images')
def main(image_dir: str):
    start(image_dir)

if __name__ == "__main__":
    main()
