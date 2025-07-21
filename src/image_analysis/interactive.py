import json
from pathlib import Path
import shutil
import tempfile

import cv2
import gradio as gr
import yaml

from image_analysis.segmenter import UnifiedSegmenter
from image_analysis.config import load_segmenter_config, save_segmenter_config


# --- Gradio UI ---
segmenter = UnifiedSegmenter()
temp_dirs_to_cleanup = set()


def process_and_update(
    input_image_rgb,
    min_area,
    max_area,
    iou_threshold,
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
    # Edge detection options
    use_canny,
    use_morph,
    canny_low,
    canny_high,
    canny_aperture,
    # Color detection options
    color_morph_kernel,
    color_morph_iter,
    use_color_close,
    use_color_open,
    # Box processing options
    enable_merge,
    merge_strategy,
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
            "Status: Please upload an image.",
            None,
        )

    run_temp_dir = Path(tempfile.mkdtemp())
    temp_dirs_to_cleanup.add(str(run_temp_dir))

    try:
        # Update segmenter parameters from UI
        config_update = {
            "min_component_area": min_area,
            "max_component_area": max_area,
            "iou_threshold": iou_threshold,
            "merge_threshold": merge_dist,
            "group_x": group_x,
            "group_y": group_y,
            "table_aspect_ratio_threshold": table_ratio,
            "blur_kernel": blur_kernel,
            "adaptive_block_size": block_size,
            "adaptive_c": c_val,
            "adaptive_method": adaptive_method,
            "morph_op": morph_op,
            "morph_kernel": morph_kernel,
            "morph_iter": morph_iter,
            "min_aspect_ratio": min_aspect,
            "max_aspect_ratio": max_aspect,
            # Edge detection options
            "use_canny": use_canny,
            "use_morph": use_morph,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "canny_aperture": canny_aperture,
            # Color detection options
            "color_morph_kernel": color_morph_kernel,
            "color_morph_iter": color_morph_iter,
            "use_color_close": use_color_close,
            "use_color_open": use_color_open,
            # Box processing options
            "enable_merge": enable_merge,
            "merge_strategy": merge_strategy,
        }
        segmenter.update_config(config_update)

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
            step_gallery_labels,
            status,
            str(run_temp_dir),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_message = f"Status: An error occurred: {e}"
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
                    5,
                    10000,
                    value=100,
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
                iou_threshold_slider = gr.Slider(
                    0.0,
                    1.0,
                    value=0.15,
                    step=0.01,
                    label="IoU Threshold",
                    info="IoU threshold for merging boxes. Increase to merge more boxes.",
                )
                merge_thresh_slider = gr.Slider(
                    0,
                    100,
                    value=10,
                    step=1,
                    label="Merge Distance Threshold",
                    info="How close (in pixels) two components can be before being merged into one.",
                )
                group_x_slider = gr.Slider(
                    0,
                    200,
                    value=10,
                    step=1,
                    label="Group X Threshold",
                    info="Horizontal distance for grouping/merging components in the same row.",
                )
                group_y_slider = gr.Slider(
                    0,
                    200,
                    value=5,
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
                    0,
                    15,
                    value=1,
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

                use_morph_checkbox = gr.Checkbox(
                    value=True, label="Use Morphological Operation"
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
                    value=1,
                    step=1,
                    label="Morph Iterations",
                    info="Number of times the morphological operation is applied.",
                )
                min_aspect_slider = gr.Slider(
                    0.05,
                    2.0,
                    value=0.05,
                    step=0.05,
                    label="Min Aspect Ratio",
                    info="Minimum width/height ratio for a component to be considered valid.",
                )
                max_aspect_slider = gr.Slider(
                    2.0,
                    100.0,
                    value=70.0,
                    step=0.1,
                    label="Max Aspect Ratio",
                    info="Maximum width/height ratio for a component to be considered valid.",
                )
                # Add edge detection options
                use_canny_checkbox = gr.Checkbox(
                    value=False, label="Use Canny Edge Detection"
                )
                canny_low_slider = gr.Slider(
                    0, 255, value=50, step=1, label="Canny Low Threshold"
                )
                canny_high_slider = gr.Slider(
                    0, 255, value=150, step=1, label="Canny High Threshold"
                )
                canny_aperture_slider = gr.Slider(
                    3, 7, value=3, step=2, label="Canny Aperture Size (odd)"
                )
                # Add color detection options
                color_morph_kernel_slider = gr.Slider(
                    1, 15, value=1, step=2, label="Color Morph Kernel Size (odd)"
                )
                color_morph_iter_slider = gr.Slider(
                    1, 10, value=1, step=1, label="Color Morph Iterations"
                )
                use_color_close_checkbox = gr.Checkbox(
                    value=True, label="Use Color Morph Close"
                )
                use_color_open_checkbox = gr.Checkbox(
                    value=True, label="Use Color Morph Open"
                )
                # Add box processing options
                enable_merge_checkbox = gr.Checkbox(
                    value=True, label="Enable Box Merging",
                    info="Enable merging of nearby boxes. Disable to keep all detected boxes separate."
                )
                merge_strategy_radio = gr.Radio(
                    ["ignore_detail", "keep_detail", "spatial_relation"],
                    value="ignore_detail",
                    label="Merge Strategy",
                    info="Strategy for merging boxes: 'ignore_detail' merges boxes based on their bounding box, 'keep_detail' merges boxes based on their detail.",
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
        iou_threshold,
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
        import os

        params = {
            "min_component_area": min_area,
            "max_component_area": max_area,
            "nms_iou_threshold": iou_threshold,
            "merge_threshold": merge_dist,
            "group_x": group_x,
            "group_y": group_y,
            "table_aspect_ratio_threshold": table_ratio,
            "blur_kernel": blur_kernel,
            "adaptive_block_size": block_size,
            "adaptive_c": c_val,
            "adaptive_method": adaptive_method,
            "morph_op": morph_op,
            "morph_kernel": morph_kernel,
            "morph_iter": morph_iter,
            "min_aspect_ratio": min_aspect,
            "max_aspect_ratio": max_aspect,
        }
        fd, path = tempfile.mkstemp(suffix="_params.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(params, f, default_flow_style=False, indent=2, allow_unicode=True)
        os.close(fd)
        return path

    # Define the lists of inputs and outputs for the event handlers
    all_sliders = [
        min_area_slider,
        max_area_slider,
        iou_threshold_slider,
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
        # Add new sliders
        canny_low_slider,
        canny_high_slider,
        canny_aperture_slider,
        color_morph_kernel_slider,
        color_morph_iter_slider,
    ]
    all_radios = [adaptive_method_radio, morph_op_radio, merge_strategy_radio]
    all_checkboxes = [
        use_canny_checkbox,
        use_morph_checkbox,
        use_color_close_checkbox,
        use_color_open_checkbox,
        enable_merge_checkbox,
    ]

    all_inputs = [
        input_image,
        min_area_slider,
        max_area_slider,
        iou_threshold_slider,
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
        use_canny_checkbox,
        use_morph_checkbox,
        canny_low_slider,
        canny_high_slider,
        canny_aperture_slider,
        color_morph_kernel_slider,
        color_morph_iter_slider,
        use_color_close_checkbox,
        use_color_open_checkbox,
        enable_merge_checkbox,
        merge_strategy_radio,
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
            iou_threshold_slider,
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

    # Add event handlers for checkboxes
    for checkbox in all_checkboxes:
        checkbox.change(fn=process_and_update,
                       inputs=all_inputs, outputs=all_outputs)

    input_image.upload(fn=process_and_update,
                       inputs=all_inputs, outputs=all_outputs)

    # Register the cleanup function to run when the app session is closed (as a fallback)
    # demo.unload(cleanup_temp_dirs)

if __name__ == "__main__":
    demo.launch()
