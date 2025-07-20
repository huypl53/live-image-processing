from typing import List
import json
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
from skimage import color as skimage_color
from skimage.feature import canny
from skimage.morphology import dilation, square


@dataclass
class ColorExtractorConfigV2:
    """
    Advanced configuration for the UIColorExtractorV2.

    Attributes:
        palette (List[Tuple[int, int, int]]):
            The target color palette.
        color_space_for_mapping (Literal['RGB', 'LAB']):
            Color space for finding the closest palette color. 'LAB' is more
            perceptually accurate for human vision.
        edge_detection_thresholds (Tuple[float, float]):
            Low and high thresholds for the Canny edge detector to create a mask
            and ignore anti-aliased pixels.
        edge_dilation_radius (int):
            Radius for dilating the detected edges to ensure the mask covers
            all blurry/anti-aliased pixels.
        center_sample_ratio (float):
            The ratio of the patch's dimension to use for defining the "center" area.
            Used to semantically identify foreground color (e.g., text in a button).
    """

    palette: List[Tuple[int, int, int]]
    color_space_for_mapping: Literal["RGB", "LAB"] = "LAB"
    edge_detection_thresholds: Tuple[float, float] = (50, 150)
    edge_dilation_radius: int = 2
    center_sample_ratio: float = 0.4


class UIColorExtractorV2:
    """
    An advanced color extractor using histogram analysis, edge masking,
    and semantic classification to improve accuracy.
    """

    def __init__(self, config: ColorExtractorConfigV2):
        self.config = config
        self.palette_rgb_np = np.array(self.config.palette)

        if not self.config.palette:
            raise ValueError("The color palette cannot be empty.")

        # Pre-convert palette to LAB space if needed, for efficiency
        if self.config.color_space_for_mapping == "LAB":
            # skimage expects RGB in float format [0, 1]
            palette_float = self.palette_rgb_np / 255.0
            self.palette_lab_np = skimage_color.rgb2lab(palette_float)

    def _find_closest_palette_color(
        self, rgb_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Finds the closest color in the palette using the configured color space."""
        if self.config.color_space_for_mapping == "LAB":
            # Convert single color to LAB
            color_float = np.array(rgb_color) / 255.0
            color_lab = skimage_color.rgb2lab(color_float.reshape(1, 1, 3))
            # Calculate Euclidean distance in LAB space
            distances = np.linalg.norm(self.palette_lab_np - color_lab, axis=1)
        else:  # RGB distance
            distances = np.linalg.norm(self.palette_rgb_np - rgb_color, axis=1)

        closest_index = np.argmin(distances)
        return tuple(self.palette_rgb_np[closest_index])

    def extract_colors(
        self, image_patch: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Extracts foreground and background colors from a single UI item patch.
        """
        h, w, _ = image_patch.shape
        if w <= 0 or h <= 0:
            return None

        # --- 1. Edge Detection and Masking to ignore anti-aliasing ---
        grayscale_patch = skimage_color.rgb2gray(image_patch)
        # Convert to 8-bit for canny
        grayscale_patch_8bit = (grayscale_patch * 255).astype(np.uint8)

        low_thresh, high_thresh = self.config.edge_detection_thresholds
        edges = canny(
            grayscale_patch_8bit,
            sigma=1,
            low_threshold=low_thresh,
            high_threshold=high_thresh,
        )

        # Dilate edges to create a wider mask
        dilated_edges = dilation(
            edges, square(self.config.edge_dilation_radius * 2 + 1)
        )

        # Create a mask of pixels to *keep* (non-edge pixels)
        mask = ~dilated_edges

        # --- 2. Mode/Histogram-based Color Extraction ---
        # Get pixels that are not part of the edge mask
        valid_pixels = image_patch[mask]
        if valid_pixels.shape[0] < 2:
            return None  # Not enough data after masking

        # Find the two most frequent colors (mode)
        unique_colors, counts = np.unique(
            valid_pixels, axis=0, return_counts=True)

        if len(unique_colors) < 2:
            return None  # Monochromatic after masking

        # Get indices of the two most frequent colors
        top_two_indices = counts.argsort()[-2:]
        color1 = tuple(unique_colors[top_two_indices[1]])
        color2 = tuple(unique_colors[top_two_indices[0]])

        # --- 3. Semantic Classification (Center vs. Border) ---
        # The color more prevalent in the center is likely the foreground.
        center_ratio = self.config.center_sample_ratio
        cx_start, cx_end = int(w * (0.5 - center_ratio / 2)), int(
            w * (0.5 + center_ratio / 2)
        )
        cy_start, cy_end = int(h * (0.5 - center_ratio / 2)), int(
            h * (0.5 + center_ratio / 2)
        )

        center_patch = image_patch[cy_start:cy_end, cx_start:cx_end]
        center_pixels = center_patch.reshape(-1, 3)

        # Count occurrences of our dominant colors in the center
        count1_center = np.sum(np.all(center_pixels == color1, axis=1))
        count2_center = np.sum(np.all(center_pixels == color2, axis=1))

        if count1_center > count2_center:
            raw_fg_color, raw_bg_color = color1, color2
        else:
            raw_fg_color, raw_bg_color = color2, color1

        # --- 4. Map to Predefined Palette ---
        palette_fg = self._find_closest_palette_color(raw_fg_color)
        palette_bg = self._find_closest_palette_color(raw_bg_color)

        # Avoid returning the same color for both fg and bg
        if palette_fg == palette_bg:
            # If they map to the same color, maybe the raw colors were too close.
            # We can return None or just accept the result. Let's return None for a cleaner demo.
            return None

        return palette_fg, palette_bg


# Import our NEW advanced classes
# from color_extractor_v2 import UIColorExtractorV2, ColorExtractorConfigV2


# (Helper functions create_demo_visualization and create_error_visualization are identical to before)
def create_demo_visualization(
    fg_color: tuple, bg_color: tuple, original_size: tuple
) -> Image.Image:
    w, h = 250, 120
    demo_img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(demo_img)
    text_lines = [
        "Extracted Colors:",
        f"Foreground: {fg_color}",
        f"Background: {bg_color}",
        f"Original Size: {original_size[0]}x{original_size[1]}",
    ]
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = None
    y_pos = 10
    for line in text_lines:
        draw.text((10, y_pos), line, fill=fg_color, font=font)
        y_pos += 22
    return demo_img


def create_error_visualization(message: str) -> Image.Image:
    w, h = 250, 120
    demo_img = Image.new("RGB", (w, h), color=(220, 220, 220))
    draw = ImageDraw.Draw(demo_img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = None
    draw.text((10, 10), message, fill=(200, 0, 0), font=font)
    return demo_img


# --- Main processing function for Gradio ---


def process_ui_patches_v2(
    json_config_str: str, uploaded_files: List[str]
) -> List[Image.Image]:
    if not json_config_str:
        return [create_error_visualization("Error: Config JSON cannot be empty.")]
    if not uploaded_files:
        return [create_error_visualization("Error: Please upload at least one image.")]

    try:
        config_dict = json.loads(json_config_str)
        # We can now directly pass the dict to the dataclass constructor if keys match
        config = ColorExtractorConfigV2(**config_dict)
        extractor = UIColorExtractorV2(config)
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
        return [create_error_visualization(f"Config Error:\n{str(e)}")]

    output_images = []
    for file_obj in uploaded_files:
        try:
            img_pil = Image.open(file_obj.name).convert("RGB")
            img_np = np.array(img_pil)
            h, w, _ = img_np.shape

            # The entire uploaded image is treated as the patch
            result = extractor.extract_colors(img_np)

            if result:
                fg_color, bg_color = result
                demo_img = create_demo_visualization(
                    fg_color, bg_color, (w, h))
                output_images.append(demo_img)
            else:
                error_msg = "Extraction Failed:\n- Check if image is monochromatic.\n- Try adjusting edge thresholds."
                output_images.append(create_error_visualization(error_msg))
        except Exception as e:
            output_images.append(
                create_error_visualization(f"Processing Error:\n{str(e)}")
            )

    return output_images


# --- Define NEW default configuration for the V2 UI ---

default_config_v2 = {
    "palette": [
        [255, 255, 255],
        [0, 0, 0],
        [0, 122, 204],
        [200, 200, 200],
        [240, 240, 240],
        [50, 50, 50],
        [43, 132, 69],  # Added a green
    ],
    "color_space_for_mapping": "LAB",
    "edge_detection_thresholds": [50, 150],
    "edge_dilation_radius": 2,
    "center_sample_ratio": 0.4,
}

default_json_str_v2 = json.dumps(default_config_v2, indent=2)

# --- Build the Gradio App ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Advanced UI Color Extractor (V2)
        This version uses a more robust algorithm to handle anti-aliasing and improve foreground/background detection.
        - **Histogram Analysis**: Finds the most frequent colors, not averages.
        - **Edge Masking**: Ignores blurry anti-aliased pixels around text and shapes.
        - **Semantic Analysis**: Assumes the color in the center of the patch is the foreground.
        - **LAB Color Space**: Uses a perceptually uniform color space for more accurate palette mapping.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configure Extractor (V2)")
            json_config = gr.Textbox(
                lines=15,
                label="ColorExtractorConfigV2 (JSON)",
                value=default_json_str_v2,
            )

            gr.Markdown("### 2. Upload UI Patches")
            file_upload = gr.File(
                label="Upload Images", file_count="multiple", file_types=["image"]
            )

            process_btn = gr.Button("Extract Colors", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 3. Results")
            gallery_output = gr.Gallery(
                label="Extracted Color Demonstrations",
                show_label=False,
                columns=4,
                object_fit="contain",
                height="auto",
            )

    process_btn.click(
        fn=process_ui_patches_v2,
        inputs=[json_config, file_upload],
        outputs=[gallery_output],
    )

if __name__ == "__main__":
    demo.launch()
