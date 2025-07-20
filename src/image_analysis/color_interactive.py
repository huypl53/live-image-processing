import gradio as gr
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, Tuple, Optional
import traceback

# Import your color extraction classes
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import cv2
from collections import Counter
from scipy import ndimage
import colorsys


@dataclass
class ColorExtractionConfig:
    """Enhanced configuration for robust UI color extraction"""

    # Mode-based extraction
    use_histogram_mode: bool = True
    histogram_bins: int = 32  # Reduce color space for clustering similar colors
    # Minimum area to be considered significant
    min_color_area_percentage: float = 0.08

    # Edge detection and masking
    enable_edge_masking: bool = True
    edge_detection_method: str = "canny"  # 'canny', 'sobel', 'laplacian'
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    # Expand edge mask to remove more anti-aliasing
    edge_dilation_kernel_size: int = 3

    # Semantic classification
    use_semantic_classification: bool = True
    center_weight_factor: float = (
        2.0  # Weight center pixels more for background detection
    )
    corner_sample_size: int = 5  # Size of corner regions to sample for background

    # Color space analysis
    primary_color_space: str = "hsv"  # 'rgb', 'hsv', 'lab'
    use_perceptual_distance: bool = True

    # Multi-method consensus
    enable_consensus: bool = True
    consensus_methods: List[str] = None

    # Palette matching
    # 'euclidean', 'delta_e', 'hsv_weighted'
    color_distance_metric: str = "delta_e"
    max_color_distance: float = 25.0

    # Default color palette
    color_palette: List[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.consensus_methods is None:
            self.consensus_methods = ["histogram",
                                      "corner_sampling", "center_analysis"]

        if self.color_palette is None:
            self.color_palette = [
                # Whites and grays
                (255, 255, 255),
                (248, 248, 248),
                (240, 240, 240),
                (230, 230, 230),
                (220, 220, 220),
                (200, 200, 200),
                (180, 180, 180),
                (160, 160, 160),
                (128, 128, 128),
                (96, 96, 96),
                (64, 64, 64),
                (32, 32, 32),
                (0, 0, 0),
                # UI Blues
                (0, 120, 215),
                (0, 103, 192),
                (16, 110, 190),
                (41, 128, 185),
                (52, 152, 219),
                (74, 144, 226),
                (106, 137, 204),
                # UI Greens (for your button)
                (46, 125, 50),
                (76, 175, 80),
                (67, 160, 71),
                (56, 142, 60),
                (34, 139, 34),
                (0, 128, 0),
                (50, 205, 50),
                (0, 100, 0),
                (39, 174, 96),
                (46, 204, 113),
                (22, 160, 133),
                # Additional UI colors
                (244, 67, 54),
                (233, 30, 99),
                (156, 39, 176),
                (103, 58, 183),
                (255, 193, 7),
                (255, 152, 0),
                (255, 87, 34),
                (121, 85, 72),
            ]


class UIColorExtractor:
    """Advanced UI color extractor using multiple robust methods"""

    def __init__(self, config: ColorExtractionConfig):
        self.config = config

    def extract_patch_colors(
        self, image: np.ndarray, x: int, y: int, width: int, height: int
    ) -> Dict[str, Tuple[int, int, int]]:
        """Extract colors using multiple robust methods"""

        # Extract patch
        patch = image[y: y + height, x: x + width]
        if patch.size == 0:
            return self._get_default_colors()

        # Convert to RGB if needed
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            rgb_patch = patch.copy()
        else:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        # Apply multiple extraction methods
        results = {}

        if "histogram" in self.config.consensus_methods:
            results["histogram"] = self._extract_by_histogram_mode(rgb_patch)

        if "corner_sampling" in self.config.consensus_methods:
            results["corner_sampling"] = self._extract_by_corner_sampling(
                rgb_patch)

        if "center_analysis" in self.config.consensus_methods:
            results["center_analysis"] = self._extract_by_center_analysis(
                rgb_patch)

        if self.config.enable_edge_masking:
            results["edge_masked"] = self._extract_with_edge_masking(rgb_patch)

        # Get consensus result
        if self.config.enable_consensus and len(results) > 1:
            final_colors = self._get_consensus_colors(results)
        else:
            # Use the first available method
            final_colors = list(results.values())[0]

        # Apply semantic classification
        if self.config.use_semantic_classification:
            final_colors = self._apply_semantic_classification(
                rgb_patch, final_colors)

        # Map to palette
        final_colors["background"] = self._map_to_palette(
            final_colors["background"])
        final_colors["foreground"] = self._map_to_palette(
            final_colors["foreground"])

        return final_colors

    def _extract_by_histogram_mode(
        self, patch: np.ndarray
    ) -> Dict[str, Tuple[int, int, int]]:
        """Extract colors using histogram mode (most frequent colors)"""

        # Convert to chosen color space for better separation
        if self.config.primary_color_space == "hsv":
            patch_converted = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        elif self.config.primary_color_space == "lab":
            patch_converted = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
        else:
            patch_converted = patch

        # Reduce color space by quantization
        patch_quantized = self._quantize_colors(
            patch_converted, self.config.histogram_bins
        )

        # Convert back to RGB for counting
        if self.config.primary_color_space == "hsv":
            patch_rgb = cv2.cvtColor(patch_quantized, cv2.COLOR_HSV2RGB)
        elif self.config.primary_color_space == "lab":
            patch_rgb = cv2.cvtColor(patch_quantized, cv2.COLOR_LAB2RGB)
        else:
            patch_rgb = patch_quantized

        # Count color frequencies
        pixels = patch_rgb.reshape(-1, 3)
        color_counts = Counter(tuple(pixel) for pixel in pixels)

        # Filter by minimum area percentage
        total_pixels = len(pixels)
        significant_colors = []

        for color, count in color_counts.most_common():
            percentage = count / total_pixels
            if percentage >= self.config.min_color_area_percentage:
                significant_colors.append((color, percentage))

        # Ensure we have at least 2 colors
        if len(significant_colors) < 2:
            significant_colors = [
                (color, count / total_pixels)
                for color, count in color_counts.most_common(2)
            ]

        # Return top 2 colors
        bg_color = significant_colors[0][0]  # Most frequent
        fg_color = (
            significant_colors[1][0] if len(
                significant_colors) > 1 else (0, 0, 0)
        )

        return {"background": bg_color, "foreground": fg_color}

    def _extract_by_corner_sampling(
        self, patch: np.ndarray
    ) -> Dict[str, Tuple[int, int, int]]:
        """Extract background from corners, foreground from center"""
        h, w = patch.shape[:2]
        corner_size = self.config.corner_sample_size

        # Sample corner regions (likely background)
        corners = [
            patch[0:corner_size, 0:corner_size],  # Top-left
            patch[0:corner_size, w - corner_size: w],  # Top-right
            patch[h - corner_size: h, 0:corner_size],  # Bottom-left
            patch[h - corner_size: h, w - corner_size: w],  # Bottom-right
        ]

        # Get most common color from corners
        corner_pixels = np.concatenate(
            [corner.reshape(-1, 3) for corner in corners])
        corner_colors = Counter(tuple(pixel) for pixel in corner_pixels)
        bg_color = corner_colors.most_common(1)[0][0]

        # Sample center region (likely foreground/text)
        center_h, center_w = h // 4, w // 4
        center_region = patch[center_h: h - center_h, center_w: w - center_w]

        if center_region.size > 0:
            center_pixels = center_region.reshape(-1, 3)
            center_colors = Counter(tuple(pixel) for pixel in center_pixels)

            # Find most common color that's different from background
            fg_color = bg_color
            for color, _ in center_colors.most_common():
                if self._color_distance(color, bg_color) > 30:  # Different enough
                    fg_color = color
                    break
        else:
            fg_color = (0, 0, 0)

        return {"background": bg_color, "foreground": fg_color}

    def _extract_by_center_analysis(
        self, patch: np.ndarray
    ) -> Dict[str, Tuple[int, int, int]]:
        """Weight center pixels more for background, edges for foreground"""
        h, w = patch.shape[:2]

        # Create weight matrix (higher weight for center)
        y_coords, x_coords = np.ogrid[0:h, 0:w]
        center_y, center_x = h // 2, w // 2

        # Distance from center (normalized)
        distances = np.sqrt((y_coords - center_y) ** 2 +
                            (x_coords - center_x) ** 2)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        normalized_distances = distances / max_distance

        # Weight matrix (higher weight for center)
        weights = self.config.center_weight_factor * (1 - normalized_distances)

        # Weighted color counting
        pixels = patch.reshape(-1, 3)
        weights_flat = weights.flatten()

        color_weights = {}
        for pixel, weight in zip(pixels, weights_flat):
            color = tuple(pixel)
            color_weights[color] = color_weights.get(color, 0) + weight

        # Sort by weighted frequency
        sorted_colors = sorted(color_weights.items(),
                               key=lambda x: x[1], reverse=True)

        bg_color = sorted_colors[0][0]
        fg_color = sorted_colors[1][0] if len(sorted_colors) > 1 else (0, 0, 0)

        return {"background": bg_color, "foreground": fg_color}

    def _extract_with_edge_masking(
        self, patch: np.ndarray
    ) -> Dict[str, Tuple[int, int, int]]:
        """Extract colors while masking out anti-aliased edges"""

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # Detect edges
        if self.config.edge_detection_method == "canny":
            edges = cv2.Canny(
                gray, self.config.canny_low_threshold, self.config.canny_high_threshold
            )
        elif self.config.edge_detection_method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges > edges.mean()).astype(np.uint8) * 255
        else:  # laplacian
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.abs(edges)
            edges = (edges > edges.mean()).astype(np.uint8) * 255

        # Dilate edges to remove more anti-aliasing
        kernel = np.ones(
            (
                self.config.edge_dilation_kernel_size,
                self.config.edge_dilation_kernel_size,
            ),
            np.uint8,
        )
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Create mask (True for non-edge pixels)
        mask = edges_dilated == 0

        # Extract colors from non-edge pixels only
        masked_pixels = patch[mask]

        if len(masked_pixels) == 0:
            return self._get_default_colors()

        # Count colors in masked region
        color_counts = Counter(tuple(pixel) for pixel in masked_pixels)

        # Get top colors
        top_colors = color_counts.most_common(2)
        bg_color = top_colors[0][0]
        fg_color = top_colors[1][0] if len(top_colors) > 1 else (0, 0, 0)

        return {"background": bg_color, "foreground": fg_color}

    def _get_consensus_colors(self, results: Dict) -> Dict[str, Tuple[int, int, int]]:
        """Get consensus from multiple extraction methods"""

        # Collect all background and foreground colors
        bg_colors = [result["background"] for result in results.values()]
        fg_colors = [result["foreground"] for result in results.values()]

        # Find consensus background (most similar colors)
        bg_consensus = self._find_color_consensus(bg_colors)
        fg_consensus = self._find_color_consensus(fg_colors)

        return {"background": bg_consensus, "foreground": fg_consensus}

    def _find_color_consensus(
        self, colors: List[Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        """Find consensus color from a list of colors"""
        if not colors:
            return (128, 128, 128)

        if len(colors) == 1:
            return colors[0]

        # Group similar colors
        color_groups = []
        for color in colors:
            # Find if this color belongs to existing group
            added_to_group = False
            for group in color_groups:
                if any(
                    self._color_distance(color, group_color) < 30
                    for group_color in group
                ):
                    group.append(color)
                    added_to_group = True
                    break

            if not added_to_group:
                color_groups.append([color])

        # Find largest group
        largest_group = max(color_groups, key=len)

        # Return average of largest group
        avg_r = int(np.mean([c[0] for c in largest_group]))
        avg_g = int(np.mean([c[1] for c in largest_group]))
        avg_b = int(np.mean([c[2] for c in largest_group]))

        return (avg_r, avg_g, avg_b)

    def _apply_semantic_classification(
        self, patch: np.ndarray, colors: Dict
    ) -> Dict[str, Tuple[int, int, int]]:
        """Apply semantic rules for better bg/fg classification"""

        bg_color = colors["background"]
        fg_color = colors["foreground"]

        # Calculate area coverage for each color
        pixels = patch.reshape(-1, 3)
        total_pixels = len(pixels)

        bg_pixels = sum(
            1 for pixel in pixels if self._color_distance(tuple(pixel), bg_color) < 15
        )
        fg_pixels = sum(
            1 for pixel in pixels if self._color_distance(tuple(pixel), fg_color) < 15
        )

        bg_coverage = bg_pixels / total_pixels
        fg_coverage = fg_pixels / total_pixels

        # Rule 1: Background should typically cover more area
        if fg_coverage > bg_coverage * 1.5:  # Foreground covers much more
            bg_color, fg_color = fg_color, bg_color  # Swap

        # Rule 2: In UI, text is usually darker OR much more colorful than background
        bg_brightness = sum(bg_color) / 3
        fg_brightness = sum(fg_color) / 3

        # Calculate color saturation
        bg_saturation = self._get_saturation(bg_color)
        fg_saturation = self._get_saturation(fg_color)

        # If supposed foreground is much brighter and less saturated, it might be background
        if (
            fg_brightness > bg_brightness + 50
            and fg_saturation < bg_saturation
            and bg_saturation > 0.3
        ):  # Background is colorful (like your green button)
            bg_color, fg_color = fg_color, bg_color  # Swap

        return {"background": bg_color, "foreground": fg_color}

    def _quantize_colors(self, image: np.ndarray, bins: int) -> np.ndarray:
        """Reduce color space by quantization"""
        quantized = np.floor(image / (256 / bins)) * (256 / bins)
        return quantized.astype(np.uint8)

    def _color_distance(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]
    ) -> float:
        """Calculate distance between two colors"""
        if self.config.color_distance_metric == "delta_e":
            return self._delta_e_distance(color1, color2)
        elif self.config.color_distance_metric == "hsv_weighted":
            return self._hsv_weighted_distance(color1, color2)
        else:  # euclidean
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

    def _delta_e_distance(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]
    ) -> float:
        """Calculate Delta E color difference (perceptually uniform)"""
        # Simple Delta E approximation
        r1, g1, b1 = color1
        r2, g2, b2 = color2

        # Weighted Euclidean distance that approximates Delta E
        delta_r = r1 - r2
        delta_g = g1 - g2
        delta_b = b1 - b2

        return np.sqrt(2 * delta_r**2 + 4 * delta_g**2 + 3 * delta_b**2)

    def _hsv_weighted_distance(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]
    ) -> float:
        """Calculate HSV-weighted distance"""
        # Convert to HSV
        hsv1 = colorsys.rgb_to_hsv(
            color1[0] / 255, color1[1] / 255, color1[2] / 255)
        hsv2 = colorsys.rgb_to_hsv(
            color2[0] / 255, color2[1] / 255, color2[2] / 255)

        # Weighted difference
        h_diff = (
            min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0])) * 360
        )  # Hue is circular
        s_diff = abs(hsv1[1] - hsv2[1]) * 100
        v_diff = abs(hsv1[2] - hsv2[2]) * 100

        return np.sqrt(2 * h_diff**2 + s_diff**2 + v_diff**2)

    def _get_saturation(self, color: Tuple[int, int, int]) -> float:
        """Get color saturation"""
        r, g, b = [c / 255.0 for c in color]
        max_val = max(r, g, b)
        min_val = min(r, g, b)

        if max_val == 0:
            return 0

        return (max_val - min_val) / max_val

    def _map_to_palette(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Map color to closest palette color using configured distance metric"""
        if not self.config.color_palette:
            return color

        min_distance = float("inf")
        closest_color = color

        for palette_color in self.config.color_palette:
            distance = self._color_distance(color, palette_color)

            if distance < min_distance:
                min_distance = distance
                closest_color = palette_color

        # Only use palette color if close enough
        if min_distance <= self.config.max_color_distance:
            return closest_color
        else:
            return color

    def _get_default_colors(self) -> Dict[str, Tuple[int, int, int]]:
        return {"background": (240, 240, 240), "foreground": (0, 0, 0)}


def create_sample_ui_image(width=400, height=300):
    """Create a sample UI image to test color extraction"""
    # Create a sample UI with a green button
    img = Image.new("RGB", (width, height), color="#f0f0f0")
    draw = ImageDraw.Draw(img)

    # Draw a green button
    button_x, button_y = 50, 100
    button_w, button_h = 150, 50
    draw.rectangle(
        [button_x, button_y, button_x + button_w, button_y + button_h],
        fill="#2E7D32",
        outline="#1B5E20",
    )

    # Add button text
    try:
        # Try to load a font, fallback to default if not available
        font = ImageFont.load_default()
    except:
        font = None

    text = "Click Me"
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    else:
        text_w, text_h = 50, 10  # Rough estimate

    text_x = button_x + (button_w - text_w) // 2
    text_y = button_y + (button_h - text_h) // 2
    draw.text((text_x, text_y), text, fill="white", font=font)

    # Add some other UI elements
    draw.rectangle([250, 50, 350, 80], fill="white", outline="#cccccc")
    draw.text((255, 58), "Text Field", fill="#333333", font=font)

    draw.rectangle([250, 100, 350, 130], fill="#1976D2", outline="#0D47A1")
    draw.text((275, 108), "Blue Btn", fill="white", font=font)

    return np.array(img)


def generate_demo_image(bg_color, fg_color, original_patch=None):
    """Generate a demo image showing the extracted colors"""
    demo_width, demo_height = 600, 400

    # Create the demo image
    demo_img = Image.new("RGB", (demo_width, demo_height), color="white")
    draw = ImageDraw.Draw(demo_img)

    # Draw background color swatch
    bg_rect = [50, 50, 200, 150]
    draw.rectangle(bg_rect, fill=bg_color, outline="black")
    draw.text((50, 160), f"Background: RGB{bg_color}", fill="black")

    # Draw foreground color swatch
    fg_rect = [250, 50, 400, 150]
    draw.rectangle(fg_rect, fill=fg_color, outline="black")
    draw.text((250, 160), f"Foreground: RGB{fg_color}", fill="black")

    # Show original patch if provided
    if original_patch is not None:
        patch_pil = Image.fromarray(original_patch)
        # Resize patch to fit
        patch_resized = patch_pil.resize((150, 100), Image.Resampling.NEAREST)
        demo_img.paste(patch_resized, (450, 50))
        draw.text((450, 160), "Original Patch", fill="black")

    # Create a sample UI preview with extracted colors
    preview_y = 220
    draw.rectangle(
        [50, preview_y, 550, preview_y + 150], fill=bg_color, outline="black"
    )

    # Sample button with foreground color
    btn_rect = [100, preview_y + 30, 250, preview_y + 70]
    draw.rectangle(btn_rect, fill=fg_color, outline="black")

    # Button text (use opposite color for visibility)
    text_color = (255, 255, 255) if sum(fg_color) < 384 else (0, 0, 0)
    draw.text((150, preview_y + 45), "Sample Button", fill=text_color)

    # Sample text with foreground color
    draw.text((300, preview_y + 45), "Sample Text", fill=fg_color)

    draw.text((50, preview_y - 20),
              "UI Preview with Extracted Colors:", fill="black")

    return demo_img


def process_color_extraction(uploaded_image, json_config_str, patch_coords):
    """Process color extraction with uploaded image and config"""
    try:
        # Parse JSON config
        if json_config_str.strip():
            config_dict = json.loads(json_config_str)
            # Convert color palette tuples from lists
            if "color_palette" in config_dict and config_dict["color_palette"]:
                config_dict["color_palette"] = [
                    tuple(c) for c in config_dict["color_palette"]
                ]
            config = ColorExtractionConfig(**config_dict)
        else:
            config = ColorExtractionConfig()

        # Use uploaded image or create sample
        if uploaded_image is not None:
            if isinstance(uploaded_image, str):
                # File path
                image_pil = Image.open(uploaded_image)
            else:
                # PIL Image or numpy array
                image_pil = (
                    uploaded_image
                    if isinstance(uploaded_image, Image.Image)
                    else Image.fromarray(uploaded_image)
                )

            # Convert to RGB if needed
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
            image_array = np.array(image_pil)
        else:
            # Create sample UI image
            image_array = create_sample_ui_image()
            image_pil = Image.fromarray(image_array)

        # Parse patch coordinates
        if patch_coords.strip():
            try:
                coords = [int(x.strip()) for x in patch_coords.split(",")]
                if len(coords) == 4:
                    x, y, width, height = coords
                else:
                    raise ValueError("Need exactly 4 coordinates")
            except:
                # Default to green button area in sample image
                x, y, width, height = 50, 100, 150, 50
        else:
            # Default patch coordinates (green button in sample)
            x, y, width, height = 50, 100, 150, 50

        # Ensure coordinates are within image bounds
        img_height, img_width = image_array.shape[:2]
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))

        # Extract colors
        extractor = UIColorExtractor(config)
        colors = extractor.extract_patch_colors(
            image_array, x, y, width, height)

        # Get the patch for display
        patch = image_array[y: y + height, x: x + width]

        # Generate demo image
        demo_img = generate_demo_image(
            colors["background"], colors["foreground"], patch
        )

        # Create result text
        result_text = f"""
Color Extraction Results:
#{colors['background'][0]:02x}{colors['background'][1]:02x}{colors['background'][2]:02x}
Background Color: RGB{colors['background']} | Hex:
#{colors['foreground'][0]:02x}{colors['foreground'][1]:02x}{colors['foreground'][2]:02x}
Foreground Color: RGB{colors['foreground']} | Hex:

Patch Coordinates: x={x}, y={y}, width={width}, height={height}
        """

        return demo_img, result_text, image_pil

    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        # Return a simple error image
        error_img = Image.new("RGB", (400, 200), color="#ffcccc")
        draw = ImageDraw.Draw(error_img)
        draw.text((10, 10), f"Error occurred:\n{str(e)}", fill="red")
        return error_img, error_msg, None


# Default configuration JSON
default_config = {
    "use_histogram_mode": True,
    "histogram_bins": 32,
    "min_color_area_percentage": 0.08,
    "enable_edge_masking": True,
    "edge_detection_method": "canny",
    "canny_low_threshold": 50,
    "canny_high_threshold": 150,
    "edge_dilation_kernel_size": 3,
    "use_semantic_classification": True,
    "center_weight_factor": 2.0,
    "corner_sample_size": 5,
    "primary_color_space": "hsv",
    "use_perceptual_distance": True,
    "enable_consensus": True,
    "consensus_methods": ["histogram", "corner_sampling", "center_analysis"],
    "color_distance_metric": "delta_e",
    "max_color_distance": 25.0,
}


class ColorExtractorGradioApp:
    """Gradio app wrapper for UIColorExtractor"""

    def __init__(self):
        self.default_config_json = json.dumps(default_config, indent=2)

    def create_interface(self):
        """Create the Gradio interface"""

        with gr.Blocks(title="UI Color Extractor", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# UI Color Extractor")
            gr.Markdown(
                "Upload an image and configure the color extraction settings to extract background and foreground colors from a specific patch."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Input controls
                    gr.Markdown("### Configuration")

                    config_input = gr.Textbox(
                        label="JSON Configuration",
                        value=self.default_config_json,
                        lines=20,
                        placeholder="Enter JSON configuration for ColorExtractionConfig...",
                    )

                    image_input = gr.Image(
                        label="Upload Image (optional - will use sample if not provided)",
                        type="pil",
                    )

                    coords_input = gr.Textbox(
                        label="Patch Coordinates (x,y,width,height)",
                        value="50,100,150,50",
                        placeholder="x,y,width,height (e.g., 50,100,150,50)",
                    )

                    extract_btn = gr.Button(
                        "Extract Colors", variant="primary")

                    # Preset buttons
                    gr.Markdown("### Quick Presets")
                    with gr.Row():
                        preset_default_btn = gr.Button(
                            "Default Config", size="sm")
                        preset_edge_btn = gr.Button("Edge Focus", size="sm")
                        preset_histogram_btn = gr.Button(
                            "Histogram Only", size="sm")

                with gr.Column(scale=2):
                    # Output area
                    gr.Markdown("### Results")

                    demo_output = gr.Image(
                        label="Color Extraction Demo", interactive=False
                    )

                    result_text = gr.Textbox(
                        label="Extraction Results", lines=6, interactive=False
                    )

                    original_image_output = gr.Image(
                        label="Original Image Used", interactive=False
                    )

            # Event handlers
            extract_btn.click(
                fn=self.process_extraction,
                inputs=[image_input, config_input, coords_input],
                outputs=[demo_output, result_text, original_image_output],
            )

            # Preset configurations
            preset_default_btn.click(
                fn=lambda: self.get_preset_config("default"), outputs=config_input
            )

            preset_edge_btn.click(
                fn=lambda: self.get_preset_config("edge_focus"), outputs=config_input
            )

            preset_histogram_btn.click(
                fn=lambda: self.get_preset_config("histogram_only"),
                outputs=config_input,
            )

            # Examples
            gr.Markdown("### Usage Examples")
            gr.Markdown(
                """
            **Coordinates Format:** `x,y,width,height` where:
            - `x,y` is the top-left corner of the patch
            - `width,height` is the size of the patch

            **Sample Coordinates for Default UI:**
            - Green Button: `50,100,150,50`
            - Blue Button: `250,100,100,30`
            - Text Field: `250,50,100,30`
            """
            )

        return interface

    def process_extraction(self, uploaded_image, json_config_str, patch_coords):
        """Wrapper for the color extraction process"""
        return process_color_extraction(uploaded_image, json_config_str, patch_coords)

    def get_preset_config(self, preset_name):
        """Get preset configurations"""
        presets = {
            "default": default_config,
            "edge_focus": {
                **default_config,
                "enable_edge_masking": True,
                "edge_detection_method": "canny",
                "edge_dilation_kernel_size": 5,
                "consensus_methods": ["edge_masked", "corner_sampling"],
            },
            "histogram_only": {
                **default_config,
                "enable_consensus": False,
                "consensus_methods": ["histogram"],
                "histogram_bins": 16,
                "min_color_area_percentage": 0.05,
            },
        }

        return json.dumps(presets.get(preset_name, default_config), indent=2)


def main():
    """Main function to launch the Gradio app"""
    app = ColorExtractorGradioApp()
    interface = app.create_interface()

    # Launch the interface
    interface.launch(share=False, debug=True, show_error=True)


if __name__ == "__main__":
    main()
