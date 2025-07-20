from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import cv2
from collections import Counter


@dataclass
class ColorExtractionConfig:
    """Configuration for UI color extraction"""

    # Color clustering parameters
    # Number of dominant colors to extract (bg + fg)
    n_dominant_colors: int = 2
    kmeans_max_iter: int = 20
    kmeans_n_init: int = 10

    # Color filtering
    min_color_percentage: float = 0.05  # Ignore colors below this threshold
    brightness_threshold: float = 0.1  # Threshold to separate dark/light colors

    # Palette matching
    color_distance_metric: str = "euclidean"  # 'euclidean' or 'cie_delta_e'
    max_color_distance: float = 50.0  # Maximum distance to consider a match

    # Image preprocessing
    blur_kernel_size: int = 3  # Gaussian blur to reduce noise
    resize_for_clustering: Optional[Tuple[int, int]] = (
        100,
        100,
    )  # Resize for faster clustering

    # Default color palette (Windows-like colors)
    color_palette: List[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                # Windows default colors
                (255, 255, 255),  # White
                (240, 240, 240),  # Light gray
                (225, 225, 225),  # Silver
                (192, 192, 192),  # Gray
                (128, 128, 128),  # Dark gray
                (64, 64, 64),  # Charcoal
                (0, 0, 0),  # Black
                (0, 120, 215),  # Blue
                (0, 103, 192),  # Dark blue
                (16, 110, 190),  # Medium blue
                (255, 255, 255),  # Control backgrounds
                (245, 245, 245),  # Button face
                (230, 230, 230),  # Button shadow
                (173, 173, 173),  # Button dark shadow
                (255, 0, 0),  # Red
                (0, 255, 0),  # Green
                (255, 255, 0),  # Yellow
            ]


class UIColorExtractor:
    """Extract and synthesize colors from UI element patches"""

    def __init__(self, config: ColorExtractionConfig):
        self.config = config

    def extract_patch_colors(
        self, image: np.ndarray, x: int, y: int, width: int, height: int
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Extract foreground and background colors from a UI element patch

        Args:
            image: Full screenshot as numpy array (H, W, C)
            x, y: Top-left coordinates of the patch
            width, height: Dimensions of the patch

        Returns:
            Dictionary with 'background' and 'foreground' color tuples (R, G, B)
        """
        # Extract patch
        patch = image[y: y + height, x: x + width]

        if patch.size == 0:
            return self._get_default_colors()

        # Preprocess patch
        processed_patch = self._preprocess_patch(patch)

        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(processed_patch)

        # Classify as background/foreground
        bg_color, fg_color = self._classify_bg_fg_colors(dominant_colors)

        # Map to palette
        bg_color = self._map_to_palette(bg_color)
        fg_color = self._map_to_palette(fg_color)

        return {"background": bg_color, "foreground": fg_color}

    def extract_batch_colors(
        self, image: np.ndarray, elements: List[Dict]
    ) -> List[Dict[str, Tuple[int, int, int]]]:
        """
        Extract colors for multiple UI elements

        Args:
            image: Full screenshot as numpy array
            elements: List of dicts with keys: x, y, width, height

        Returns:
            List of color dictionaries for each element
        """
        results = []
        for element in elements:
            colors = self.extract_patch_colors(
                image, element["x"], element["y"], element["width"], element["height"]
            )
            results.append(colors)
        return results

    def _preprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        """Preprocess patch for better color extraction"""
        # Convert to RGB if needed
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            patch_rgb = patch
        else:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        # Apply Gaussian blur to reduce noise
        if self.config.blur_kernel_size > 0:
            patch_rgb = cv2.GaussianBlur(
                patch_rgb,
                (self.config.blur_kernel_size, self.config.blur_kernel_size),
                0,
            )

        # Resize for faster clustering if specified
        if self.config.resize_for_clustering:
            patch_rgb = cv2.resize(
                patch_rgb, self.config.resize_for_clustering)

        return patch_rgb

    def _extract_dominant_colors(self, patch: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = patch.reshape(-1, 3)

        # Remove any invalid pixels
        valid_pixels = pixels[np.all(pixels >= 0, axis=1)]
        if len(valid_pixels) == 0:
            return [(128, 128, 128), (64, 64, 64)]  # Default colors

        # Perform K-means clustering
        n_colors = min(self.config.n_dominant_colors, len(valid_pixels))
        if n_colors < 2:
            # Not enough pixels, return default
            return [(128, 128, 128), (64, 64, 64)]

        kmeans = KMeans(
            n_clusters=n_colors,
            max_iter=self.config.kmeans_max_iter,
            n_init=self.config.kmeans_n_init,
            random_state=42,
        )

        kmeans.fit(valid_pixels)

        # Get cluster centers and their frequencies
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Calculate color frequencies
        color_counts = Counter(labels)
        total_pixels = len(labels)

        # Filter colors by minimum percentage
        dominant_colors = []
        for i, color in enumerate(colors):
            percentage = color_counts[i] / total_pixels
            if percentage >= self.config.min_color_percentage:
                dominant_colors.append(tuple(color))

        # Ensure we have at least 2 colors
        if len(dominant_colors) < 2:
            # Add most common colors
            sorted_colors = sorted(
                enumerate(colors), key=lambda x: color_counts[x[0]], reverse=True
            )
            dominant_colors = [tuple(color) for _, color in sorted_colors[:2]]

        return dominant_colors[: self.config.n_dominant_colors]

    def _classify_bg_fg_colors(
        self, colors: List[Tuple[int, int, int]]
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Classify colors as background (lighter) and foreground (darker)"""
        if len(colors) < 2:
            colors.extend([(128, 128, 128)] * (2 - len(colors)))

        # Calculate brightness (luminance) for each color
        brightness_scores = []
        for color in colors:
            r, g, b = color
            # Use standard luminance formula
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            brightness_scores.append((brightness, color))

        # Sort by brightness (brightest first)
        brightness_scores.sort(reverse=True)

        # Background is typically brighter, foreground darker
        bg_color = brightness_scores[0][1]
        fg_color = brightness_scores[-1][1]

        return bg_color, fg_color

    def _map_to_palette(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Map extracted color to closest color in predefined palette"""
        if not self.config.color_palette:
            return color

        min_distance = float("inf")
        closest_color = color

        r1, g1, b1 = color

        for palette_color in self.config.color_palette:
            r2, g2, b2 = palette_color

            if self.config.color_distance_metric == "euclidean":
                # Simple Euclidean distance in RGB space
                distance = np.sqrt((r1 - r2) ** 2 + (g1 - g2)
                                   ** 2 + (b1 - b2) ** 2)
            elif self.config.color_distance_metric == "cie_delta_e":
                # More perceptually accurate (requires colorspacious library)
                # For now, use weighted Euclidean as approximation
                distance = np.sqrt(
                    2 * (r1 - r2) ** 2 + 4 *
                    (g1 - g2) ** 2 + 3 * (b1 - b2) ** 2
                )
            else:
                distance = (
                    abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
                )  # Manhattan distance

            if distance < min_distance:
                min_distance = distance
                closest_color = palette_color

        # Only use palette color if it's close enough
        if min_distance <= self.config.max_color_distance:
            return closest_color
        else:
            return color

    def _get_default_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Return default colors when extraction fails"""
        return {
            "background": (240, 240, 240),  # Light gray
            "foreground": (0, 0, 0),  # Black
        }

    def update_palette(self, new_palette: List[Tuple[int, int, int]]):
        """Update the color palette"""
        self.config.color_palette = new_palette

    def add_palette_colors(self, colors: List[Tuple[int, int, int]]):
        """Add colors to existing palette"""
        self.config.color_palette.extend(colors)


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = ColorExtractionConfig(
        n_dominant_colors=2, min_color_percentage=0.05, max_color_distance=30.0
    )

    # Initialize extractor
    extractor = UIColorExtractor(config)

    # Example: Extract colors from a button
    # image = cv2.imread('winform_screenshot.png')
    # colors = extractor.extract_patch_colors(image, x=100, y=50, width=80, height=30)
    # print(f"Button colors: {colors}")

    # Example: Extract colors for multiple elements
    # elements = [
    #     {'x': 100, 'y': 50, 'width': 80, 'height': 30},    # Button
    #     {'x': 200, 'y': 100, 'width': 150, 'height': 20},  # Label
    #     {'x': 50, 'y': 200, 'width': 200, 'height': 100}   # TextArea
    # ]
    # batch_colors = extractor.extract_batch_colors(image, elements)
    # for i, colors in enumerate(batch_colors):
    #     print(f"Element {i}: {colors}")
