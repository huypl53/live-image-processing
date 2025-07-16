"""
Enhanced clustering for bounding boxes with color information.
"""

import json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image


class ColorBoundingBoxClusterer:
    """
    Enhanced hierarchical clustering for bounding boxes with color information.
    
    This class performs clustering on bounding boxes using various distance metrics
    including spatial, size, and color similarity for real-world UI components.
    """

    def __init__(self, 
                 boxes: Optional[List[Tuple[float, float, float, float]]] = None,
                 colors: Optional[List[Tuple[int, int, int]]] = None,
                 image_path: Optional[str] = None):
        """
        Initialize the clusterer with bounding boxes and color information.

        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
            colors: List of dominant colors as (R, G, B) tuples
            image_path: Path to the source image for color extraction
        """
        self.boxes = []
        self.colors = []
        self.image_path = image_path
        
        if boxes is not None:
            self.add_boxes(boxes, colors)

    def add_boxes(self, 
                  boxes: List[Tuple[float, float, float, float]], 
                  colors: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """
        Add bounding boxes and their colors to the clusterer.

        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
            colors: List of dominant colors as (R, G, B) tuples
        """
        self.boxes.extend(boxes)
        
        if colors is not None:
            self.colors.extend(colors)
        else:
            # Extract colors from image if available
            if self.image_path and len(self.colors) == 0:
                self._extract_colors_from_image()

    def _extract_colors_from_image(self) -> None:
        """
        Extract dominant colors from the source image for each bounding box.
        """
        if not self.image_path or not Path(self.image_path).exists():
            print(f"Warning: Image not found at {self.image_path}")
            return
        
        try:
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                print(f"Warning: Could not load image at {self.image_path}")
                return
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract colors for each bounding box
            for box in self.boxes:
                x, y, width, height = box
                
                # Ensure coordinates are within image bounds
                x = max(0, int(x))
                y = max(0, int(y))
                width = min(width, image_rgb.shape[1] - x)
                height = min(height, image_rgb.shape[0] - y)
                
                if width > 0 and height > 0:
                    # Extract region
                    region = image_rgb[y:y+height, x:x+width]
                    
                    # Calculate dominant color (mean of the region)
                    dominant_color = np.mean(region, axis=(0, 1)).astype(int)
                    self.colors.append(tuple(dominant_color))
                else:
                    # Default color for invalid regions
                    self.colors.append((128, 128, 128))
                    
        except Exception as e:
            print(f"Error extracting colors: {e}")
            # Use default colors
            self.colors = [(128, 128, 128)] * len(self.boxes)

    def _calculate_color_distance(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """
        Calculate color distance using LAB color space for better perceptual similarity.
        
        Args:
            color1, color2: Colors as (R, G, B) tuples
            
        Returns:
            Normalized color distance (0-1)
        """
        # Convert RGB to LAB color space for better perceptual distance
        color1_lab = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2LAB)[0, 0]
        color2_lab = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2LAB)[0, 0]
        
        # Calculate Euclidean distance in LAB space
        distance = np.sqrt(np.sum((color1_lab - color2_lab) ** 2))
        
        # Normalize to 0-1 range (max LAB distance is approximately 255*sqrt(3))
        max_distance = 255 * np.sqrt(3)
        return min(distance / max_distance, 1.0)

    def _calculate_color_similarity(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """
        Calculate color similarity (inverse of distance).
        
        Args:
            color1, color2: Colors as (R, G, B) tuples
            
        Returns:
            Color similarity (0-1, where 1 is identical)
        """
        distance = self._calculate_color_distance(color1, color2)
        return 1.0 - distance

    def _calculate_iou_min(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculate IoU using minimum area as denominator.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0  # No intersection

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2

        # Use minimum area as denominator
        min_area = min(area1, area2)

        if min_area == 0:
            return 0.0

        return intersection_area / min_area

    def _calculate_center_distance(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculate normalized center distance between two boxes.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate centers
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Normalize by the average size of the two boxes
        avg_size = (w1 + h1 + w2 + h2) / 4
        return distance / (avg_size + 1e-6)  # Avoid division by zero

    def _calculate_size_similarity(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculate size similarity between two boxes.
        """
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        
        # Calculate aspect ratios
        aspect1 = w1 / (h1 + 1e-6)
        aspect2 = w2 / (h2 + 1e-6)
        
        # Calculate area ratios
        area1 = w1 * h1
        area2 = w2 * h2
        area_ratio = min(area1, area2) / (max(area1, area2) + 1e-6)
        
        # Combine aspect ratio and area similarity
        aspect_diff = abs(aspect1 - aspect2) / (max(aspect1, aspect2) + 1e-6)
        size_similarity = area_ratio * (1 - aspect_diff)
        
        return size_similarity

    def _calculate_enhanced_distance_with_color(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate enhanced distance combining spatial, size, and color metrics.
        
        Args:
            box1, box2: Bounding boxes as (x, y, width, height)
            color1, color2: Colors as (R, G, B) tuples
            weights: Dictionary with weights for different metrics
                    {'iou': 0.2, 'center': 0.3, 'size': 0.2, 'color': 0.3}
        """
        if weights is None:
            weights = {'iou': 0.2, 'center': 0.3, 'size': 0.2, 'color': 0.3}
        
        # Calculate individual metrics
        iou_min = self._calculate_iou_min(box1, box2)
        center_dist = self._calculate_center_distance(box1, box2)
        size_sim = self._calculate_size_similarity(box1, box2)
        color_sim = self._calculate_color_similarity(color1, color2)
        
        # Combine metrics
        iou_distance = 1.0 - iou_min
        size_distance = 1.0 - size_sim
        color_distance = 1.0 - color_sim
        
        # Weighted combination
        total_distance = (
            weights.get('iou', 0.2) * iou_distance +
            weights.get('center', 0.3) * min(center_dist, 1.0) +
            weights.get('size', 0.2) * size_distance +
            weights.get('color', 0.3) * color_distance
        )
        
        return total_distance

    def _calculate_distance_matrix(self, 
                                 distance_type: str = 'enhanced_with_color', 
                                 weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Calculate the distance matrix between all pairs of bounding boxes.
        
        Args:
            distance_type: 'iou', 'center', 'size', 'color', 'enhanced', or 'enhanced_with_color'
            weights: Weights for enhanced distance metric
        """
        n_boxes = len(self.boxes)
        distance_matrix = np.zeros((n_boxes, n_boxes))

        for i in range(n_boxes):
            for j in range(i + 1, n_boxes):
                if distance_type == 'iou':
                    iou_min = self._calculate_iou_min(self.boxes[i], self.boxes[j])
                    distance = 1.0 - iou_min
                elif distance_type == 'center':
                    distance = self._calculate_center_distance(self.boxes[i], self.boxes[j])
                elif distance_type == 'size':
                    size_sim = self._calculate_size_similarity(self.boxes[i], self.boxes[j])
                    distance = 1.0 - size_sim
                elif distance_type == 'color':
                    if i < len(self.colors) and j < len(self.colors):
                        distance = self._calculate_color_distance(self.colors[i], self.colors[j])
                    else:
                        distance = 0.5  # Default distance if no color info
                elif distance_type == 'enhanced_with_color':
                    if i < len(self.colors) and j < len(self.colors):
                        distance = self._calculate_enhanced_distance_with_color(
                            self.boxes[i], self.boxes[j], 
                            self.colors[i], self.colors[j], 
                            weights
                        )
                    else:
                        # Fallback to enhanced without color
                        distance = self._calculate_enhanced_distance_with_color(
                            self.boxes[i], self.boxes[j], 
                            (128, 128, 128), (128, 128, 128), 
                            weights
                        )
                else:  # enhanced (without color)
                    distance = self._calculate_enhanced_distance_with_color(
                        self.boxes[i], self.boxes[j], 
                        (128, 128, 128), (128, 128, 128), 
                        weights
                    )
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric matrix

        return distance_matrix

    def cluster(
        self,
        method: str = "ward",
        distance_type: str = "enhanced_with_color",
        weights: Optional[Dict[str, float]] = None,
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        auto_threshold: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hierarchical clustering on the bounding boxes with color information.
        
        Args:
            method: Linkage method ('ward', 'complete', 'average', 'single')
            distance_type: Distance metric type
            weights: Weights for enhanced distance
            distance_threshold: Distance threshold for cutting the dendrogram
            n_clusters: Number of clusters to form
            auto_threshold: Automatically determine threshold based on data
            
        Returns:
            Tuple of (linkage_matrix, cluster_labels)
        """
        if len(self.boxes) < 2:
            raise ValueError("Need at least 2 boxes for clustering")

        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(distance_type, weights)
        
        # Print distance statistics for debugging
        non_zero_distances = distance_matrix[distance_matrix > 0]
        print(f"Distance matrix stats: min={non_zero_distances.min():.3f}, "
              f"max={non_zero_distances.max():.3f}, "
              f"mean={non_zero_distances.mean():.3f}")

        # Convert to condensed distance matrix (required by scipy)
        condensed_distances = squareform(distance_matrix)

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=method)

        # Determine clustering threshold
        if auto_threshold and distance_threshold is None and n_clusters is None:
            # Auto-determine threshold based on distance distribution
            distances = linkage_matrix[:, 2]  # Distances from linkage matrix
            if len(distances) > 0:
                # Use 75th percentile of distances as threshold
                distance_threshold = np.percentile(distances, 75)
                print(f"Auto-determined threshold: {distance_threshold:.3f}")
            else:
                distance_threshold = 0.5

        # Determine number of clusters
        if distance_threshold is not None:
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion="distance")
        elif n_clusters is not None:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        else:
            # Default: cut at distance 0.5
            cluster_labels = fcluster(linkage_matrix, 0.5, criterion="distance")

        return linkage_matrix, cluster_labels

    def cluster_dbscan(
        self,
        distance_type: str = "enhanced_with_color",
        weights: Optional[Dict[str, float]] = None,
        eps: float = 0.3,
        min_samples: int = 2,
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering as an alternative to hierarchical clustering.
        
        Args:
            distance_type: Distance metric type
            weights: Weights for enhanced distance
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Cluster labels
        """
        if len(self.boxes) < 2:
            raise ValueError("Need at least 2 boxes for clustering")

        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(distance_type, weights)
        
        # Use DBSCAN with precomputed distance matrix
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        return cluster_labels

    def get_cluster_boxes(
        self, cluster_labels: np.ndarray
    ) -> List[List[Tuple[float, float, float, float]]]:
        """
        Group bounding boxes by their cluster labels.
        """
        unique_labels = np.unique(cluster_labels)
        clustered_boxes = []

        for label in unique_labels:
            if label == -1:  # DBSCAN noise points
                continue
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_boxes = [self.boxes[i] for i in cluster_indices]
            clustered_boxes.append(cluster_boxes)

        return clustered_boxes

    def get_cluster_colors(
        self, cluster_labels: np.ndarray
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Group colors by their cluster labels.
        """
        unique_labels = np.unique(cluster_labels)
        clustered_colors = []

        for label in unique_labels:
            if label == -1:  # DBSCAN noise points
                continue
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_colors = [self.colors[i] for i in cluster_indices if i < len(self.colors)]
            clustered_colors.append(cluster_colors)

        return clustered_colors

    def plot_dendrogram(
        self,
        linkage_matrix: np.ndarray,
        distance_threshold: Optional[float] = None,
        title: str = "Bounding Box Clustering Dendrogram",
    ) -> None:
        """
        Plot the dendrogram of the hierarchical clustering.
        """
        plt.figure(figsize=(15, 10))

        # Create dendrogram
        dendrogram(
            linkage_matrix,
            labels=[f"Box_{i}" for i in range(len(self.boxes))],
            orientation="top",
            distance_sort=True,
            show_leaf_counts=True,
        )

        if distance_threshold is not None:
            plt.axhline(
                y=distance_threshold,
                color="r",
                linestyle="--",
                label=f"Distance threshold: {distance_threshold:.3f}",
            )
            plt.legend()

        plt.title(title)
        plt.xlabel("Bounding Box Index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig("dendrogram.png", dpi=300, bbox_inches='tight')
        plt.show()

    def get_cluster_statistics(self, cluster_labels: np.ndarray) -> dict:
        """
        Get statistics about the clustering results.
        """
        unique_labels = np.unique(cluster_labels)
        # Remove noise label (-1) for DBSCAN
        unique_labels = unique_labels[unique_labels != -1]
        
        stats = {
            "n_clusters": len(unique_labels),
            "n_boxes": len(self.boxes),
            "cluster_sizes": [],
            "cluster_details": {},
            "noise_points": len(cluster_labels[cluster_labels == -1]) if -1 in cluster_labels else 0,
        }

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_size = len(cluster_indices)
            stats["cluster_sizes"].append(cluster_size)
            stats["cluster_details"][f"cluster_{label}"] = {
                "size": cluster_size,
                "box_indices": cluster_indices.tolist(),
                "boxes": [self.boxes[i] for i in cluster_indices],
                "colors": [self.colors[i] for i in cluster_indices if i < len(self.colors)]
            }

        return stats

    def analyze_clustering_quality(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the quality of clustering results.
        """
        stats = self.get_cluster_statistics(cluster_labels)
        
        # Calculate intra-cluster distances
        intra_distances = []
        
        for label in stats["cluster_details"]:
            cluster_indices = stats["cluster_details"][label]["box_indices"]
            if len(cluster_indices) > 1:
                # Calculate average intra-cluster distance
                cluster_distances = []
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx1, idx2 = cluster_indices[i], cluster_indices[j]
                        if idx1 < len(self.colors) and idx2 < len(self.colors):
                            distance = self._calculate_enhanced_distance_with_color(
                                self.boxes[idx1], self.boxes[idx2],
                                self.colors[idx1], self.colors[idx2]
                            )
                        else:
                            distance = self._calculate_enhanced_distance_with_color(
                                self.boxes[idx1], self.boxes[idx2],
                                (128, 128, 128), (128, 128, 128)
                            )
                        cluster_distances.append(distance)
                intra_distances.extend(cluster_distances)
        
        quality_metrics = {
            "n_clusters": stats["n_clusters"],
            "avg_cluster_size": np.mean(stats["cluster_sizes"]) if stats["cluster_sizes"] else 0,
            "max_cluster_size": max(stats["cluster_sizes"]) if stats["cluster_sizes"] else 0,
            "min_cluster_size": min(stats["cluster_sizes"]) if stats["cluster_sizes"] else 0,
            "avg_intra_distance": np.mean(intra_distances) if intra_distances else 0,
            "max_intra_distance": np.max(intra_distances) if intra_distances else 0,
            "noise_points": stats["noise_points"],
        }
        
        return quality_metrics 