import json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class BoundingBoxClusterer:
    """
    Enhanced hierarchical clustering for bounding boxes with multiple distance metrics.
    
    This class performs clustering on bounding boxes using various distance metrics
    suitable for real-world UI components that may have minimal overlap.
    """

    def __init__(self, boxes: Optional[List[Tuple[float, float, float, float]]] = None):
        """
        Initialize the clusterer with bounding boxes.

        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples.
                  If None, boxes can be added later using add_boxes().
        """
        self.boxes = []
        if boxes is not None:
            self.add_boxes(boxes)

    def add_boxes(self, boxes: List[Tuple[float, float, float, float]]) -> None:
        """
        Add bounding boxes to the clusterer.

        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples.
        """
        self.boxes.extend(boxes)

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

    def _calculate_enhanced_distance(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate enhanced distance combining multiple metrics.
        
        Args:
            box1, box2: Bounding boxes as (x, y, width, height)
            weights: Dictionary with weights for different metrics
                    {'iou': 0.3, 'center': 0.4, 'size': 0.3}
        """
        if weights is None:
            weights = {'iou': 0.2, 'center': 0.5, 'size': 0.3}
        
        # Calculate individual metrics
        iou_min = self._calculate_iou_min(box1, box2)
        center_dist = self._calculate_center_distance(box1, box2)
        size_sim = self._calculate_size_similarity(box1, box2)
        
        # Combine metrics
        iou_distance = 1.0 - iou_min
        size_distance = 1.0 - size_sim
        
        # Weighted combination
        total_distance = (
            weights['iou'] * iou_distance +
            weights['center'] * min(center_dist, 1.0) +  # Cap center distance
            weights['size'] * size_distance
        )
        
        return total_distance

    def _calculate_distance_matrix(self, distance_type: str = 'enhanced', 
                                 weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Calculate the distance matrix between all pairs of bounding boxes.
        
        Args:
            distance_type: 'iou', 'center', 'size', or 'enhanced'
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
                else:  # enhanced
                    distance = self._calculate_enhanced_distance(self.boxes[i], self.boxes[j], weights)
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric matrix

        return distance_matrix

    def cluster(
        self,
        method: str = "ward",
        distance_type: str = "enhanced",
        weights: Optional[Dict[str, float]] = None,
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        auto_threshold: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hierarchical clustering on the bounding boxes.
        
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
        distance_type: str = "enhanced",
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
            }

        return stats

    def analyze_clustering_quality(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the quality of clustering results.
        """
        stats = self.get_cluster_statistics(cluster_labels)
        
        # Calculate intra-cluster distances
        intra_distances = []
        inter_distances = []
        
        for label in stats["cluster_details"]:
            cluster_indices = stats["cluster_details"][label]["box_indices"]
            if len(cluster_indices) > 1:
                # Calculate average intra-cluster distance
                cluster_distances = []
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx1, idx2 = cluster_indices[i], cluster_indices[j]
                        distance = self._calculate_enhanced_distance(
                            self.boxes[idx1], self.boxes[idx2]
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


# Example usage and testing
def example_usage():
    """
    Example usage of the modular clustering system.
    """
    import json
    from pathlib import Path
    from .clustering_manager import ClusteringManager
    
    # Load boxes from JSON file
    boxes_file = Path(__file__).parent.parent.parent / "examples" / "bboxes.json"
    boxes = json.load(boxes_file.open(encoding="utf-8"))
    
    # Extract bounding boxes
    sample_boxes = [
        tuple(box["bbox"].values()) for box in boxes["components"]
    ]

    # Create clustering manager
    manager = ClusteringManager(sample_boxes)
    
    # Set output directory
    output_dir = Path(__file__).parent.parent.parent / "clustering_results"
    
    print("=== Running Comprehensive Clustering Analysis ===")
    
    # Run comprehensive testing
    results = manager.run_comprehensive_testing(
        output_dir=output_dir,
        save_results=True
    )
    
    # Print summary of all methods
    print(f"\n=== Clustering Results Summary ===")
    print(f"Total methods tested: {len(results) - 1}")  # -1 for best_result
    
    for method_name, result in results.items():
        if method_name == "best_result":
            continue
            
        quality = result.get("quality_metrics", {})
        merge_stats = result.get("merge_statistics", {})
        
        print(f"\n{method_name}:")
        print(f"  Clusters: {quality.get('n_clusters', 0)}")
        print(f"  Avg cluster size: {quality.get('avg_cluster_size', 0):.1f}")
        print(f"  Reduction ratio: {merge_stats.get('reduction_ratio', 0):.2f}")
        print(f"  Avg intra-distance: {quality.get('avg_intra_distance', 0):.3f}")
    
    # Show best result
    best_result = results.get("best_result")
    if best_result:
        print(f"\n=== Best Result: {best_result['method']} ===")
        
        quality = best_result.get("quality_metrics", {})
        merge_stats = best_result.get("merge_statistics", {})
        
        print(f"Clusters: {quality.get('n_clusters', 0)}")
        print(f"Reduction ratio: {merge_stats.get('reduction_ratio', 0):.2f}")
        print(f"Avg intra-distance: {quality.get('avg_intra_distance', 0):.3f}")
        
        # Show merged boxes
        merged_boxes = best_result.get("merged_boxes", [])
        print(f"\nMerged boxes ({len(merged_boxes)}):")
        for i, box in enumerate(merged_boxes):
            print(f"  Box {i}: {box}")
        
        # Plot dendrogram if hierarchical
        if "linkage_matrix" in best_result:
            manager.plot_best_dendrogram(results)
    else:
        print("No successful clustering found!")


if __name__ == "__main__":
    example_usage()
