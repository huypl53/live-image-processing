"""
Bounding box merging utilities for clustering results.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class BoxMerger:
    """
    Utility class for merging bounding boxes within clusters.
    """
    
    @staticmethod
    def merge_boxes(boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """
        Merge multiple bounding boxes into a single bounding box.
        
        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
            
        Returns:
            Merged bounding box as (x, y, width, height) tuple
        """
        if not boxes:
            raise ValueError("Cannot merge empty list of boxes")
        
        if len(boxes) == 1:
            return boxes[0]
        
        # Calculate the bounding box that encompasses all boxes
        x_coords = [box[0] for box in boxes]
        y_coords = [box[1] for box in boxes]
        x_right_coords = [box[0] + box[2] for box in boxes]
        y_bottom_coords = [box[1] + box[3] for box in boxes]
        
        # Find the minimum and maximum coordinates
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_right_coords)
        y_max = max(y_bottom_coords)
        
        # Calculate the merged box dimensions
        merged_x = x_min
        merged_y = y_min
        merged_width = x_max - x_min
        merged_height = y_max - y_min
        
        return (merged_x, merged_y, merged_width, merged_height)
    
    @staticmethod
    def merge_clusters(clustered_boxes: List[List[Tuple[float, float, float, float]]]) -> List[Tuple[float, float, float, float]]:
        """
        Merge all clusters of boxes into individual bounding boxes.
        
        Args:
            clustered_boxes: List of clusters, where each cluster is a list of boxes
            
        Returns:
            List of merged bounding boxes, one per cluster
        """
        merged_boxes = []
        
        for cluster in clustered_boxes:
            if cluster:  # Skip empty clusters
                merged_box = BoxMerger.merge_boxes(cluster)
                merged_boxes.append(merged_box)
        
        return merged_boxes
    
    @staticmethod
    def calculate_merge_statistics(
        original_boxes: List[Tuple[float, float, float, float]],
        clustered_boxes: List[List[Tuple[float, float, float, float]]],
        merged_boxes: List[Tuple[float, float, float, float]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about the merging process.
        
        Args:
            original_boxes: Original bounding boxes
            clustered_boxes: Boxes grouped by clusters
            merged_boxes: Final merged boxes
            
        Returns:
            Dictionary with merging statistics
        """
        stats = {
            "original_count": len(original_boxes),
            "cluster_count": len(clustered_boxes),
            "merged_count": len(merged_boxes),
            "reduction_ratio": len(original_boxes) / len(merged_boxes) if merged_boxes else 0,
            "cluster_sizes": [len(cluster) for cluster in clustered_boxes],
            "merged_areas": [],
            "original_areas": []
        }
        
        # Calculate areas
        for box in original_boxes:
            stats["original_areas"].append(box[2] * box[3])
        
        for box in merged_boxes:
            stats["merged_areas"].append(box[2] * box[3])
        
        # Calculate area statistics
        if stats["original_areas"]:
            stats["original_total_area"] = sum(stats["original_areas"])
            stats["original_avg_area"] = np.mean(stats["original_areas"])
        
        if stats["merged_areas"]:
            stats["merged_total_area"] = sum(stats["merged_areas"])
            stats["merged_avg_area"] = np.mean(stats["merged_areas"])
            stats["area_coverage_ratio"] = stats["merged_total_area"] / stats["original_total_area"] if stats["original_total_area"] > 0 else 0
        
        return stats 