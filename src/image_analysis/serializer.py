"""
Serialization utilities for bounding boxes and clustering results.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional


class BoxSerializer:
    """
    Utility class for serializing bounding boxes to various formats.
    """
    
    @staticmethod
    def boxes_to_json_format(boxes: List[Tuple[float, float, float, float]], 
                           label: str = "box") -> List[List[Union[float, str]]]:
        """
        Convert bounding boxes to the specified JSON format.
        
        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
            label: Label to append to each box (default: "box")
            
        Returns:
            List of boxes in format [x, y, width, height, label]
        """
        json_boxes = []
        for box in boxes:
            x, y, width, height = box
            json_box = [float(x), float(y), float(width), float(height), label]
            json_boxes.append(json_box)
        
        return json_boxes
    
    @staticmethod
    def save_boxes_to_json(boxes: List[Tuple[float, float, float, float]], 
                          output_path: Union[str, Path], 
                          label: str = "box") -> None:
        """
        Save bounding boxes to a JSON file.
        
        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
            output_path: Path to save the JSON file
            label: Label to append to each box
        """
        json_boxes = BoxSerializer.boxes_to_json_format(boxes, label)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_boxes, f, indent=2)
    
    @staticmethod
    def save_clustering_results(
        original_boxes: List[Tuple[float, float, float, float]],
        clustered_boxes: List[List[Tuple[float, float, float, float]]],
        merged_boxes: List[Tuple[float, float, float, float]],
        output_dir: Union[str, Path],
        method_name: str,
        cluster_labels: Any = None,
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Save comprehensive clustering results to multiple JSON files.
        
        Args:
            original_boxes: Original bounding boxes
            clustered_boxes: Boxes grouped by clusters
            merged_boxes: Final merged boxes
            output_dir: Directory to save results
            method_name: Name of the clustering method used
            cluster_labels: Cluster labels from clustering algorithm
            quality_metrics: Quality metrics from clustering
            
        Returns:
            Dictionary mapping file types to their saved paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save merged boxes
        merged_file = output_dir / f"{method_name}_merged_boxes.json"
        BoxSerializer.save_boxes_to_json(merged_boxes, merged_file)
        saved_files["merged_boxes"] = merged_file
        
        # Save original boxes
        original_file = output_dir / f"{method_name}_original_boxes.json"
        BoxSerializer.save_boxes_to_json(original_boxes, original_file)
        saved_files["original_boxes"] = original_file
        
        # Save clustering details
        clustering_details = {
            "method": method_name,
            "original_count": len(original_boxes),
            "cluster_count": len(clustered_boxes),
            "merged_count": len(merged_boxes),
            "clusters": []
        }
        
        for i, cluster in enumerate(clustered_boxes):
            cluster_data = {
                "cluster_id": i,
                "size": len(cluster),
                "boxes": [list(box) for box in cluster]
            }
            clustering_details["clusters"].append(cluster_data)
        
        if cluster_labels is not None:
            clustering_details["cluster_labels"] = cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else cluster_labels
        
        if quality_metrics is not None:
            clustering_details["quality_metrics"] = quality_metrics
        
        details_file = output_dir / f"{method_name}_clustering_details.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(clustering_details, f, indent=2)
        saved_files["clustering_details"] = details_file
        
        return saved_files
    
    @staticmethod
    def load_boxes_from_json(file_path: Union[str, Path]) -> List[Tuple[float, float, float, float]]:
        """
        Load bounding boxes from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        boxes = []
        for item in data:
            if isinstance(item, list) and len(item) >= 4:
                x, y, width, height = item[:4]
                boxes.append((float(x), float(y), float(width), float(height)))
        
        return boxes 