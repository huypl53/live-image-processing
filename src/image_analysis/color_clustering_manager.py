"""
Color-aware clustering manager that orchestrates the complete clustering pipeline with color information.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .color_cluster import ColorBoundingBoxClusterer
from .box_merger import BoxMerger
from .serializer import BoxSerializer


class ColorClusteringManager:
    """
    Manager class that orchestrates the complete color-aware clustering pipeline:
    1. Clustering bounding boxes with color information
    2. Merging clusters into new boxes
    3. Serializing results to JSON
    """
    
    def __init__(self, 
                 boxes: Optional[List[Tuple[float, float, float, float]]] = None,
                 colors: Optional[List[Tuple[int, int, int]]] = None,
                 image_path: Optional[str] = None):
        """
        Initialize the color-aware clustering manager.
        
        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
            colors: List of dominant colors as (R, G, B) tuples
            image_path: Path to the source image for color extraction
        """
        self.clusterer = ColorBoundingBoxClusterer(boxes, colors, image_path)
        self.original_boxes = boxes if boxes else []
        self.original_colors = colors if colors else []
        self.image_path = image_path
    
    def add_boxes(self, 
                  boxes: List[Tuple[float, float, float, float]], 
                  colors: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """
        Add bounding boxes and their colors to the manager.
        
        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
            colors: List of dominant colors as (R, G, B) tuples
        """
        self.clusterer.add_boxes(boxes, colors)
        self.original_boxes.extend(boxes)
        if colors:
            self.original_colors.extend(colors)
    
    def run_hierarchical_clustering(
        self,
        method: str = "ward",
        distance_type: str = "enhanced_with_color",
        weights: Optional[Dict[str, float]] = None,
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        auto_threshold: bool = True,
        output_dir: Optional[Path] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run hierarchical clustering with color information, merging and serialization.
        
        Args:
            method: Linkage method for hierarchical clustering
            distance_type: Distance metric type
            weights: Weights for enhanced distance metric
            distance_threshold: Distance threshold for clustering
            n_clusters: Number of clusters to form
            auto_threshold: Automatically determine threshold
            output_dir: Directory to save results
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with clustering results and statistics
        """
        # Perform clustering
        linkage_matrix, cluster_labels = self.clusterer.cluster(
            method=method,
            distance_type=distance_type,
            weights=weights,
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            auto_threshold=auto_threshold
        )
        
        # Get clustered boxes and colors
        clustered_boxes = self.clusterer.get_cluster_boxes(cluster_labels)
        clustered_colors = self.clusterer.get_cluster_colors(cluster_labels)
        
        # Merge clusters into new boxes
        merged_boxes = BoxMerger.merge_clusters(clustered_boxes)
        
        # Calculate statistics
        quality_metrics = self.clusterer.analyze_clustering_quality(cluster_labels)
        merge_stats = BoxMerger.calculate_merge_statistics(
            self.original_boxes, clustered_boxes, merged_boxes
        )
        
        # Prepare results
        results = {
            "method": f"color_hierarchical_{method}_{distance_type}",
            "original_boxes": self.original_boxes,
            "original_colors": self.original_colors,
            "clustered_boxes": clustered_boxes,
            "clustered_colors": clustered_colors,
            "merged_boxes": merged_boxes,
            "cluster_labels": cluster_labels,
            "linkage_matrix": linkage_matrix,
            "quality_metrics": quality_metrics,
            "merge_statistics": merge_stats,
            "clustering_params": {
                "method": method,
                "distance_type": distance_type,
                "weights": weights,
                "distance_threshold": distance_threshold,
                "n_clusters": n_clusters,
                "auto_threshold": auto_threshold
            }
        }
        
        # Save results if requested
        if save_results and output_dir:
            saved_files = self._save_color_clustering_results(
                self.original_boxes,
                self.original_colors,
                clustered_boxes,
                clustered_colors,
                merged_boxes,
                output_dir,
                results["method"],
                cluster_labels,
                quality_metrics
            )
            results["saved_files"] = saved_files
        
        return results
    
    def run_dbscan_clustering(
        self,
        distance_type: str = "enhanced_with_color",
        weights: Optional[Dict[str, float]] = None,
        eps: float = 0.3,
        min_samples: int = 2,
        output_dir: Optional[Path] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run DBSCAN clustering with color information, merging and serialization.
        
        Args:
            distance_type: Distance metric type
            weights: Weights for enhanced distance metric
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            output_dir: Directory to save results
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with clustering results and statistics
        """
        # Perform clustering
        cluster_labels = self.clusterer.cluster_dbscan(
            distance_type=distance_type,
            weights=weights,
            eps=eps,
            min_samples=min_samples
        )
        
        # Get clustered boxes and colors
        clustered_boxes = self.clusterer.get_cluster_boxes(cluster_labels)
        clustered_colors = self.clusterer.get_cluster_colors(cluster_labels)
        
        # Merge clusters into new boxes
        merged_boxes = BoxMerger.merge_clusters(clustered_boxes)
        
        # Calculate statistics
        quality_metrics = self.clusterer.analyze_clustering_quality(cluster_labels)
        merge_stats = BoxMerger.calculate_merge_statistics(
            self.original_boxes, clustered_boxes, merged_boxes
        )
        
        # Prepare results
        results = {
            "method": f"color_dbscan_{distance_type}",
            "original_boxes": self.original_boxes,
            "original_colors": self.original_colors,
            "clustered_boxes": clustered_boxes,
            "clustered_colors": clustered_colors,
            "merged_boxes": merged_boxes,
            "cluster_labels": cluster_labels,
            "quality_metrics": quality_metrics,
            "merge_statistics": merge_stats,
            "clustering_params": {
                "distance_type": distance_type,
                "weights": weights,
                "eps": eps,
                "min_samples": min_samples
            }
        }
        
        # Save results if requested
        if save_results and output_dir:
            saved_files = self._save_color_clustering_results(
                self.original_boxes,
                self.original_colors,
                clustered_boxes,
                clustered_colors,
                merged_boxes,
                output_dir,
                results["method"],
                cluster_labels,
                quality_metrics
            )
            results["saved_files"] = saved_files
        
        return results
    
    def run_comprehensive_testing(
        self,
        output_dir: Optional[Path] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive testing of all color-aware clustering methods.
        
        Args:
            output_dir: Directory to save results
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with all clustering results
        """
        all_results = {}
        
        # Test hierarchical clustering methods
        methods = ['ward', 'complete', 'average']
        distance_types = ['enhanced_with_color', 'color', 'enhanced']
        
        for method in methods:
            for distance_type in distance_types:
                try:
                    result = self.run_hierarchical_clustering(
                        method=method,
                        distance_type=distance_type,
                        output_dir=output_dir,
                        save_results=save_results
                    )
                    all_results[f"color_hierarchical_{method}_{distance_type}"] = result
                except Exception as e:
                    print(f"Error with color_hierarchical_{method}_{distance_type}: {e}")
        
        # Test DBSCAN
        try:
            dbscan_result = self.run_dbscan_clustering(
                output_dir=output_dir,
                save_results=save_results
            )
            all_results["color_dbscan_enhanced_with_color"] = dbscan_result
        except Exception as e:
            print(f"Error with color DBSCAN: {e}")
        
        # Find best result
        best_result = self._find_best_result(all_results)
        all_results["best_result"] = best_result
        
        return all_results
    
    def _find_best_result(self, all_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find the best clustering result based on quality metrics.
        
        Args:
            all_results: Dictionary of all clustering results
            
        Returns:
            Best clustering result or None
        """
        best_result = None
        best_score = 0
        
        for method_name, result in all_results.items():
            if method_name == "best_result":
                continue
                
            quality = result.get("quality_metrics", {})
            merge_stats = result.get("merge_statistics", {})
            
            # Calculate a composite score
            n_clusters = quality.get("n_clusters", 0)
            avg_intra_distance = quality.get("avg_intra_distance", float('inf'))
            reduction_ratio = merge_stats.get("reduction_ratio", 0)
            
            # Prefer methods that create multiple clusters with low intra-distance
            if n_clusters > 1 and avg_intra_distance < float('inf'):
                score = n_clusters * reduction_ratio / (avg_intra_distance + 1e-6)
                if score > best_score:
                    best_score = score
                    best_result = result
        
        return best_result
    
    def _save_color_clustering_results(
        self,
        original_boxes: List[Tuple[float, float, float, float]],
        original_colors: List[Tuple[int, int, int]],
        clustered_boxes: List[List[Tuple[float, float, float, float]]],
        clustered_colors: List[List[Tuple[int, int, int]]],
        merged_boxes: List[Tuple[float, float, float, float]],
        output_dir: Path,
        method_name: str,
        cluster_labels: Any = None,
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Save comprehensive color clustering results to multiple JSON files.
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
        
        # Save clustering details with color information
        clustering_details = {
            "method": method_name,
            "original_count": len(original_boxes),
            "cluster_count": len(clustered_boxes),
            "merged_count": len(merged_boxes),
            "clusters": []
        }
        
        for i, (cluster, colors) in enumerate(zip(clustered_boxes, clustered_colors)):
            cluster_data = {
                "cluster_id": i,
                "size": len(cluster),
                "boxes": [list(box) for box in cluster],
                "colors": [list(color) for color in colors]
            }
            clustering_details["clusters"].append(cluster_data)
        
        if cluster_labels is not None:
            clustering_details["cluster_labels"] = cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else cluster_labels
        
        if quality_metrics is not None:
            clustering_details["quality_metrics"] = quality_metrics
        
        details_file = output_dir / f"{method_name}_clustering_details.json"
        import json
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(clustering_details, f, indent=2)
        saved_files["clustering_details"] = details_file
        
        return saved_files
    
    def plot_best_dendrogram(self, results: Dict[str, Any]) -> None:
        """
        Plot dendrogram for the best hierarchical clustering result.
        
        Args:
            results: Clustering results from run_comprehensive_testing
        """
        best_result = results.get("best_result")
        if best_result and "linkage_matrix" in best_result:
            self.clusterer.plot_dendrogram(
                best_result["linkage_matrix"],
                title=f"Best Color Result: {best_result['method']}"
            ) 