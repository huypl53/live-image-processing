"""
Clustering manager that orchestrates the complete clustering pipeline.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .cluster import BoundingBoxClusterer
from .box_merger import BoxMerger
from .serializer import BoxSerializer


class ClusteringManager:
    """
    Manager class that orchestrates the complete clustering pipeline:
    1. Clustering bounding boxes
    2. Merging clusters into new boxes
    3. Serializing results to JSON
    """
    
    def __init__(self, boxes: Optional[List[Tuple[float, float, float, float]]] = None):
        """
        Initialize the clustering manager.
        
        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
        """
        self.clusterer = BoundingBoxClusterer(boxes)
        self.original_boxes = boxes if boxes else []
    
    def add_boxes(self, boxes: List[Tuple[float, float, float, float]]) -> None:
        """
        Add bounding boxes to the manager.
        
        Args:
            boxes: List of bounding boxes as (x, y, width, height) tuples
        """
        self.clusterer.add_boxes(boxes)
        self.original_boxes.extend(boxes)
    
    def run_hierarchical_clustering(
        self,
        method: str = "ward",
        distance_type: str = "enhanced",
        weights: Optional[Dict[str, float]] = None,
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        auto_threshold: bool = True,
        output_dir: Optional[Path] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run hierarchical clustering with merging and serialization.
        
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
        
        # Get clustered boxes
        clustered_boxes = self.clusterer.get_cluster_boxes(cluster_labels)
        
        # Merge clusters into new boxes
        merged_boxes = BoxMerger.merge_clusters(clustered_boxes)
        
        # Calculate statistics
        quality_metrics = self.clusterer.analyze_clustering_quality(cluster_labels)
        merge_stats = BoxMerger.calculate_merge_statistics(
            self.original_boxes, clustered_boxes, merged_boxes
        )
        
        # Prepare results
        results = {
            "method": f"hierarchical_{method}_{distance_type}",
            "original_boxes": self.original_boxes,
            "clustered_boxes": clustered_boxes,
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
            saved_files = BoxSerializer.save_clustering_results(
                self.original_boxes,
                clustered_boxes,
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
        distance_type: str = "enhanced",
        weights: Optional[Dict[str, float]] = None,
        eps: float = 0.3,
        min_samples: int = 2,
        output_dir: Optional[Path] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run DBSCAN clustering with merging and serialization.
        
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
        
        # Get clustered boxes
        clustered_boxes = self.clusterer.get_cluster_boxes(cluster_labels)
        
        # Merge clusters into new boxes
        merged_boxes = BoxMerger.merge_clusters(clustered_boxes)
        
        # Calculate statistics
        quality_metrics = self.clusterer.analyze_clustering_quality(cluster_labels)
        merge_stats = BoxMerger.calculate_merge_statistics(
            self.original_boxes, clustered_boxes, merged_boxes
        )
        
        # Prepare results
        results = {
            "method": f"dbscan_{distance_type}",
            "original_boxes": self.original_boxes,
            "clustered_boxes": clustered_boxes,
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
            saved_files = BoxSerializer.save_clustering_results(
                self.original_boxes,
                clustered_boxes,
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
        Run comprehensive testing of all clustering methods.
        
        Args:
            output_dir: Directory to save results
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with all clustering results
        """
        all_results = {}
        
        # Test hierarchical clustering methods
        methods = ['ward', 'complete', 'average']
        distance_types = ['enhanced', 'center', 'size']
        
        for method in methods:
            for distance_type in distance_types:
                try:
                    result = self.run_hierarchical_clustering(
                        method=method,
                        distance_type=distance_type,
                        output_dir=output_dir,
                        save_results=save_results
                    )
                    all_results[f"hierarchical_{method}_{distance_type}"] = result
                except Exception as e:
                    print(f"Error with hierarchical_{method}_{distance_type}: {e}")
        
        # Test DBSCAN
        try:
            dbscan_result = self.run_dbscan_clustering(
                output_dir=output_dir,
                save_results=save_results
            )
            all_results["dbscan_enhanced"] = dbscan_result
        except Exception as e:
            print(f"Error with DBSCAN: {e}")
        
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
                title=f"Best Result: {best_result['method']}"
            ) 