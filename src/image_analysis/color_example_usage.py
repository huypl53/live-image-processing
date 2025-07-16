"""
Example usage of the color-aware clustering system.
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

from image_analysis.color_clustering_manager import ColorClusteringManager


def load_boxes_from_json(file_path: Path) -> List[Tuple[float, float, float, float]]:
    """
    Load bounding boxes from JSON file.
    
    Args:
        file_path: Path to JSON file with bounding boxes
        
    Returns:
        List of bounding boxes as (x, y, width, height) tuples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract bounding boxes from the JSON structure
    boxes = []
    for component in data.get("components", []):
        bbox = component.get("bbox", {})
        if bbox:
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)
            boxes.append((x, y, width, height))
    
    return boxes


def main():
    """
    Main example demonstrating the color-aware clustering pipeline.
    """
    # Load bounding boxes from JSON file
    boxes_file = Path(__file__).parent.parent.parent / "examples" / "bboxes.json"
    
    if not boxes_file.exists():
        print(f"Error: Boxes file not found at {boxes_file}")
        print("Please ensure the bboxes.json file exists in the examples directory.")
        return
    
    boxes = load_boxes_from_json(boxes_file)
    print(f"Loaded {len(boxes)} bounding boxes from {boxes_file}")
    
    # Path to the source image for color extraction
    image_path = "samples/03.png"
    
    # Create color-aware clustering manager
    manager = ColorClusteringManager(boxes, image_path=image_path)
    
    # Set output directory for results
    output_dir = Path(__file__).parent.parent.parent / "color_clustering_results"
    
    print("\n=== Running Color-Aware Clustering Analysis ===")
    
    # Method 1: Run color-aware hierarchical clustering
    print("\n1. Testing color-aware hierarchical clustering...")
    result1 = manager.run_hierarchical_clustering(
        method="ward",
        distance_type="enhanced_with_color",
        weights={'iou': 0.2, 'center': 0.3, 'size': 0.2, 'color': 0.3},
        output_dir=output_dir,
        save_results=True
    )
    
    print(f"   Result: {result1['method']}")
    print(f"   Clusters: {result1['quality_metrics']['n_clusters']}")
    print(f"   Merged boxes: {len(result1['merged_boxes'])}")
    
    # Method 2: Run color-aware DBSCAN clustering
    print("\n2. Testing color-aware DBSCAN clustering...")
    result2 = manager.run_dbscan_clustering(
        eps=0.3,
        min_samples=2,
        output_dir=output_dir,
        save_results=True
    )
    
    print(f"   Result: {result2['method']}")
    print(f"   Clusters: {result2['quality_metrics']['n_clusters']}")
    print(f"   Merged boxes: {len(result2['merged_boxes'])}")
    
    # Method 3: Run comprehensive testing
    print("\n3. Running comprehensive testing of all color-aware methods...")
    all_results = manager.run_comprehensive_testing(
        output_dir=output_dir,
        save_results=True
    )
    
    # Show best result
    best_result = all_results.get("best_result")
    if best_result:
        print(f"\n=== Best Color-Aware Result: {best_result['method']} ===")
        
        quality = best_result.get("quality_metrics", {})
        merge_stats = best_result.get("merge_statistics", {})
        
        print(f"Clusters: {quality.get('n_clusters', 0)}")
        print(f"Reduction ratio: {merge_stats.get('reduction_ratio', 0):.2f}")
        print(f"Avg intra-distance: {quality.get('avg_intra_distance', 0):.3f}")
        
        # Show merged boxes in the requested format
        merged_boxes = best_result.get("merged_boxes", [])
        print(f"\nMerged boxes in [x, y, width, height, 'box'] format:")
        for i, box in enumerate(merged_boxes):
            x, y, width, height = box
            print(f"  [{x}, {y}, {width}, {height}, 'box']")
        
        # Show color information for clusters
        clustered_colors = best_result.get("clustered_colors", [])
        print(f"\nColor information for clusters:")
        for i, cluster_colors in enumerate(clustered_colors):
            print(f"  Cluster {i}: {len(cluster_colors)} boxes")
            if cluster_colors:
                # Show average color for the cluster
                avg_color = tuple(np.mean(cluster_colors, axis=0).astype(int))
                print(f"    Average color: RGB{avg_color}")
        
        # Save the best merged boxes to a separate file
        best_merged_file = output_dir / "best_color_merged_boxes.json"
        from image_analysis.serializer import BoxSerializer
        BoxSerializer.save_boxes_to_json(merged_boxes, best_merged_file)
        print(f"\nBest color-aware merged boxes saved to: {best_merged_file}")
        
        # Plot dendrogram if it's a hierarchical method
        if "linkage_matrix" in best_result:
            print("\nGenerating color-aware dendrogram plot...")
            manager.plot_best_dendrogram(all_results)
    
    print(f"\nAll color-aware results saved to: {output_dir}")
    print("Check the output directory for detailed JSON files with clustering results.")


def compare_color_vs_no_color():
    """
    Compare clustering results with and without color information.
    """
    from image_analysis.clustering_manager import ClusteringManager
    
    # Load boxes
    boxes_file = Path(__file__).parent.parent.parent / "examples" / "bboxes.json"
    boxes = load_boxes_from_json(boxes_file)
    
    # Create both managers
    color_manager = ColorClusteringManager(boxes, image_path="samples/03.png")
    regular_manager = ClusteringManager(boxes)
    
    output_dir = Path(__file__).parent.parent.parent / "comparison_results"
    
    print("=== Comparing Color vs No-Color Clustering ===")
    
    # Run same method on both
    color_result = color_manager.run_hierarchical_clustering(
        method="ward",
        distance_type="enhanced_with_color",
        output_dir=output_dir / "color",
        save_results=True
    )
    
    regular_result = regular_manager.run_hierarchical_clustering(
        method="ward",
        distance_type="enhanced",
        output_dir=output_dir / "no_color",
        save_results=True
    )
    
    print(f"\nColor-aware clustering:")
    print(f"  Clusters: {color_result['quality_metrics']['n_clusters']}")
    print(f"  Avg intra-distance: {color_result['quality_metrics']['avg_intra_distance']:.3f}")
    
    print(f"\nRegular clustering:")
    print(f"  Clusters: {regular_result['quality_metrics']['n_clusters']}")
    print(f"  Avg intra-distance: {regular_result['quality_metrics']['avg_intra_distance']:.3f}")
    
    # Determine which is better
    color_score = color_result['quality_metrics']['n_clusters'] / (color_result['quality_metrics']['avg_intra_distance'] + 1e-6)
    regular_score = regular_result['quality_metrics']['n_clusters'] / (regular_result['quality_metrics']['avg_intra_distance'] + 1e-6)
    
    print(f"\nComparison:")
    print(f"  Color-aware score: {color_score:.3f}")
    print(f"  Regular score: {regular_score:.3f}")
    
    if color_score > regular_score:
        print("  Color-aware clustering performed better!")
    else:
        print("  Regular clustering performed better!")


if __name__ == "__main__":
    main()
    
    # Uncomment to run comparison
    # compare_color_vs_no_color() 