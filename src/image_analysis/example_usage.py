"""
Simple example usage of the modular clustering system.
"""

import json
from pathlib import Path
from typing import List, Tuple

from .clustering_manager import ClusteringManager


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
    Main example demonstrating the complete clustering pipeline.
    """
    # Load bounding boxes from JSON file
    boxes_file = Path(__file__).parent.parent.parent / "examples" / "bboxes.json"
    
    if not boxes_file.exists():
        print(f"Error: Boxes file not found at {boxes_file}")
        print("Please ensure the bboxes.json file exists in the examples directory.")
        return
    
    boxes = load_boxes_from_json(boxes_file)
    print(f"Loaded {len(boxes)} bounding boxes from {boxes_file}")
    
    # Create clustering manager
    manager = ClusteringManager(boxes)
    
    # Set output directory for results
    output_dir = Path(__file__).parent.parent.parent / "clustering_results"
    
    print("\n=== Running Clustering Analysis ===")
    
    # Method 1: Run a specific hierarchical clustering method
    print("\n1. Testing hierarchical clustering with ward method...")
    result1 = manager.run_hierarchical_clustering(
        method="ward",
        distance_type="enhanced",
        output_dir=output_dir,
        save_results=True
    )
    
    print(f"   Result: {result1['method']}")
    print(f"   Clusters: {result1['quality_metrics']['n_clusters']}")
    print(f"   Merged boxes: {len(result1['merged_boxes'])}")
    
    # Method 2: Run DBSCAN clustering
    print("\n2. Testing DBSCAN clustering...")
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
    print("\n3. Running comprehensive testing of all methods...")
    all_results = manager.run_comprehensive_testing(
        output_dir=output_dir,
        save_results=True
    )
    
    # Show best result
    best_result = all_results.get("best_result")
    if best_result:
        print(f"\n=== Best Result: {best_result['method']} ===")
        
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
        
        # Save the best merged boxes to a separate file
        best_merged_file = output_dir / "best_merged_boxes.json"
        from .serializer import BoxSerializer
        BoxSerializer.save_boxes_to_json(merged_boxes, best_merged_file)
        print(f"\nBest merged boxes saved to: {best_merged_file}")
        
        # Plot dendrogram if it's a hierarchical method
        if "linkage_matrix" in best_result:
            print("\nGenerating dendrogram plot...")
            manager.plot_best_dendrogram(all_results)
    
    print(f"\nAll results saved to: {output_dir}")
    print("Check the output directory for detailed JSON files with clustering results.")


if __name__ == "__main__":
    main() 