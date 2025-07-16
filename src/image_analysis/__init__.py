"""
Image Analysis Package

This package provides tools for analyzing and clustering bounding boxes in images.
"""

from .cluster import BoundingBoxClusterer
from .box_merger import BoxMerger
from .serializer import BoxSerializer
from .clustering_manager import ClusteringManager
from .color_cluster import ColorBoundingBoxClusterer
from .color_clustering_manager import ColorClusteringManager

__all__ = [
    "BoundingBoxClusterer",
    "BoxMerger", 
    "BoxSerializer",
    "ClusteringManager",
    "ColorBoundingBoxClusterer",
    "ColorClusteringManager"
]
