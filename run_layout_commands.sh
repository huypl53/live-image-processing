#!/bin/bash

# Simple script to run layout commands on all merged_boxes.json files
# Usage: ./run_layout_commands.sh [sample_image] [output_dir]

# Default values
SAMPLE_IMAGE="${1:-samples/03.png}"
OUTPUT_DIR="${2:-color_clustering_results}"
CLUSTERING_RESULTS_DIR="color_clustering_results"

echo "=== Layout Image Generator ==="
echo "Sample image: $SAMPLE_IMAGE"
echo "Output directory: $OUTPUT_DIR"
echo "Clustering results: $CLUSTERING_RESULTS_DIR"
echo ""

# Check if sample image exists
if [[ ! -f "$SAMPLE_IMAGE" ]]; then
    echo "Error: Sample image not found: $SAMPLE_IMAGE"
    exit 1
fi

# Check if clustering results directory exists
if [[ ! -d "$CLUSTERING_RESULTS_DIR" ]]; then
    echo "Error: Clustering results directory not found: $CLUSTERING_RESULTS_DIR"
    echo "Please run the clustering analysis first."
    exit 1
fi

# Find all merged_boxes.json files
json_files=($(find "$CLUSTERING_RESULTS_DIR" -name "*_merged_boxes.json" -type f))

if [[ ${#json_files[@]} -eq 0 ]]; then
    echo "No merged_boxes.json files found in $CLUSTERING_RESULTS_DIR"
    exit 0
fi

echo "Found ${#json_files[@]} merged_boxes.json files to process"
echo ""

# Process each file
for json_file in "${json_files[@]}"; do
    # Get base name without extension
    base_name=$(basename "$json_file" .json)
    output_png="$OUTPUT_DIR/${base_name}.png"
    
    echo "Processing: $base_name"
    
    # Skip if JSON file is empty
    if [[ ! -s "$json_file" ]]; then
        echo "  Skipping empty file"
        continue
    fi
    
    # Run the layout command
    echo "  Running: uv run layout bbox -i $SAMPLE_IMAGE -f xywh -b $json_file -o $output_png"
    
    if uv run layout bbox -i "$SAMPLE_IMAGE" -f xywh -b "$json_file" -o "$output_png"; then
        echo "  ✓ Generated: $output_png"
    else
        echo "  ✗ Failed to generate: $output_png"
    fi
    
    echo ""
done

echo "Layout image generation completed!" 