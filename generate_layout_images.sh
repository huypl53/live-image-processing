#!/bin/bash

# Script to generate layout images from clustering results
# This script processes all merged_boxes.json files and generates corresponding PNG images

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTERING_RESULTS_DIR="$SCRIPT_DIR/clustering_results"
SAMPLE_IMAGE="samples/03.png"
OUTPUT_DIR="$CLUSTERING_RESULTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [[ ! -d "$CLUSTERING_RESULTS_DIR" ]]; then
    print_error "Clustering results directory not found: $CLUSTERING_RESULTS_DIR"
    print_error "Please run the clustering analysis first."
    exit 1
fi

# Check if sample image exists
if [[ ! -f "$SAMPLE_IMAGE" ]]; then
    print_error "Sample image not found: $SAMPLE_IMAGE"
    print_error "Please ensure the sample image exists."
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error "uv command not found. Please install uv first."
    exit 1
fi

print_status "Starting layout image generation..."
print_status "Clustering results directory: $CLUSTERING_RESULTS_DIR"
print_status "Sample image: $SAMPLE_IMAGE"
print_status "Output directory: $OUTPUT_DIR"

# Counter for statistics
total_files=0
processed_files=0
skipped_files=0
error_files=0

# Find all merged_boxes.json files
merged_boxes_files=($(find "$CLUSTERING_RESULTS_DIR" -name "*_merged_boxes.json" -type f))

if [[ ${#merged_boxes_files[@]} -eq 0 ]]; then
    print_warning "No merged_boxes.json files found in $CLUSTERING_RESULTS_DIR"
    print_warning "Please run the clustering analysis first to generate JSON files."
    exit 0
fi

print_status "Found ${#merged_boxes_files[@]} merged_boxes.json files to process"

# Process each merged_boxes.json file
for json_file in "${merged_boxes_files[@]}"; do
    total_files=$((total_files + 1))
    
    # Get the base name without extension
    base_name=$(basename "$json_file" .json)
    
    # Generate output PNG filename
    output_png="$OUTPUT_DIR/${base_name}.png"
    
    print_status "Processing: $base_name"
    
    # Check if JSON file is not empty
    if [[ ! -s "$json_file" ]]; then
        print_warning "Skipping empty file: $json_file"
        skipped_files=$((skipped_files + 1))
        continue
    fi
    
    # Check if output PNG already exists
    if [[ -f "$output_png" ]]; then
        print_warning "Output file already exists: $output_png"
        print_warning "Skipping to avoid overwriting..."
        skipped_files=$((skipped_files + 1))
        continue
    fi
    
    # Run the layout command
    print_status "Running: uv run layout bbox -i $SAMPLE_IMAGE -f xywh -b $json_file -o $output_png"
    
    if uv run layout bbox -i "$SAMPLE_IMAGE" -f xywh -b "$json_file" -o "$output_png"; then
        print_success "Generated: $output_png"
        processed_files=$((processed_files + 1))
    else
        print_error "Failed to generate: $output_png"
        error_files=$((error_files + 1))
    fi
done

# Print summary
echo ""
print_status "=== Processing Summary ==="
print_status "Total files found: $total_files"
print_success "Successfully processed: $processed_files"
print_warning "Skipped (empty/existing): $skipped_files"
if [[ $error_files -gt 0 ]]; then
    print_error "Errors: $error_files"
fi

# List generated PNG files
echo ""
print_status "=== Generated PNG Files ==="
png_files=($(find "$OUTPUT_DIR" -name "*_merged_boxes.png" -type f))
if [[ ${#png_files[@]} -gt 0 ]]; then
    for png_file in "${png_files[@]}"; do
        echo "  $(basename "$png_file")"
    done
else
    print_warning "No PNG files were generated."
fi

echo ""
print_success "Layout image generation completed!" 