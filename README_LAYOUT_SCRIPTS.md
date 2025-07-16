# Layout Image Generation Scripts

This directory contains bash scripts to automatically generate layout images from clustering results.

## Scripts Overview

### 1. `generate_layout_images.sh` (Comprehensive)
- **Features**: Full-featured script with error handling, colored output, and statistics
- **Usage**: `./generate_layout_images.sh`
- **Best for**: Production use, detailed logging

### 2. `run_layout_commands.sh` (Simple)
- **Features**: Simple, customizable script
- **Usage**: `./run_layout_commands.sh [sample_image] [output_dir]`
- **Best for**: Quick testing, customization

## Prerequisites

1. **uv**: Make sure `uv` is installed and available in your PATH
2. **Sample Image**: Ensure `samples/03.png` exists (or specify a different image)
3. **Clustering Results**: Run the clustering analysis first to generate JSON files

## Usage Examples

### Basic Usage (Simple Script)
```bash
# Use default settings
./run_layout_commands.sh

# Specify custom sample image
./run_layout_commands.sh samples/other_image.png

# Specify custom sample image and output directory
./run_layout_commands.sh samples/other_image.png custom_output/
```

### Advanced Usage (Comprehensive Script)
```bash
# Run with full logging and error handling
./generate_layout_images.sh
```

## What the Scripts Do

1. **Find JSON Files**: Automatically discovers all `*_merged_boxes.json` files in the clustering results directory
2. **Generate PNG Files**: For each JSON file, creates a corresponding PNG file with the same base name
3. **Layout Command**: Runs `uv run layout bbox` with the following parameters:
   - `-i`: Input sample image
   - `-f xywh`: Format specification
   - `-b`: Bounding boxes JSON file
   - `-o`: Output PNG file

## File Naming Convention

The scripts expect JSON files with the naming pattern:
```
{method}_{distance_type}_merged_boxes.json
```

Examples:
- `hierarchical_ward_enhanced_merged_boxes.json`
- `hierarchical_average_center_merged_boxes.json`
- `dbscan_enhanced_merged_boxes.json`

Output PNG files will have the same base name:
- `hierarchical_ward_enhanced_merged_boxes.png`
- `hierarchical_average_center_merged_boxes.png`
- `dbscan_enhanced_merged_boxes.png`

## Directory Structure

```
package/image_analysis/
├── clustering_results/
│   ├── hierarchical_ward_enhanced_merged_boxes.json
│   ├── hierarchical_ward_enhanced_merged_boxes.png
│   ├── hierarchical_average_center_merged_boxes.json
│   ├── hierarchical_average_center_merged_boxes.png
│   └── ...
├── samples/
│   └── 03.png
├── generate_layout_images.sh
├── run_layout_commands.sh
└── README_LAYOUT_SCRIPTS.md
```

## Troubleshooting

### Common Issues

1. **"Sample image not found"**
   - Ensure the sample image exists at the specified path
   - Use the `-i` parameter to specify a different image

2. **"Clustering results directory not found"**
   - Run the clustering analysis first to generate JSON files
   - Check that the clustering results are in the expected directory

3. **"uv command not found"**
   - Install uv: `pip install uv` or follow the official installation guide
   - Ensure uv is in your PATH

4. **"No merged_boxes.json files found"**
   - Run the clustering analysis to generate the JSON files
   - Check that the clustering completed successfully

### Debug Mode

To see more detailed output, you can modify the scripts to add `set -x` at the beginning:

```bash
#!/bin/bash
set -x  # Add this line for debug output
# ... rest of script
```

## Integration with Clustering Pipeline

These scripts are designed to work seamlessly with the clustering system:

1. **Run Clustering**: Use `ClusteringManager` to generate JSON files
2. **Generate Layout Images**: Use these scripts to create visual representations
3. **Review Results**: Compare different clustering methods visually

## Customization

### Modifying the Layout Command

To change the layout command parameters, edit the script and modify this line:

```bash
uv run layout bbox -i "$SAMPLE_IMAGE" -f xywh -b "$json_file" -o "$output_png"
```

### Adding New Parameters

You can add additional parameters to the layout command:

```bash
uv run layout bbox -i "$SAMPLE_IMAGE" -f xywh -b "$json_file" -o "$output_png" --additional-param value
```

### Batch Processing Different Images

To process multiple sample images, you can create a wrapper script:

```bash
#!/bin/bash
for image in samples/*.png; do
    echo "Processing $image..."
    ./run_layout_commands.sh "$image" "output_$(basename "$image" .png)/"
done
``` 