# Live image processing

## Usage
```bash
uv venv
source .venv/bin/activate
uv sync
uv run unified_patcher.py
```
Absolutely! Here’s a concise explanation for each parameter in your Gradio UI, followed by practical advice on which parameters to focus on for the most impactful segmentation improvements.

---

## 1. **Concise Explanation of Each Parameter**

| Parameter                | Description                                                                                   |
|--------------------------|----------------------------------------------------------------------------------------------|
| **Min Component Area**   | Minimum pixel area for a detected region to be considered a valid component.                 |
| **Max Component Area**   | Maximum pixel area for a detected region (filters out very large, likely background, areas). |
| **Merge Distance Threshold** | How close (in pixels) two components can be before being merged into one.                 |
| **Group X Threshold**    | Horizontal distance (in pixels) for grouping/merging components in the same row.             |
| **Group Y Threshold**    | Vertical distance (in pixels) for grouping/merging components in the same column.            |
| **Table Aspect Ratio**   | Width-to-height ratio above which a component is classified as a table.                      |
| **Gaussian Blur Kernel Size (odd)** | Size of the Gaussian blur kernel (must be odd); helps reduce noise before thresholding. |
| **Adaptive Block Size (odd)** | Size of the local window for adaptive thresholding (must be odd).                        |
| **Adaptive Threshold C** | Constant subtracted from the mean/weighted mean in adaptive thresholding.                    |
| **Adaptive Threshold Method** | Method for adaptive thresholding: 'mean' or 'gaussian'.                                 |
| **Morphological Operation** | Type of morphological operation: dilate, erode, open, or close.                           |
| **Morph Kernel Size (odd)** | Size of the structuring element for morphological operations (must be odd).                |
| **Morph Iterations**     | Number of times the morphological operation is applied.                                      |
| **Min Aspect Ratio**     | Minimum width/height ratio for a component to be considered valid.                           |
| **Max Aspect Ratio**     | Maximum width/height ratio for a component to be considered valid.                           |

---

## 2. **Which Parameters to Focus on for Best Segmentation Enhancement**

### **Most Impactful Parameters (Start Here):**

1. **Min Component Area**
   - **Why:** Filters out noise and tiny artifacts. If you get too many small, irrelevant detections, increase this.
   - **Tip:** Set just below the area of the smallest UI element you care about.

2. **Merge Distance Threshold, Group X Threshold, Group Y Threshold**
   - **Why:** Control how components are merged/grouped. If elements that should be together are split, increase these. If unrelated elements are merged, decrease them.
   - **Tip:** For forms or toolbars, increasing Group X or Y can help merge related fields/buttons.

3. **Adaptive Block Size & Adaptive Threshold C**
   - **Why:** Directly affect how the image is binarized for contour detection. If you miss faint or thin elements, try lowering block size or C. If you get too much noise, increase them.
   - **Tip:** Block size should be odd and roughly match the size of the smallest feature you want to detect.

4. **Morphological Operation, Kernel Size, Iterations**
   - **Why:** Help close gaps (close), remove noise (open), or merge/split regions (dilate/erode). If borders are broken or elements are fragmented, try 'close' with a larger kernel or more iterations.
   - **Tip:** 'Close' is good for joining broken lines; 'open' is good for removing small noise.

### **Secondary Parameters (Fine-tuning):**

- **Max Component Area:** Useful if you want to ignore very large regions (e.g., background panels).
- **Table Aspect Ratio, Min/Max Aspect Ratio:** Useful for classifying or filtering out oddly shaped components.
- **Gaussian Blur Kernel Size:** If you have a lot of speckle noise, a slightly larger blur can help, but too much will blur out small features.

---

### **Quick Recommendations for Common Issues:**

- **Too many small/noisy detections:**  
  Increase **Min Component Area**, try 'open' morph op, increase kernel size/iterations.

- **Missing small/thin elements:**  
  Decrease **Min Component Area**, decrease **Adaptive Block Size**, decrease **Adaptive Threshold C**, use smaller blur.

- **Components are fragmented (split):**  
  Increase **Merge Distance Threshold**, **Group X/Y**, use 'close' morph op, increase kernel size/iterations.

- **Unrelated components are merged:**  
  Decrease **Merge Distance Threshold**, **Group X/Y**, decrease morph kernel/iterations.

- **Background or large panels detected:**  
  Decrease **Max Component Area**.

---

**Summary:**  
Start with Min Component Area, Merge/Group thresholds, and Adaptive Thresholding parameters. These have the biggest impact. Morphological operations are your next best tool for cleaning up the result.

If you describe a specific segmentation problem, I can give you even more targeted advice!