# VisionTransformer-and-SAM2-on-CIFAR10
# 🎯 Text-Driven Image Segmentation with SAM 
Automatic object segmentation powered by **Meta’s Segment Anything Model 2 (SAM 2)**. Segment objects in images using **plain text prompts**.

---

## 🚀 Quick Start

```python
# 1. Run installation cells
# 2. Upload your image
# 3. Enter text prompt: "cat", "person", "car", etc.
# 4. Get segmentation results!
📋 Overview

Upload an image and enter a text prompt. The system automatically:

🔍 Generates intelligent prompt points/boxes

🎨 Segments the object using SAM 2

📊 Displays mask and overlay

💾 Saves results as PNG files

Example:

Input: Image of a cat + Text: "cat"

Output: Segmentation mask highlighting the cat

🛠️ How It Works

Pipeline:

Text Prompt → Geometric Prompts → SAM 2 Model → Segmentation Mask

Three-Stage Process:

Stage 1: Text-to-Prompt Conversion

Generates multiple prompt strategies from your text

Creates center points, grid points, and bounding boxes

Uses heuristics for object localization

Stage 2: SAM 2 Segmentation

Tests all prompt strategies automatically

Runs SAM 2 (Hiera-Large, 879MB)

Selects best result based on confidence scores

Stage 3: Visualization & Export

Displays original image, mask, and colored overlay

Shows segmentation statistics (coverage %, quality score)

Saves files: mask_[prompt].png & overlay_[prompt].png

💡 Implementation Details

Heuristic-based prompt generation (no GroundingDINO/GLIP)

Prompt Strategies:

Center Point: Single center point (good for centered objects)

Grid Points: 5-point grid for distributed objects

Center Box: 60% coverage box (for larger objects)

System automatically selects strategy with highest confidence

✨ Features

✅ Fully Automatic – No manual annotation needed
✅ Multi-Strategy – Tests multiple approaches automatically
✅ Quality Scoring – Picks best segmentation result
✅ Detailed Stats – Shows coverage %, confidence scores
✅ Export Ready – Saves masks and overlays as PNG
✅ GPU Accelerated – Fast inference with CUDA support

📊 Sample Output
==========================================================
Segmentation Statistics:
Text Prompt: 'cat'
Method Used: Center Point
Quality Score: 0.987
Mask Coverage: 23.45%
Image Size: 1024x768
==========================================================


