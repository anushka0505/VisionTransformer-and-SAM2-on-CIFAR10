# VisionTransformer-and-SAM2-on-CIFAR10
# 🎯 Vision Transformer on CIFAR-10
Assignment: Q1 — Vision Transformer on CIFAR-10 (PyTorch)
Implementation of a Vision Transformer (ViT) for CIFAR-10 image classification, based on the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., ICLR 2021).
```python
Overview
This project implements a compact Vision Transformer trained on CIFAR-10 with various regularization techniques and training tricks to maximize test accuracy.

Architecture Components
Patch Embedding: Convolutional layer to split images into patches and project to embedding dimension
CLS Token: Learnable classification token prepended to sequence
Positional Embeddings: Learnable position encodings for spatial information
Transformer Encoder: Stack of Multi-Head Self-Attention (MHSA) + MLP blocks with residual connections and layer normalization
Classification Head: Linear layer that classifies from the CLS token representation

How to Run in Google Colab
Step 1: Setup Environment
python   # No additional installations needed - uses standard PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

Step 2: Upload and Run Code
Open Google Colab
Upload the vit_cifar10.py file or copy-paste the code into a cell
Run the cell (the dataset will auto-download on first run)

Step 3: Monitor Training
The script will:
Download CIFAR-10 dataset automatically
Train the model with progress output each epoch
Apply early stopping when validation accuracy plateaus
Display a training/test accuracy curve at the end
Expected Runtime: ~15-25 minutes on Colab GPU (T4)

Best Model Configuration
HyperparameterValueDescriptionPatch Size4×4Divides 32×32 images into 8×8 = 64 patchesEmbedding Dim128Feature dimension for tokensDepth6Number of Transformer encoder layersAttention Heads4Multi-head attention headsMLP Ratio2.0Hidden dimension multiplier for MLP blocksDropout0.1Dropout rateBatch Size128Training batch sizeOptimizerAdamWWith weight decay 0.05Learning Rate3e-4Base learning rateLR ScheduleWarmup (5 ep) + CosineWarmup then cosine annealingEpochs120Maximum epochs (with early stopping)Label Smoothing0.1Smoothing for cross-entropy lossEMA0.999Exponential Moving Average of weightsMixupα=0.2Data augmentation mixing parameter

Data Augmentation
Random resized crop (scale 0.8-1.0)
Random horizontal flip
Random erasing (p=0.25)
Mixup (α=0.2)

Results
MetricValueTest Accuracy~82-85%Training Time~15-25 min (Colab T4 GPU)Parameters~0.5MEarly StoppingPatience = 5 epochs
Performance Notes

EMA model is used for evaluation (provides more stable predictions)
Early stopping prevents overfitting
Best accuracy typically achieved around epochs 60-100
Results may vary slightly due to random initialization and augmentation

Key Implementation Details

1. Vision Transformer Architecture
pythonInput Image (3×32×32)
    ↓ Conv2d patch embedding (4×4 patches)
Patch Tokens (64×128)
    ↓ Prepend CLS token + Add positional embeddings
Token Sequence (65×128)
    ↓ 6× Transformer Encoder Layers
        - Multi-Head Self-Attention (4 heads)
        - MLP (128→256→128)
        - Layer Norm (pre-norm)
        - Residual connections
Encoded Sequence (65×128)
    ↓ Extract CLS token
CLS Representation (128)
    ↓ Linear classification head
Output Logits (10 classes)

2. Training Techniques
Mixup: Mixes pairs of images and labels for better generalization
EMA: Maintains exponential moving average of model weights
Label Smoothing: Reduces overconfidence in predictions
Warmup + Cosine LR: Gradual warmup followed by cosine decay
Early Stopping: Stops when validation accuracy doesn't improve

3. Regularization
Weight decay (0.05)
Dropout (0.1)
Random erasing augmentation
Label smoothing (0.1)

Code Structure
├── Dataset Loading (CIFAR-10 with augmentation)
├── ViT Model Definition
│   ├── Patch Embedding (Conv2d)
│   ├── CLS Token & Positional Embedding
│   ├── Transformer Encoder Stack
│   └── Classification Head
├── Training Utilities
│   ├── Mixup augmentation
│   ├── EMA weight update
│   └── Warmup-Cosine LR scheduler
├── Training Loop
│   ├── Train one epoch
│   ├── Evaluate on test set
│   └── Early stopping logic
└── Visualization (Accuracy curves)

Dependencies
All dependencies are pre-installed in Google Colab:
PyTorch >= 1.10
torchvision
numpy
matplotlib
tqdm

References
Paper: Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
Dataset: CIFAR-10 (Krizhevsky, 2009)

🔮 Tips for Further Improvement
Increase model capacity: Try larger embedding dimensions (192, 256) or more layers
Data augmentation: Add AutoAugment, CutMix, or RandAugment
Training longer: Increase epochs to 200-300 with adjusted patience
Stochastic depth: Add drop path regularization
Knowledge distillation: Use a teacher model for soft labels
Test-time augmentation: Average predictions over multiple augmented views


Note: This is a lightweight ViT designed for CIFAR-10's small 32×32 images. For larger datasets like ImageNet, you would typically use larger patch sizes (16×16) and deeper models.
```

# 🎯 Text-Driven Image Segmentation with SAM 2
Assignment: Q2 For a single image, perform text-prompted segmentation of a chosen object using SAM 2.

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

🎮 Usage (Colab)
Open notebook using the badge above
Enable GPU: Runtime → Change runtime type → GPU
Run installation cells
Upload your image
Enter text prompt ("dog", "person", "bottle")
View results – mask & overlay displayed automatically
Download output PNGs
Example Text Prompts:
"person"   # Segments people
"cat"      # Segments cats
"car"      # Segments vehicles
"bottle"   # Segments bottles
"laptop"   # Segments electronics

⚠️ Limitations
Simplified Prompt Generation – Uses geometric heuristics, not semantic understanding
No Grounding Model – Text doesn’t guide point placement semantically
Object Localization – May miss off-center or small objects
Performance Constraints – GPU required for reasonable speed (~2-5 sec/image)
No Video Support – Only static image segmentation

📁 Output Files

mask_[your_prompt].png – Binary segmentation mask
overlay_[your_prompt].png – Original image with mask overlay

🔮 Future Improvements
Integrate GroundingDINO for semantic text-to-box conversion
Add CLIPSeg for text-guided point selection
Multi-object detection & segmentation
Video segmentation with temporal propagation
Support complex queries with spatial relationships
