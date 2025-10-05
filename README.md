# 🧠 Vision Transformer (ViT) on CIFAR-10 & Text-Driven Image Segmentation (SAM 2)
### 🎓 Assignment – Google Colab Implementation  
**Files included:**  
- `q1.ipynb` → Vision Transformer (CIFAR-10)  
- `q2.ipynb` → Text-Driven Segmentation with SAM 2  
- `README.md` → Documentation & Results  

---

## 🧩 Q1 – Vision Transformer (ViT) on CIFAR-10

### 📋 Overview
This notebook implements a **Vision Transformer (ViT)** from scratch and trains it on the **CIFAR-10 dataset** (10 image classes).  
No pretrained weights are used — fully custom implementation following:  
📄 *“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” (Dosovitskiy et al., ICLR 2021)*  

The objective is to achieve **maximum test accuracy** under the assignment’s constraints using **Google Colab GPU**.

---

### 🚀 Quick Start
1. Open in **Google Colab**  
2. Run all installation and import cells  
3. Dataset auto-downloads (CIFAR-10)  
4. Execute training cells  
5. View the accuracy curve and test results  

---

### 🧠 Model Architecture
Custom Vision Transformer built **from scratch**:

| Component | Details |
|------------|----------|
| **Patchify** | 4×4 patches (non-overlapping) |
| **Embedding Dim** | 128 |
| **Transformer Depth** | 6 encoder blocks |
| **Attention Heads** | 4 |
| **Positional Embedding** | Learnable |
| **CLS Token** | Prepended for classification |
| **MLP Ratio** | 2.0 |
| **Dropout** | 0.1 |

---

### ⚙️ Training Details
| Setting | Value |
|----------|--------|
| Optimizer | AdamW |
| Base LR | 3e-4 (Warmup + Cosine Decay) |
| Loss | CrossEntropy with Label Smoothing (0.1) |
| Batch Size | 128 |
| Regularization | Mixup (α=0.2), Random Erasing (p=0.25), Weight Decay (0.05) |
| EMA | Exponential Moving Average (decay=0.999) |
| Early Stopping | Patience = 5 |
| Total Epochs | 120 |

---

### 🧩 Techniques Used
- 🔄 **Mixup augmentation**  
- 🧘 **Label smoothing**  
- 💫 **EMA weights** for stable convergence  
- 📉 **Warmup + Cosine LR scheduler**  
- ⏹️ **Early stopping** to prevent overfitting  
- 🔀 **Random Erasing** for data diversity  

---

### 📊 Results
| Metric | Result |
|--------|---------|
| **Best Test Accuracy** | 🏆 **79.66%** |
| **Epoch Reached** | 101 |
| **Training Accuracy** | 75.53% |
| **Early Stopping Triggered** | ✅ Yes (Patience=5) |
| **Device** | Google Colab GPU (T4/A100) |

Epoch 101 | Tr 75.53% | Test 79.50%
⏹️ Early stopping at epoch 101 (patience=5)
✅ Best Test Accuracy: 79.66%


---

### 📈 Accuracy Curve
Shows a balanced and regularized training pattern with good generalization.  
*(Plot displayed automatically at the end of notebook)*

---

### ⚠️ Notes
- Fully implemented **from scratch** (no pretrained models)  
- Code structured into clear sections: Dataset, Model, Training, Evaluation, and Plotting  
- Trained entirely on **Colab GPU**

---

### 🔮 Future Improvements
- Larger **embedding dimension (256)**  
- Add **CutMix** or **Stochastic Depth**  
- Integrate **LayerScale or DropPath** for stronger regularization  

---
🧠 Short Analysis (Bonus)
🔍 Model Choices

🧩 Patch Size: 4×4 — captures fine spatial features on small 32×32 CIFAR-10 images.
🧱 Depth & Heads: 6 layers and 4 heads — balanced accuracy vs. compute.
🔡 Embedding Dim (128): captures both local & global relationships efficiently.
⚙️ Regularization & Training Tricks
🧊 Label Smoothing (0.1): prevents overconfidence in predictions.
🎛️ Mixup (α=0.2): improved generalization by blending examples.
🔁 EMA (Exponential Moving Average): stabilized weights and smoother convergence.
🌈 Cosine LR + Warmup: smooth start and adaptive decay of learning rate.
⏹️ Early Stopping (patience=5): halted training once accuracy plateaued.

📊 Results Comparison
Model Variant	Best Test Accuracy	Observation
🧠 Baseline ViT	81.7%	Slight overfitting beyond 80+ epochs
⚙️ Regularized ViT	79.66%	Stable, consistent convergence

💡 Key Insights

Regularization improved training stability and reduced overfitting.
Smaller patch size offered better fine-grained feature learning.
Increasing model depth or patch overlap can push accuracy >82%,
though with longer training and higher GPU cost.
The chosen setup achieved a strong balance of accuracy, stability, and efficiency.

## 🖼️ Q2 – Text-Driven Image Segmentation with SAM 2

### 📋 Overview
Upload an image and type a **text prompt** like `"cat"`, `"person"`, or `"car"`.  
The system automatically:
- 🔍 Generates intelligent geometric prompts  
- 🎨 Segments the object using **Meta’s SAM 2 (Segment Anything Model 2)**  
- 📊 Displays mask, overlay, and quality statistics  

---

### 🚀 Quick Start
1. Run installation cells  
2. Upload your image  
3. Enter a **text prompt**  
4. Get instant segmentation mask and overlay results  

---

### 🧠 How It Works
**Pipeline:**  
`Text Prompt → Geometric Prompts → SAM 2 → Segmentation Mask`

#### Three-Stage Process:
1. **Text-to-Prompt Conversion:**  
   - Converts text into heuristic prompt sets (center, grid, box)
2. **SAM 2 Segmentation:**  
   - Runs *Hiera-Large* model (879 MB)  
   - Selects mask with highest confidence score
3. **Visualization & Export:**  
   - Displays original image + mask overlay  
   - Prints segmentation statistics  
   - Saves `.png` output files  

---

### 💡 Prompt Strategies

| Strategy | Description | Ideal For |
|-----------|--------------|------------|
| **Center Point** | One prompt at image center | Centered objects |
| **Grid Points** | 5 distributed prompts | Multi-object scenes |
| **Center Box** | 60% region bounding box | Large objects |

✅ System automatically picks the best-performing method.

---

### ✨ Features
- ✅ Fully Automatic – no manual annotation  
- ✅ Multi-Strategy Prompting  
- ✅ Confidence-based Quality Scoring  
- ✅ GPU Accelerated for fast inference  
- ✅ Exports `mask_*.png` and `overlay_*.png`  

---

### 📊 Sample Output
Segmentation Statistics:
Text Prompt: 'cat'
Method Used: Center Point
Quality Score: 0.987
Mask Coverage: 23.45%
Image Size: 1024x768


---

### 🎮 Usage (Google Colab)
1. Open notebook in Colab  
2. Enable GPU: `Runtime → Change runtime type → GPU`  
3. Run setup and upload your image  
4. Type text prompt (e.g. `"dog"`, `"bottle"`, `"car"`)  
5. View mask and overlay automatically  

---

### ⚠️ Limitations
- Uses **geometric heuristics**, no semantic grounding  
- May miss **small or off-center** objects  
- Requires **GPU** for acceptable speed (~2–5 sec/image)  
- Only supports **static images** (no video segmentation)

---

### 📁 Output Files
| File | Description |
|------|--------------|
| `mask_[prompt].png` | Binary segmentation mask |
| `overlay_[prompt].png` | Colored overlay result |

---

### 🔮 Future Improvements
- Integrate **GroundingDINO** for text-to-box mapping  
- Add **CLIPSeg** for semantic prompt generation  
- Support **multi-object** & **video segmentation**  
- Improve **text-to-point alignment** with CLIP features  

---

## 🧾 Repository Summary

| File | Description |
|------|--------------|
| `q1.ipynb` | Vision Transformer (CIFAR-10) |
| `q2.ipynb` | SAM 2 Text-Driven Image Segmentation |
| `README.md` | Project Documentation & Results |


