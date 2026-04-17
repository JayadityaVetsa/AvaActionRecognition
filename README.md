# Single-Frame Atomic Action Recognition (AVA v2.2)

> *"Can pretrained CNNs recognize atomic human actions from single frames? I benchmarked a from-scratch CNN vs. a fine-tuned EfficientNet-B2 on a curated AVA v2.2 subset and analyzed where frame-level models fail."*

## 📌 Project Overview
Action recognition in video is a complex problem, primarily because it requires understanding both *spatial* features (what is in the scene) and *temporal* features (how things move over time). 

This project explores a fundamental question: **How far can I get with purely spatial features?** Using a curated 15-class subset of the [AVA v2.2 dataset](https://research.google.com/ava/) (a dataset of localized atomic actions from movies), I extracted single frames of localized actors to evaluate the capability (and limitations) of 2D Convolutional Neural Networks in classifying human actions without any temporal context.

## 🔬 Experimental Setup
- **Dataset**: A curated subset of the AVA v2.2 dataset, utilizing single extracted face/body crops based on bounding boxes.
- **Classes**: 15 distinct atomic actions, highly imbalanced, ranging from static postures (`sit`, `stand`) to subtle object interactions (`smoke`, `drink`).
- **Models Benchmarked**:
  1. **Baseline CNN**: A custom 4-block Convolutional Neural Network trained entirely from scratch. (~3.27M parameters)
  2. **EfficientNet-B2**: Pretrained on ImageNet, fine-tuned with a customized classification head. I froze the stem and early layers to retain generic low-level feature extractors while fine-tuning the deeper semantic layers. (~8.43M parameters)

## 📊 Results & Analysis

### Performance Comparison
| Model | Params | Val Accuracy | mAP |
|---|---|---|---|
| Baseline CNN (from scratch) | 3.2M | 3.81% | 6.99% |
| **EfficientNet-B2 (fine-tuned)** | 8.4M | **22.58%** | **12.99%** |

*The AVA dataset is notoriously difficult—even state-of-the-art 3D spatiotemporal models struggle to achieve high mAP due to severe long-tail distribution and multi-label complexity. For a single-frame 2D approach, these numbers highlight both the immense value of transfer learning and the strict ceiling of spatial-only analysis.*

### Where do single-frame models fail? (The "Why")
Analyzing the per-class Average Precision (AP) for our best model (EfficientNet-B2) reveals exactly what temporal context contributes to action recognition:

1. **Static Postures are (Relatively) Solvable**: The model performs best on static poses that are visually distinct and stable in a single frame. 
   - `sit` (AP: 28.9%)
   - `stand` (AP: 28.0%)
   - `lie/sleep` (AP: 19.4%)
2. **Context-Heavy Actions Perform Okay**: Actions heavily correlated with proximity to others or specific facial alignments extracted enough semantic context to do moderately well.
   - `talk to` (AP: 25.9%)
   - `watch` (AP: 22.1%)
3. **Motion and Object-Interactions Fail Completely**: This is the crux of the 2D limitation. Actions that require an understanding of an object's state change or subtle frame-to-frame biomechanical movement failed spectacularly.
   - `smoke` (AP: 0.04%)
   - `read` (AP: 0.04%)
   - `drink` (AP: 0.48%)

**Conclusion:** Without a temporal dimension (like optical flow or 3D convolutions), the network simply cannot distinguish the act of holding a cigarette from *smoking* it, looking at a cup versus actively *drinking* from it, or holding a book versus *reading* it. 

## ⚙️ Repository Structure
- `notebooks/`: Contains the training pipelines and logic:
  - Data preparation and manifest routing.
  - Baseline CNN definition and training loop.
  - EfficientNet-B2 transfer learning pipeline.
- `models/`: Saved model weights (`.pth`) and training history/metrics (`.json`).
- `results/`: Contains generated visualizations (training curves, confusion matrices, AP charts).

## 🚀 Future Directions
As demonstrated by the stark drop-off in AP for motion-dependent classes, spatial features alone are insufficient for robust action recognition. The logical next step in this research pipeline is to introduce the temporal dimension by implementing architectures such as **SlowFast** networks or 3D ResNets on sequential video clips.