# Space Station Safety Object Detection

###  OVERVIEW
This project uses the YOLOv8 object detection framework to identify critical safety objects inside space station environments. The objects detected include fire extinguishers, alarms, oxygen tanks, control panels, and other essential safety equipment to ensure crew safety.

The dataset consists of images collected from simulated space station environments, labeled for these key objects. The model was trained on CPU, utilizing the YOLOv8n architecture, and fine-tuned for improved accuracy.

## Project Structure
```
Spacestation_Object_Detection/
├── data/
│   └── data.yaml
├── report/
│   ├── detected_samples/
│   │   ├── sample1.png
│   │   ├── sample2.png
│   │   ├── Final_Report.pdf.pdf
│   │   ├── precision_recall.png
│   │   └── training_performance.png
├── runs/
│   └── detect/
│       ├── comparison_results/
│       ├── predict50/
│       ├── train30/
│       ├── train50/
│       ├── val30/
│       └── val50/
├── scripts/
│   ├── analyze_results_30_50.py
│   ├── demo_app.py
│   ├── predict_model.py
│   └── train_model.py
├── .gitattributes
├── README.md
└── requirements.txt
```
---

## Setup and Installation

### 1. Clone or download this repository

```bash
git clone https://github.com/nitinpcs/Spacestation_Object_Detection.git
cd Spacestation_Object_Detection
```

### 2. Create and activate your Python environment (Anaconda recommended)

```bash
conda create -n spacestation python=3.10 -y
conda activate spacestation
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Requirements

- **Python:** 3.10.x
- **Operating System:** Windows 10/11, Linux, macOS
- **Hardware:** CPU-based (Intel i5-12500H or equivalent)
- **RAM:** Minimum 8GB (16GB recommended)
- **Disk Space:** ~5GB for dataset and model weights

---

## How to Reproduce Final Results

Follow these exact steps to reproduce the final model performance (mAP@0.5: 0.69):

### Step 1: Initial Training (30 Epochs)

```bash
yolo train model=yolov8n.pt data=data/data.yaml epochs=30 imgsz=640 device=cpu
```

**Expected Output:**
- Training completes in ~45-60 minutes (CPU)
- Model weights saved in `runs/detect/train30/weights/best.pt`
- Initial mAP@0.5: ~0.58

### Step 2: Fine-Tuning (50 Additional Epochs)

```bash
yolo train model=runs/detect/train30/weights/best.pt data=data/data.yaml epochs=50 imgsz=640 device=cpu scale=0.5 translate=0.1
```

**Expected Output:**
- Training completes in ~75-90 minutes (CPU)
- Final model weights saved in `runs/detect/train50/weights/best.pt`
- **Final mAP@0.5: 0.69** (18.6% improvement)

### Step 3: Validate the Final Model

```bash
yolo val model=runs/detect/train50/weights/best.pt data=data/data.yaml device=cpu
```

**Expected Output:**
```
mAP@0.5: 0.69
mAP@0.5:0.95: 0.45
Precision: 0.765
Recall: 0.709
```

### Step 4: Run Predictions on Test Set

```bash
yolo predict model=runs/detect/train50/weights/best.pt source=data/images/test save=True device=cpu
```

**Or use the prediction script:**

```bash
python scripts/predict_model.py
```

**Expected Output:**
- Annotated images saved in `runs/detect/predict50/`
- Each image shows bounding boxes with class labels and confidence scores
- Average inference time: ~35ms per image

### Step 5: Generate Performance Analysis

```bash
python scripts/analyze_results_30_50.py
```

**Expected Output:**
- Comparison charts saved in `runs/detect/comparison_results/`
- Files generated:
  - `mAP50_comparison.png`
  - `precision_recall_comparison.png`
  - `loss_comparison.png`
  - `comparison_summary.csv`

---

## Running and Testing the Model

### Option 1: Command Line Prediction

```bash
# Single image prediction
yolo predict model=runs/detect/train50/weights/best.pt source=path/to/image.jpg save=True

# Batch prediction (entire folder)
yolo predict model=runs/detect/train50/weights/best.pt source=data/images/test save=True
```

### Option 2: Python Script

```bash
python scripts/predict_model.py
```

### Option 3: Interactive Web App

```bash
streamlit run scripts/demo_app.py
```

Then:
1. Open browser at `http://localhost:8501`
2. Upload a space station image
3. View real-time detection results with bounding boxes

---

## Expected Outputs and Interpretation

### 1. Training Outputs (`results.csv`)

Located in: `runs/detect/train30/` and `runs/detect/train50/`

| Column | Description | Good Value |
|--------|-------------|------------|
| `metrics/mAP50(B)` | Mean Average Precision at IoU=0.5 | > 0.65 |
| `metrics/mAP50-95(B)` | mAP across IoU 0.5-0.95 | > 0.40 |
| `metrics/precision(B)` | Precision (fewer false positives) | > 0.70 |
| `metrics/recall(B)` | Recall (fewer missed detections) | > 0.65 |
| `train/box_loss` | Bounding box regression loss | Decreasing trend |
| `train/cls_loss` | Classification loss | Decreasing trend |

**How to Interpret:**
- **mAP@0.5 = 0.69** means the model correctly detects 69% of objects with at least 50% overlap
- **Precision = 0.765** means 76.5% of detections are correct (low false alarms)
- **Recall = 0.709** means 70.9% of actual objects are detected (few missed objects)

### 2. Validation Outputs

After running `yolo val`, you'll see:

```
Class     Images  Instances  Precision  Recall  mAP50  mAP50-95
all          150        450      0.765   0.709  0.690     0.450
oxygen_tank   22         45      0.812   0.778  0.801     0.521
fire_ext      28         52      0.789   0.731  0.765     0.485
alarm         19         38      0.734   0.684  0.698     0.412
...
```

**How to Interpret:**
- Each class shows individual performance
- Classes with lower mAP may need more training data
- Overall mAP is weighted average across all classes

### 3. Prediction Outputs

Images saved in `runs/detect/predict50/` with:
- **Green bounding boxes** around detected objects
- **Class labels** (e.g., "oxygen_tank", "fire_extinguisher")
- **Confidence scores** (e.g., 0.87 = 87% confidence)

**Interpretation Guidelines:**
- Confidence > 0.75: High confidence detection (very reliable)
- Confidence 0.5-0.75: Medium confidence (generally reliable)
- Confidence < 0.5: Low confidence (may be false positive)

### 4. Performance Graphs

#### `results.png` (Auto-generated by YOLO)
Shows training progression over epochs:
- **Top row:** Loss curves (should decrease)
- **Bottom row:** mAP, Precision, Recall curves (should increase or stabilize)

#### `confusion_matrix.png`
- **Diagonal values** should be high (correct predictions)
- **Off-diagonal values** show misclassifications between classes

#### `PR_curve.png` (Precision-Recall Curve)
- **Area under curve** indicates model quality
- Higher area = better balance between precision and recall

---

## Performance Metrics Summary

### Final Model Performance (50 Epochs)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **mAP@0.5** | 0.69 | 69% average detection accuracy at 50% IoU threshold |
| **mAP@0.5:0.95** | 0.45 | 45% average across stricter IoU thresholds (0.5 to 0.95) |
| **Precision** | 0.765 | 76.5% of detections are true positives |
| **Recall** | 0.709 | 70.9% of actual objects are detected |
| **Inference Speed** | ~35ms | Real-time capable (<50ms requirement) |
| **Improvement** | +18.6% | Relative improvement from initial 58.2% to final 69% |

### Benchmark Comparison

| Benchmark | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Minimum mAP@0.5 | 0.50 | 0.69 | ✅ Exceeded |
| Real-time inference | <50ms | ~35ms | ✅ Exceeded |
| Precision | >0.70 | 0.765 | ✅ Met |
| Recall | >0.65 | 0.709 | ✅ Met |

---

## Troubleshooting

### Issue: Model not loading
**Solution:** Ensure correct path to `best.pt` file:
```bash
ls runs/detect/train50/weights/best.pt
```

### Issue: Low detection accuracy
**Possible causes:**
- Image quality too low (ensure >640px resolution)
- Objects too small (model trained on specific object sizes)
- Lighting conditions very different from training data

### Issue: Slow inference speed
**Solution:** Model is CPU-optimized. For faster inference:
```bash
# Reduce image size
yolo predict model=best.pt source=image.jpg imgsz=416
```

### Issue: Installation errors
**Solution:** Ensure Python 3.10 and all dependencies installed:
```bash
python --version  # Should be 3.10.x
pip list | grep ultralytics  # Should show ultralytics package
```

---

## Key Features

- ✅ **Multi-class detection**: Identifies 7 different safety objects
- ✅ **CPU-optimized**: Trained entirely on CPU without GPU requirements
- ✅ **Fine-tuned model**: Sequential training approach improves accuracy by 18.6%
- ✅ **Data augmentation**: Scale and translation augmentation for robustness
- ✅ **Interactive demo**: Streamlit app for real-time inference
- ✅ **Reproducible results**: Complete training pipeline documented
- ✅ **Performance metrics**: Comprehensive evaluation and benchmarking

---

## Team - Hackstreet Boys

- **Nitin Pachauri** (Team Leader)
- **Ram Lal**
- **Moksh Upadhyay**
- **Aditya Shukla**

B.Tech CSE, 3rd Year

---

## Acknowledgments

- **Dataset:** Falcon (Duality AI) Space Station Safety Objects
- **Framework:** Ultralytics YOLOv8
- **Platform:** Anaconda, Python 3.10
- **Hardware:** Intel i5-12500H CPU

---

 
