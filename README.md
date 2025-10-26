# Space Station Safety Object Detection

#  OVERVIEW
This project uses the YOLOv8 object detection framework to identify critical safety objects inside space station environments. The objects detected include fire extinguishers, alarms, oxygen tanks, control panels, and other essential safety equipment to ensure crew safety.

The dataset consists of images collected from simulated space station environments, labeled for these key objects. The model was trained on CPU, utilizing the YOLOv8n architecture, and fine-tuned for improved accuracy.

Project Structure
text
Spacestation_Object_Detection/
│
├── data/
│   └── data.yaml
│
├── runs/
│   └── detect/
│       ├── train30/              # Training run for 30 epochs
│       │   ├── results.csv
│       │   ├── weights/
│       │   │   ├── best.pt       # Best weights model after 30 epochs
│       │   │   └── last.pt
│       ├── train50/              # Fine-tuned training run for 50 epochs
│       └── predict/              # Prediction results folder
│
├── scripts/
│   ├── train_model.py           # Training commands and pipelines
|   ├── predict_model.py         # Prediction script for test images
│   ├── analyze_results_30_50.py # Comparison and visualization script
│   ├── demo_app.py              # Streamlit demo app for live detection
│

├── requirements.txt             # Required Python packages
├── README.md                   # This file
└── report/
    ├── training_performance.png
    ├── precision_recall.png
    ├── detected_samples/
    │   ├── sample1.png
    │   └── sample2.png
    └── Final_Report.pdf
Setup and Installation
Clone or download this repository.

Create and activate your Python environment (Anaconda recommended):

bash
conda create -n spacestation python=3.10 -y
conda activate spacestation
Install dependencies:

bash
pip install -r requirements.txt
Training the Model
Train YOLOv8n model for 30 epochs:

bash
yolo train model=yolov8n.pt data=data/data.yaml epochs=30 imgsz=640 device=cpu
Fine-tune from the best weights for another 50 epochs with augmentation:

bash
yolo train model=runs/detect/train30/weights/best.pt data=data/data.yaml epochs=50 imgsz=640 device=cpu scale=0.5 translate=0.1
Validation
Validate the model on validation data:

bash
yolo val model=runs/detect/train50/weights/best.pt data=data/data.yaml device=cpu
Prediction / Testing
Run predictions on test images:

bash
yolo predict model=runs/detect/train50/weights/best.pt source=data/images/test save=True device=cpu
Output images with bounding boxes will be saved in runs/detect/predict/.

Analyzing Results
Run the comparison and visualization script to generate performance graphs:

bash
python scripts/analyze_results.py
This produces comparison charts for mAP, precision, recall, and training loss, saved in:

text
runs/detect/comparison_results/
Running the Demo App
To launch the Streamlit demo app for live object detection:

bash
streamlit run scripts/demo_app.py
Upload any image to see bounding boxes drawn on detected safety objects.

Results Summary
Metric	Value
Training Epochs	30 + 50 (fine-tuning)
Model Architecture	YOLOv8n
Final mAP@0.5	0.69
Precision	See results.csv
Recall	See results.csv
Dataset	Falcon Space Station Safety Objects
Training Device	CPU (Intel i5-12500H)
The model achieved a significant improvement after fine-tuning from 0.58 to 0.69 mAP@0.5, demonstrating effective learning and optimization.

Key Features
Multi-class detection: Identifies 7 different safety objects

CPU-optimized: Trained entirely on CPU without GPU requirements

Fine-tuned model: Sequential training approach improves accuracy

Data augmentation: Scale and translation augmentation for robustness

Interactive demo: Streamlit app for real-time inference

Team(Hackstreet Boys)
Nitin Pachauri(Team Leader)
Ram Lal
Moksh Upadhyay
Aditya Shukla

B.Tech CSE, 3rd Year

Acknowledgments
Dataset: Falcon (Duality AI) Space Station Safety Objects

Framework: Ultralytics YOLOv8

Platform: Anaconda, Python 3.10
