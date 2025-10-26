"""
predict_model.py
----------------
Run YOLO predictions on test images using subprocess.
"""

import subprocess

# Run predictions on test images
subprocess.run([
    "yolo", "predict",
    "model=runs/detect/train50/weights/best.pt",
    "source=data/images/test",
    "conf=0.25",
    "iou=0.45",
    "imgsz=640",
    "device=cpu",
    "save=True",
    "save_txt=True",
    "save_conf=True",
    "project=runs/detect",
    "name=predict"
])

print("\n✅ Predictions completed successfully!")
print("Results saved in: runs/detect/predict/")
