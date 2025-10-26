import subprocess

# Run initial training
subprocess.run([
    "yolo", "train",
    "model=yolov8n.pt",
    "data=data/data.yaml",
    "epochs=30",
    "imgsz=640",
    "device=cpu"
])

# Run fine-tuning
subprocess.run([
    "yolo", "train",
    "model=runs/detect/train30/weights/best.pt",
    "data=data/data.yaml",
    "epochs=50",
    "imgsz=640",
    "device=cpu",
    "scale=0.5",
    "translate=0.1"
])
