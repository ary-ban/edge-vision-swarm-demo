from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO('yolov8s.pt')  # "small" model
    model.export(format='onnx', opset=12, imgsz=640, dynamic=True)
    print("Exported: yolov8s.onnx")

if __name__ == "__main__":
    main()
