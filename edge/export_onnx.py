# Export YOLOv8n to ONNX (fp32). Quantization is done in a separate script.
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')  # downloads pretrained weights on first run
    model.export(format='onnx', opset=12, imgsz=640, dynamic=True)
    # Creates 'yolov8n.onnx' in the working directory.
    print("Exported to yolov8n.onnx")

if __name__ == "__main__":
    main()