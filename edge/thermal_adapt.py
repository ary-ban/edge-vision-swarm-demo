# Scaffold to fine-tune YOLOv8n on a thermal (IR) dataset once you have one.
# Provide a thermal.yaml dataset file (Ultralytics format), then un-comment train().
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    # model.train(data='thermal.yaml', epochs=20, imgsz=640, device=0, lr0=0.001)
    print("Edit thermal.yaml for your IR dataset and un-comment model.train(...).")

if __name__ == "__main__":
    main()
