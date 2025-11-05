# Edge Vision Swarm Demo

Two "agents" (like drones/cameras) detect objects with YOLOv8 (ONNX) and send detections to a coordinator.  
The coordinator fuses detections on a shared horizontal axis and uses the Hungarian algorithm to assign different targets to different agents.

## Quick start

```bash
# 1) create venv & install
pip install -r requirements.txt

# 2) export ONNX
python export_onnx.py   # creates edge/yolov8n.onnx

# 3) run coordinator
python swarm/coordinator.py

# 4) run agents (two terminals)
python swarm/agent.py --id A1 --listen 9101 --model edge/yolov8n.onnx --source video_left.mp4  --fov_scale_x 1.0 --fov_offset_x 0
python swarm/agent.py --id A2 --listen 9102 --model edge/yolov8n.onnx --source video_right.mp4 --fov_scale_x 1.0 --fov_offset_x 1280
