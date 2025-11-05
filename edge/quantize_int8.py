# edge/quantize_int8.py
# Dynamic quantization for ONNXRuntime 1.17.x (no 'optimize_model' arg).
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

here = Path(__file__).resolve().parent
src = here / 'yolov8n.onnx'
dst = here / 'yolov8n-int8.onnx'

if not src.exists():
    raise SystemExit(f"Missing {src}. Run export_onnx.py first (same folder).")

quantize_dynamic(
    model_input=str(src),
    model_output=str(dst),
    weight_type=QuantType.QInt8,   # or QuantType.QUInt8 if you prefer
    per_channel=False,             # keep simple & portable
    reduce_range=False
)

print(f"Quantized model written to {dst}")
