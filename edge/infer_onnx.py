# Minimal ONNXRuntime YOLOv8n demo (simple & light)
# Run: --model yolov8n.onnx --source 0
import argparse, time, cv2, numpy as np, onnxruntime as ort

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

IMG = 640  # fixed input size for simplicity

def letterbox(im, new_shape=(IMG, IMG), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w*r)), int(round(h*r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0]-nh)//2; bottom = new_shape[0]-nh-top
    left = (new_shape[1]-nw)//2; right = new_shape[1]-nw-left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    img = cv2.cvtColor(im_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None, ...]  # 1x3x640x640
    return img, (left, top), r

def iou_with(i, js, boxes):
    if len(js) == 0: return np.array([])
    a = boxes[i]; b = boxes[js]
    x1 = np.maximum(a[0], b[:,0]); y1 = np.maximum(a[1], b[:,1])
    x2 = np.minimum(a[2], b[:,2]); y2 = np.minimum(a[3], b[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)

def nms(boxes, scores, iou_thr=0.45, score_thr=0.25):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        if scores[i] < score_thr: break
        keep.append(i)
        iou = iou_with(i, idxs[1:], boxes)
        idxs = idxs[1:][iou < iou_thr]
    return keep

def load_session(model_path):
    return ort.InferenceSession(model_path, providers=[('CPUExecutionProvider', {})])

def parse_yolov8_output(out: np.ndarray):
    # Handles (1,84,N) or (1,N,84) or (1,N,85)
    pred = out
    if pred.ndim == 3:
        if pred.shape[1] in (84, 85):
            pred = np.transpose(pred, (0, 2, 1))
        pred = pred[0]  # (N,C)
    C = pred.shape[1]
    if C == 84:
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]    # (N,80)
        obj = None; has_obj = False
    elif C == 85:
        boxes_xywh = pred[:, :4]
        obj = pred[:, 4]            # (N,)
        cls_scores = pred[:, 5:]    # (N,80)
        has_obj = True
    else:
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, -80:]
        obj = None; has_obj = False
    # scores & labels
    cls_max = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_max * obj if has_obj else cls_max
    # xywh -> xyxy (in 640 space)
    xy = boxes_xywh.copy()
    xy[:,0] = boxes_xywh[:,0] - boxes_xywh[:,2]/2
    xy[:,1] = boxes_xywh[:,1] - boxes_xywh[:,3]/2
    xy[:,2] = boxes_xywh[:,0] + boxes_xywh[:,2]/2
    xy[:,3] = boxes_xywh[:,1] + boxes_xywh[:,3]/2
    return xy, scores, class_ids

def draw(frame, boxes, scores, class_ids, keep):
    for i in keep:
        x1,y1,x2,y2 = boxes[i].astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
        label = COCO_NAMES[class_ids[i]] if 0 <= class_ids[i] < len(COCO_NAMES) else str(int(class_ids[i]))
        txt = f"{label} {scores[i]:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, txt, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.onnx")
    ap.add_argument("--source", default="0", help="0 for webcam or path to video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.45)
    args = ap.parse_args()

    sess = load_session(args.model)
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    conf_thr = float(args.conf)
    iou_thr  = float(args.iou)

    while True:
        ok, frame = cap.read()
        if not ok: break

        inp, (dx,dy), _ = letterbox(frame, (IMG, IMG))
        t0 = time.time()
        out = sess.run(None, {input_name: inp})[0]

        boxes640, scores, class_ids = parse_yolov8_output(out)
        # undo letterbox to original frame size
        oh, ow = frame.shape[:2]
        gain = min(IMG/oh, IMG/ow)
        boxes = boxes640.copy()
        boxes[:, [0,2]] -= dx
        boxes[:, [1,3]] -= dy
        boxes /= gain

        keep = nms(boxes, scores, iou_thr=iou_thr, score_thr=conf_thr)
        vis = draw(frame.copy(), boxes, scores, class_ids, keep)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(vis, f"FPS: {fps:.1f} | Conf:{conf_thr:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("YOLOv8 ONNX (simple)", vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')): break  # ESC or Q to quit

    cap.release()
    cv2.destroyAllWindows()
