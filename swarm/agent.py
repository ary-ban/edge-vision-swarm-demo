# swarm/agent.py
# Run one "agent" (drone/UGV) that:
# - opens a video source
# - runs ONNX YOLOv8n detection (FP32)
# - sends heartbeat + detections to Coordinator
# - receives assignment and computes a pan/tilt "error" (stub for gimbal)

import argparse, time, cv2, numpy as np, onnxruntime as ort
from pathlib import Path
from common import (
    COORD_LISTEN, make_rx_socket, make_tx_socket, send_json, recv_json,
    msg_heartbeat, msg_detections
)

# --- display helpers to fit window on screen ---
def get_screen_size():
    # Works on Windows; falls back to 1920x1080 elsewhere
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080

def fit_to_screen(w, h, max_frac=0.90):
    sw, sh = get_screen_size()
    scale = min((sw * max_frac) / max(w, 1), (sh * max_frac) / max(h, 1), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return new_w, new_h, scale

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
IMG = 640

def letterbox(im, new_shape=(IMG, IMG), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    nw, nh = int(round(w*r)), int(round(h*r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0]-nh)//2; bottom = new_shape[0]-nh-top
    left = (new_shape[1]-nw)//2; right = new_shape[1]-nw-left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    img = cv2.cvtColor(im_padded, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))[None, ...]
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

def parse_yolov8_output(out):
    pred = out
    if pred.ndim == 3:
        if pred.shape[1] in (84, 85):
            pred = np.transpose(pred, (0,2,1))
        pred = pred[0]
    C = pred.shape[1]
    if C == 84:
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]; obj = None; has_obj = False
    elif C == 85:
        boxes_xywh = pred[:, :4]
        obj = pred[:, 4]; cls_scores = pred[:, 5:]; has_obj = True
    else:
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, -80:]; obj = None; has_obj = False
    cls_max = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_max * obj if has_obj else cls_max
    xy = boxes_xywh.copy()
    xy[:,0] = boxes_xywh[:,0] - boxes_xywh[:,2]/2
    xy[:,1] = boxes_xywh[:,1] - boxes_xywh[:,3]/2
    xy[:,2] = boxes_xywh[:,0] + boxes_xywh[:,2]/2
    xy[:,3] = boxes_xywh[:,1] + boxes_xywh[:,3]/2
    return xy, scores, class_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="agent ID, e.g. A1")
    ap.add_argument("--listen", type=int, required=True, help="agent's UDP port to receive assignments")
    ap.add_argument("--model", required=True, help="path to ONNX model")
    ap.add_argument("--source", default="0", help="0 for webcam or path to video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.45)
    args = ap.parse_args()

    # UDP sockets
    rx = make_rx_socket(("127.0.0.1", args.listen))  # agent listens here
    tx = make_tx_socket()                            # send to coordinator

    # Detector
    sess = ort.InferenceSession(args.model, providers=[('CPUExecutionProvider', {})])
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    # Create a resizable window for this agent
    win_name = f'Agent {args.id}'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    last_hb = 0
    assign_mode = "none"

    while True:
        # Heartbeat @ 2 Hz
        now = time.time()
        if now - last_hb > 0.5:
            send_json(tx, COORD_LISTEN, msg_heartbeat(args.id, args.listen))
            last_hb = now

        ok, frame = cap.read()
        if not ok: break
        (oh, ow) = frame.shape[:2]

        # Detect
        inp, (dx,dy), _ = letterbox(frame)
        t0 = time.time()
        out = sess.run(None, {input_name: inp})[0]
        boxes640, scores, class_ids = parse_yolov8_output(out)
        gain = min(IMG/oh, IMG/ow)
        boxes = boxes640.copy()
        boxes[:, [0,2]] -= dx
        boxes[:, [1,3]] -= dy
        boxes /= gain
        keep = nms(boxes, scores, iou_thr=args.iou, score_thr=args.conf)

        # Pick best detection (our "target") for demo
        target = None
        if len(keep):
            i = keep[0]  # highest score after sort
            x1,y1,x2,y2 = boxes[i]
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            target = {
                "c": (COCO_NAMES[class_ids[i]] if 0 <= class_ids[i] < len(COCO_NAMES) else str(int(class_ids[i]))),
                "p": float(scores[i]),
                "cx": int(cx), "cy": int(cy),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            }

        # Send detections
        send_json(tx, COORD_LISTEN, msg_detections(args.id, [target] if target else []))

        # Read assignment (non-blocking)
        msg, _ = recv_json(rx)
        if msg and msg.get("t") == "assign" and msg.get("to") == args.id:
            assign_mode = msg.get("mode", "track_best")

        # Draw overlay
        vis = frame.copy()
        if target:
            x1,y1,x2,y2 = target["box"]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f'{target["c"]} {target["p"]:.2f}', (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # gimbal error (stub): normalized error from image center
            midx, midy = ow//2, oh//2
            ex = (target["cx"] - midx) / max(midx,1)
            ey = (target["cy"] - midy) / max(midy,1)
            cv2.line(vis, (midx, midy), (target["cx"], target["cy"]), (0,255,255), 2)
            cv2.putText(vis, f'err:({ex:+.2f},{ey:+.2f}) mode:{assign_mode}',
                        (10, oh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(vis, f'No target | mode:{assign_mode}', (10, oh-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(vis, f'Agent {args.id} | FPS:{fps:.1f}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # ---- Auto-scale the display to fit the screen (no cropping) ----
        vh, vw = vis.shape[:2]
        disp_w, disp_h, _ = fit_to_screen(vw, vh, max_frac=0.90)
        if (disp_w, disp_h) != (vw, vh):
            vis_disp = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            vis_disp = vis
        cv2.resizeWindow(win_name, disp_w, disp_h)
        cv2.moveWindow(win_name, 20, 20)  # tweak if you want a different position
        cv2.imshow(win_name, vis_disp)
        # ----------------------------------------------------------------

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
