# realtime_app.py
# Section 5: Real-Time Demonstration (PyCharm)
# I capture webcam frames, run my trained MobileNetV2, and overlay the predicted class as a subtitle.

import argparse
from collections import deque
from pathlib import Path
import time

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

def build_mobilenetv2(n_classes: int):
    m = models.mobilenet_v2(weights=None)  # weights are loaded from my checkpoint
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, n_classes)
    return m

def draw_subtitle(frame, text, color=(50, 220, 50)):
    """I draw a black bar at the bottom with the prediction text."""
    h, w = frame.shape[:2]
    bar_h = 40
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to .pth from Colab (contains state_dict + classes)")
    ap.add_argument("--img-size", type=int, default=160, help="Resize used in training")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--ema", type=float, default=0.6, help="Exponential smoothing of probs (0=off)")
    ap.add_argument("--avgwin", type=int, default=5, help="Moving-average window of logits (>=1)")
    ap.add_argument("--imagenet_norm", action="store_true", help="Use ImageNet mean/std if you trained with normalization")
    ap.add_argument("--threshold", type=float, default=0.60, help="Min confidence to show a label; else show 'unknown'")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))
    print(f"Using device: {device}")

    # ---- load checkpoint ----
    ckpt = torch.load(args.weights, map_location=device)
    classes = ckpt.get("classes", None)
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes'. Re-save as {'state_dict':..., 'classes':[...]}")

    print("Loaded classes:", classes)  # <<< verify order here

    model = build_mobilenetv2(n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---- transforms (match training) ----
    if args.imagenet_norm:
        mean = [0.485, 0.456, 0.406]; std  = [0.229, 0.224, 0.225]
        tfm = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ])

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try --camera 1.")

    print("Controls: Q/ESC=quit | S=save frame | F=toggle FPS | H=toggle subtitle | R=toggle ROI | C=clear smoothing | D=debug")
    show_fps = True
    show_sub = True
    use_roi   = True   # start with ROI enabled
    debug     = False
    out_dir = Path("captures"); out_dir.mkdir(parents=True, exist_ok=True)

    softmax = nn.Softmax(dim=1)
    ema_prob = None
    buf = deque(maxlen=max(1, args.avgwin))
    t_last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame."); break

        H, W = frame.shape[:2]

        # Center square ROI (focus box)
        if use_roi:
            side = min(H, W) * 0.6  # 60% of min dim
            side = int(side)
            cx, cy = W // 2, H // 2
            x1, y1 = cx - side // 2, cy - side // 2
            x2, y2 = x1 + side, y1 + side
            roi = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)].copy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            to_process = roi
        else:
            to_process = frame

        # BGR -> RGB for torchvision
        rgb = cv2.cvtColor(to_process, cv2.COLOR_BGR2RGB)
        pil = transforms.functional.to_pil_image(rgb)
        x = tfm(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)               # [1, C]
            buf.append(logits.cpu())
            avg_logits = torch.stack(list(buf), dim=0).mean(dim=0).to(device)
            probs = softmax(avg_logits)     # [1, C]

            # EMA smoothing
            if args.ema > 0:
                ema_prob = probs if ema_prob is None else (args.ema * ema_prob + (1 - args.ema) * probs)
                probs_out = ema_prob
            else:
                probs_out = probs

            # top-k
            confs, idxs = torch.topk(probs_out, k=min(2, len(classes)), dim=1)
            conf, pred_idx = confs[0,0].item(), idxs[0,0].item()
            label = classes[pred_idx]

        # Confidence gate → unknown
        if conf < args.threshold:
            show_label = f"unknown  ({conf*100:.1f}%)"
        else:
            show_label = f"{label}  ({conf*100:.1f}%)"

        if show_sub:
            draw_subtitle(frame, show_label)

        # optional debug (top-2)
        if debug:
            txt = f"top1: {classes[idxs[0,0]]} {confs[0,0].item():.2f} | top2: {classes[idxs[0,1]]} {confs[0,1].item():.2f}"
            cv2.putText(frame, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

        if show_fps:
            t = time.time(); fps = 1.0 / max(1e-6, (t - t_last)); t_last = t
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 3, cv2.LINE_AA)

        cv2.imshow("Real-Time Object Recognition", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('s'):
            p = out_dir / f"frame_{int(time.time())}.jpg"; cv2.imwrite(str(p), frame); print("Saved:", p)
        elif k == ord('f'):
            show_fps = not show_fps
        elif k == ord('h'):
            show_sub = not show_sub
        elif k == ord('r'):
            use_roi = not use_roi
        elif k == ord('c'):
            buf.clear(); ema_prob = None; print("Smoothing cleared.")
        elif k == ord('d'):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
