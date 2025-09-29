# realtime_app.py — Section 5: Real-Time Demo (stable like test.py)
# I run MobileNetV2 on webcam frames with confidence gating, ROI, smoothing, and quick checks.

"""
How I run this:
    python realtime-app.py --weights runs/best_model.pth --img-size 160 --device auto --camera 0
Keys I use while running:
    Q/ESC = quit | R = toggle ROI | C = clear smoothing | D = toggle debug top-2
    G = print full probs for current frame | S = save current frame to ./captures
"""

import argparse, time
from collections import deque
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ----- model builder (same head as training) -----
def build_mobilenetv2(n_classes: int):
    # I load a MobileNetV2 backbone and replace the last linear layer to match my classes
    m = models.mobilenet_v2(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, n_classes)
    return m

def draw_subtitle(frame, text):
    # I draw a dark bar at the bottom so my subtitle stays readable
    h, w = frame.shape[:2]
    bar_h = 42
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    # I parse my runtime options
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best_model.pth (state_dict + classes)")
    ap.add_argument("--img-size", type=int, default=160, help="must match training (I used 160)")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.70, help="min prob to show a label; else 'unknown'")
    ap.add_argument("--avgwin", type=int, default=3, help="moving-average window for logits")
    ap.add_argument("--ema", type=float, default=0.3, help="EMA factor for probs (0=off)")
    args = ap.parse_args()

    # I choose my device automatically unless told otherwise
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available())
                          else (args.device if args.device!="auto" else "cpu"))

    # I load my trained checkpoint (expects {'state_dict', 'classes'})
    ckpt = torch.load(args.weights, map_location=device)
    classes = ckpt["classes"]
    print("Loaded classes:", classes)
    model = build_mobilenetv2(len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # I use the exact same transforms I used during training (no normalization by design)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    softmax = nn.Softmax(dim=1)

    # I open my webcam; on Windows I prefer CAP_DSHOW to avoid delays
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try --camera 1.")

    print("Controls: Q/ESC quit | R toggle ROI | C clear smoothing | D debug top2 | G grab still (print probs) | S save frame")
    show_fps, show_sub, use_roi, debug = True, True, True, False
    out_dir = Path("captures"); out_dir.mkdir(exist_ok=True, parents=True)

    # I keep short-term smoothing to stabilize predictions
    buf = deque(maxlen=max(1, args.avgwin))  # I average logits over this window
    ema_prob = None                          # I optionally smooth probabilities with EMA
    t_last = time.time()                     # I track FPS

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame."); break

        H, W = frame.shape[:2]

        # I optionally crop a centered ROI to reduce background distractions
        if use_roi:
            side = int(min(H, W) * 0.6)
            cx, cy = W // 2, H // 2
            x1, y1 = cx - side // 2, cy - side // 2
            x2, y2 = x1 + side, y1 + side
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            roi = frame[y1:y2, x1:x2].copy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            to_process = roi
        else:
            to_process = frame

        # I convert BGR→RGB, make a PIL image, then apply my transforms
        rgb = cv2.cvtColor(to_process, cv2.COLOR_BGR2RGB)
        pil = transforms.functional.to_pil_image(rgb)
        x = tfm(pil).unsqueeze(0).to(device)

        # I run inference without gradients, then smooth
        with torch.no_grad():
            logits = model(x)                 # [1, C]
            buf.append(logits.cpu())          # I buffer raw logits for moving average
            avg_logits = torch.stack(list(buf), dim=0).mean(dim=0).to(device)
            probs = softmax(avg_logits)       # [1, C]

            # I optionally apply EMA smoothing on top of the moving average
            if args.ema > 0:
                ema_prob = probs if ema_prob is None else (args.ema * ema_prob + (1 - args.ema) * probs)
                p = ema_prob[0]
            else:
                p = probs[0]

            # I take the top1 class and its confidence
            conf, idx = torch.max(p, dim=0)
            conf = float(conf.item()); idx = int(idx.item())
            label = classes[idx]

        # I gate low-confidence predictions as 'unknown' (honesty like test.py)
        if conf < args.threshold:
            text = f"unknown  ({conf*100:.1f}%)"
        else:
            text = f"{label}  ({conf*100:.1f}%)"

        # I draw my subtitle bar
        if show_sub:
            draw_subtitle(frame, text)

        # If I want quick intuition, I show top-2 predictions
        if debug:
            top2v, top2i = torch.topk(p, k=min(2, len(classes)))
            dbg = f"top1: {classes[int(top2i[0])]} {float(top2v[0]):.2f} | top2: {classes[int(top2i[1])]} {float(top2v[1]):.2f}"
            cv2.putText(frame, dbg, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

        # I show FPS so I can monitor performance
        if show_fps:
            t = time.time(); fps = 1.0 / max(1e-6, (t - t_last)); t_last = t
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        # I display the live window and handle key controls
        cv2.imshow("Real-Time Object Recognition", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):          # I quit
            break
        elif k == ord('r'):              # I toggle the centered ROI
            use_roi = not use_roi
        elif k == ord('c'):              # I clear smoothing buffers
            buf.clear(); ema_prob = None; print("Smoothing cleared.")
        elif k == ord('d'):              # I toggle debug overlay
            debug = not debug
        elif k == ord('s'):              # I save the current frame
            pth = out_dir / f"frame_{int(time.time())}.jpg"
            cv2.imwrite(str(pth), frame)
            print("Saved:", pth)
        elif k == ord('g'):              # I print full probs for this view (like test.py)
            with torch.no_grad():
                pp = (ema_prob[0] if ema_prob is not None else probs[0]).cpu().numpy().tolist()
            print("\n[Grab] full probabilities:")
            for c, val in sorted(zip(classes, pp), key=lambda z: z[1], reverse=True):
                print(f"  {c:>8} : {val:.3f}")

    # I clean up my video handle and windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
