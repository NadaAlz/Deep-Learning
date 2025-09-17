# test.py  — single-image sanity check (PyCharm-friendly)
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def build_mobilenetv2(nc: int) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    return m

def load_model(weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt["classes"]
    model = build_mobilenetv2(len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="path to an image file (jpg/png)")
    ap.add_argument("--weights", default="best_model.pth", help="checkpoint with state_dict + classes")
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--imagenet_norm", action="store_true", help="only if you trained with ImageNet normalization")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available())
                          else (args.device if args.device!="auto" else "cpu"))

    # If no image arg, open a file picker (falls back to input()).
    if not args.image:
        try:
            from tkinter import Tk, filedialog
            Tk().withdraw()
            path = filedialog.askopenfilename(
                title="Select test image",
                filetypes=[("Images","*.jpg *.jpeg *.png *.bmp")]
            )
            if not path:
                print("No image selected. Exiting.")
                return
            args.image = path
        except Exception:
            path = input("Enter path to an image file: ").strip()
            if not path:
                print("No image path provided. Exiting.")
                return
            args.image = path

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    model, classes = load_model(args.weights, device)
    print("Loaded classes:", classes)

    if args.imagenet_norm:
        mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
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

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
        conf, idx = prob.max(dim=0)

    print(f"\nPrediction: {classes[idx.item()]}  confidence={conf.item():.3f}\n")
    print("Probabilities:")
    for c, p in zip(classes, prob.tolist()):
        print(f"  {c:>8} : {p:.3f}")

if __name__ == "__main__":
    main()


