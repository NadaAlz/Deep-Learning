import cv2
from pathlib import Path

ROOT = Path("data/raw")

def review_folder(folder: Path):
    imgs = sorted(folder.glob("*.jpg")); i = 0
    while 0 <= i < len(imgs):
        p = imgs[i]; img = cv2.imread(str(p))
        if img is None: p.unlink(missing_ok=True); imgs.pop(i); continue
        view = img.copy()
        cv2.putText(view, f"{folder.name}/{p.name} [{i+1}/{len(imgs)}]  (k=keep, d=del, q/ESC, ←/→)",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Review & Filter", view)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'),27): break
        elif key == ord('d'): p.unlink(missing_ok=True); imgs.pop(i)
        elif key == ord('k'): i += 1
        elif key in (81, ord('a')): i = max(0, i-1)
        elif key in (83, ord('l')): i = min(len(imgs)-1, i+1)
        else: i += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    for cls in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        print(f"Reviewing: {cls.name}"); review_folder(cls)
    print("Manual filtering done.")
