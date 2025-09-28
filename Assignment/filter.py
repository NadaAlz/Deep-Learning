import cv2
from pathlib import Path

# I keep my raw dataset inside data/raw
ROOT = Path("data/raw")


def review_folder(folder: Path):
    # I collect all jpg images inside the given folder
    imgs = sorted(folder.glob("*.jpg"))
    i = 0  # I start reviewing from the first image
    while 0 <= i < len(imgs):
        p = imgs[i]
        img = cv2.imread(str(p))

        # If I can’t load the image (corrupted), I delete it immediately
        if img is None:
            p.unlink(missing_ok=True)
            imgs.pop(i)
            continue

        # I make a copy so I can overlay text without ruining the original
        view = img.copy()
        cv2.putText(
            view,
            f"{folder.name}/{p.name} [{i + 1}/{len(imgs)}]  (k=keep, d=del, q/ESC, ←/→)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # I show the current image with instructions
        cv2.imshow("Review & Filter", view)
        key = cv2.waitKey(0) & 0xFF  # I wait for a key press

        # I check what key I pressed
        if key in (ord('q'), 27):  # If I press q or ESC, I stop reviewing
            break
        elif key == ord('d'):  # If I press d, I delete the image
            p.unlink(missing_ok=True)
            imgs.pop(i)
        elif key == ord('k'):  # If I press k, I keep the image and move forward
            i += 1
        elif key in (81, ord('a')):  # Left arrow or 'a' → I go back
            i = max(0, i - 1)
        elif key in (83, ord('l')):  # Right arrow or 'l' → I go forward
            i = min(len(imgs) - 1, i + 1)
        else:
            # If I press anything else, I just move to the next
            i += 1

    # Once I’m done with this folder, I close the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # I go through each class folder under data/raw
    for cls in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        print(f"Reviewing: {cls.name}")
        review_folder(cls)

    # At the end, I confirm that I’m finished
    print("Manual filtering done.")
