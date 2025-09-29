# collect_images.py
import cv2
import os
from pathlib import Path

MIN_REQUIRED_PER_CLASS = 35       # I must keep at least this many per class after filtering (assignment rule)
DEFAULT_TARGET_PER_CLASS = 70     # I aim for ~2x so I can delete bad ones later

def create_folder(path: Path):
    # I create the folder (and parents) if it doesn't exist
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def focus_score(frame) -> float:
    # I estimate sharpness using variance of Laplacian (higher = sharper)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def collect_images():
    # I enter my four classes once, comma-separated (example: apple,book,phone,pen)
    categories = [c.strip() for c in input(
        "Enter EXACTLY 4 object categories separated by commas (e.g., apple,book,phone,pen): "
    ).split(",") if c.strip()]

    # I enforce exactly four distinct categories (dedupe but keep order)
    categories = list(dict.fromkeys(categories))
    if len(categories) != 4:
        print(f"You entered {len(categories)} unique categories. Please enter exactly 4.")
        return

    # I choose how many images I want per class (default 70)
    try:
        target_per_class = int(input(f"Target images per class [{DEFAULT_TARGET_PER_CLASS}]: ").strip() or str(DEFAULT_TARGET_PER_CLASS))
    except ValueError:
        target_per_class = DEFAULT_TARGET_PER_CLASS

    # I keep my raw dataset in data/raw/<class>/*.jpg
    dataset_dir = Path("data") / "raw"
    create_folder(dataset_dir)
    for c in categories:
        create_folder(dataset_dir / c)

    # I open the camera once (CAP_DSHOW helps on Windows/PyCharm)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("\nPress SPACE to capture an image, ESC to quit.")
    print("Tip: I vary angle, distance, background, lighting; I avoid motion blur.\n")

    try:
        for category in categories:
            cls_dir = dataset_dir / category
            # If I already have images, I resume from the current count
            existing = sorted([p for p in cls_dir.glob("*.jpg")])
            count = len(existing)
            print(f"\n--- Collecting for: {category} (have {count}/{target_per_class}) ---")

            while count < target_per_class:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to grab frame.")
                    break

                # I show myself a focus score to time my SPACE press (not an auto-filter)
                score = focus_score(frame)
                sharp = score >= 80.0  # heuristic for a green hint
                hud = frame.copy()
                cv2.putText(hud, f"class: {category}  saved: {count}/{target_per_class}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(hud, "SPACE=save   ESC=quit   (move/rotate/change lighting)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(hud, f"focus score: {score:.1f}  ({'sharp' if sharp else 'try steadier/closer'})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200 if sharp else 0, 0 if sharp else 255), 2, cv2.LINE_AA)

                # I preview the HUD and wait for my key
                cv2.imshow("Image Collection", hud)
                key = cv2.waitKey(1) & 0xFF  # stable in PyCharm

                if key == 27:  # ESC → I stop everything
                    print("Exiting...")
                    return
                elif key == 32:  # SPACE → I save the current frame
                    img_name = cls_dir / f"{category}_{count:04d}.jpg"
                    cv2.imwrite(str(img_name), frame)
                    print(f"Saved: {img_name}")
                    count += 1

            print(f"Finished collecting for {category}: {count} images")
    finally:
        # I make sure I release the camera and close windows even if I error out
        cap.release()
        cv2.destroyAllWindows()

    # After capture, I remind myself of the assignment checklist
    print("\nImage collection complete!")
    print(f"All images are under: {dataset_dir}")
    print("\nNext (as per assignment):")
    print(f"  1) I manually FILTER bad samples (wrong object, blur, extreme light/noise).")
    print(f"  2) I ensure each class keeps AT LEAST {MIN_REQUIRED_PER_CLASS} images after filtering.")
    print(f"  3) I then perform the 20/5/10 train/val/test split.")

if __name__ == "__main__":
    # I start collection with my chosen settings
    collect_images()

