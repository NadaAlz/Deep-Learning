# collect_images.py
import cv2
import os
from pathlib import Path

MIN_REQUIRED_PER_CLASS = 35       # assignment minimum per class
DEFAULT_TARGET_PER_CLASS = 70     # advised to collect ~2x (then manually filter)

def create_folder(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def focus_score(frame) -> float:
    # simple sharpness hint (variance of Laplacian)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def collect_images():
    # i enter classes once, e.g., apple,book,phone,pen
    categories = [c.strip() for c in input(
        "Enter EXACTLY 4 object categories separated by commas (e.g., apple,book,phone,pen): "
    ).split(",") if c.strip()]

    # enforce exactly 4 distinct categories
    categories = list(dict.fromkeys(categories))  # deduplicate while keeping order
    if len(categories) != 4:
        print(f"You entered {len(categories)} unique categories. Please enter exactly 4.")
        return

    # default target = 70 (≈ 2x 35) so i can delete bad ones later
    try:
        target_per_class = int(input(f"Target images per class [{DEFAULT_TARGET_PER_CLASS}]: ").strip() or str(DEFAULT_TARGET_PER_CLASS))
    except ValueError:
        target_per_class = DEFAULT_TARGET_PER_CLASS

    dataset_dir = Path("data") / "raw"
    create_folder(dataset_dir)
    for c in categories:
        create_folder(dataset_dir / c)

    # open camera once (PyCharm safe)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("\nPress SPACE to capture an image, ESC to quit.")
    print("Tip: vary angle, distance, background, and lighting; avoid motion blur.\n")

    try:
        for category in categories:
            cls_dir = dataset_dir / category
            # resume if i already have some files
            existing = sorted([p for p in cls_dir.glob("*.jpg")])
            count = len(existing)
            print(f"\n--- Collecting for: {category} (have {count}/{target_per_class}) ---")

            while count < target_per_class:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to grab frame.")
                    break

                # focus hint (not a filter; just helps me choose when to press SPACE)
                score = focus_score(frame)
                sharp = score >= 80.0  # heuristic threshold for a green hint
                hud = frame.copy()
                cv2.putText(hud, f"class: {category}  saved: {count}/{target_per_class}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(hud, "SPACE=save   ESC=quit   (move/rotate/change lighting)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(hud, f"focus score: {score:.1f}  ({'sharp' if sharp else 'try steadier/closer'})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200 if sharp else 0, 0 if sharp else 255), 2, cv2.LINE_AA)

                cv2.imshow("Image Collection", hud)
                key = cv2.waitKey(1) & 0xFF  # safer in PyCharm

                if key == 27:  # ESC
                    print("Exiting...")
                    return
                elif key == 32:  # SPACE
                    img_name = cls_dir / f"{category}_{count:04d}.jpg"
                    cv2.imwrite(str(img_name), frame)
                    print(f"Saved: {img_name}")
                    count += 1

            print(f"Finished collecting for {category}: {count} images")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # post-capture checklist to match assignment
    print("\nImage collection complete!")
    print(f"All images are under: {dataset_dir}")
    print(f"\nNext (as per assignment):")
    print(f"  1) Manually FILTER bad samples (wrong object, blur, extreme light/noise).")
    print(f"  2) Ensure each class keeps AT LEAST {MIN_REQUIRED_PER_CLASS} images after filtering.")
    print(f"  3) Then perform the 20/5/10 train/val/test split.")

if __name__ == "__main__":
    collect_images()
