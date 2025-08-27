# collect_dataset.py
# build an image dataset via your laptop camera for object detection/classification

import cv2
import os
import time
from datetime import datetime
from pathlib import Path

def try_imshow(win_name, frame):
    """Attempt to show a frame; return True if GUI works, False if not."""
    try:
        cv2.imshow(win_name, frame)
        # minimal wait to process events
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def ask_user_inputs():
    print("\n=== Image Dataset Collector ===")
    root = input("Dataset root folder (default: dataset): ").strip() or "dataset"
    cats_raw = input("Enter categories (comma separated, e.g., 'bottle,phone,mouse'): ").strip()
    while not cats_raw:
        cats_raw = input("Please enter at least one category: ").strip()
    categories = [c.strip().replace(" ", "_") for c in cats_raw.split(",") if c.strip()]

    per_class_str = input("How many images per category? (default: 50): ").strip()
    try:
        per_class = int(per_class_str) if per_class_str else 50
    except:
        per_class = 50

    delay_str = input("Auto-capture delay in seconds (default: 0.2) [set 0 to only capture on keypress]: ").strip()
    try:
        auto_delay = float(delay_str) if delay_str else 0.2
    except:
        auto_delay = 0.2

    return root, categories, per_class, auto_delay

def ensure_dirs(root, categories):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for c in categories:
        (root_path / c).mkdir(parents=True, exist_ok=True)
    return root_path

def next_filename(folder: Path, prefix="img", ext=".jpg"):
    # creates incremental filenames: img_0001.jpg, img_0002.jpg, ...
    i = 1
    while True:
        name = f"{prefix}_{i:04d}{ext}"
        candidate = folder / name
        if not candidate.exists():
            return candidate
        i += 1

def draw_overlay(frame, cat, count, goal, auto_mode, auto_delay):
    h, w = frame.shape[:2]
    # a simple central guide box (helps center the object)
    box_w, box_h = int(w*0.5), int(h*0.5)
    x1 = (w - box_w)//2
    y1 = (h - box_h)//2
    x2 = x1 + box_w
    y2 = y1 + box_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

    lines = [
        f"Category: {cat}  ({count}/{goal})",
        f"Auto-capture: {'ON' if auto_mode else 'OFF'}  delay={auto_delay:.2f}s",
        "keys: [c]=capture  [a]=toggle auto  [n]=next class  [d]=delete last  [q]=quit",
        "tip: keep object inside the green box; vary angle, distance, lighting"
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        y += 26

def main():
    root, categories, per_class, auto_delay = ask_user_inputs()
    root_path = ensure_dirs(root, categories)

    print("\nControls:")
    print("  c = capture one image")
    print("  a = toggle auto-capture")
    print("  n = next category")
    print("  d = delete the most recent image (for current category)")
    print("  q = quit")
    print("\nStart capturing when the window opens.")

    cap = cv2.VideoCapture(0)  # try 0, or 1 if you have multiple cameras
    if not cap.isOpened():
        print("ERROR: Could not access the camera. On macOS, allow camera for Terminal.")
        return

    auto_mode = auto_delay > 0
    last_capture_time = 0.0

    for cat in categories:
        folder = root_path / cat
        captured = len(list(folder.glob("*.jpg")))
        last_saved_path = None

        print(f"\n--- Category: {cat} ---")
        print(f"Target: {per_class} images. Already present: {captured}.")
        print("Press 'n' to skip to next category if you want.")

        while captured < per_class:
            ok, frame = cap.read()
            if not ok:
                print("WARN: Failed to read from camera.")
                time.sleep(0.2)
                continue

            # overlay instructions
            draw_overlay(frame, cat, captured, per_class, auto_mode, auto_delay)
            cv2.imshow("Dataset Collector", frame)

            # auto-capture logic
            now = time.time()
            should_auto_capture = auto_mode and (now - last_capture_time >= auto_delay) and cv2.getWindowProperty("Dataset Collector", 0) >= 0

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == ord('a'):
                auto_mode = not auto_mode
                print(f"Auto-capture: {'ON' if auto_mode else 'OFF'}")

            if key == ord('n'):
                print("Moving to next category.")
                break  # next category

            # manual capture
            if key == ord('c'):
                filepath = next_filename(folder)
                cv2.imwrite(str(filepath), frame)
                captured += 1
                last_saved_path = filepath
                last_capture_time = now
                print(f"Saved: {filepath}")

            # delete last
            if key == ord('d'):
                if last_saved_path and last_saved_path.exists():
                    try:
                        os.remove(last_saved_path)
                        captured = max(0, captured - 1)
                        print(f"Deleted last image: {last_saved_path}")
                        last_saved_path = None
                    except Exception as e:
                        print(f"Could not delete: {e}")
                else:
                    print("No recent image to delete.")

            # auto-capture tick
            if should_auto_capture:
                filepath = next_filename(folder)
                cv2.imwrite(str(filepath), frame)
                captured += 1
                last_saved_path = filepath
                last_capture_time = now
                # print a lightweight log to keep the UI smooth
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {cat}: {captured}/{per_class}")

        # done with this category
        print(f"Done: {cat} ({captured} images).")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Finished. Dataset saved under: {root_path.resolve()}")
    print("Tip: remove bad images, then split into train/val/test before training.")

if __name__ == "__main__":
    main()

