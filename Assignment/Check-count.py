from pathlib import Path

# I set the root of my dataset to data/raw
ROOT = Path("data/raw")

if __name__ == "__main__":
    ok = True  # I keep a flag to check if all classes are okay
    # I loop through each class folder inside data/raw
    for cls in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        # I count how many jpg files are in this class
        n = len(list(cls.glob("*.jpg")))
        # I print the class name, number of images, and whether it’s enough
        print(f"{cls.name:>12}: {n:3d}  -> {'OK' if n >= 35 else 'NEED MORE'}")
        # I update my flag: every class must have at least 35 images
        ok &= (n >= 35)

    # If at least one class had fewer than 35 images, I remind myself to capture more
    if not ok:
        print("\nSome classes < 35. Capture a few more, then re-run this.")

