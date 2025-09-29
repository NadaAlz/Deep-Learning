import random, shutil
from pathlib import Path

def split_dataset(src="data/raw", dst="data/split", train=20, val=5, test=10, seed=42):
    # I fix the random seed so my split is repeatable
    random.seed(seed)
    src, dst = Path(src), Path(dst)

    # I create train/val/test folders inside my split directory
    for s in ["train", "val", "test"]:
        (dst / s).mkdir(parents=True, exist_ok=True)

    # I go through each class folder inside my raw dataset
    for cls_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        # I collect all the jpg files and shuffle them
        imgs = sorted(cls_dir.glob("*.jpg"))
        random.shuffle(imgs)

        # I make sure I have enough images for train+val+test
        need = train + val + test
        if len(imgs) < need:
            print(f"ERROR: {cls} has {len(imgs)} < {need}. Add more and re-run.")
            continue

        # I slice the shuffled list into train, val, and test sets
        splits = {
            "train": imgs[:train],
            "val":   imgs[train:train+val],
            "test":  imgs[train+val:train+val+test],
        }

        # I copy each file into the correct split folder
        for split, files in splits.items():
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out / f.name)

        # I print how many files I copied for this class
        print(f"{cls:>12} → train={len(splits['train']):3d}  val={len(splits['val']):3d}  test={len(splits['test']):3d}")

    # At the end, I remind myself where the new split dataset is stored
    print(f"\nDone. Splits at: {Path(dst).resolve()}")

if __name__ == "__main__":
    # I run the function with default parameters
    split_dataset()

