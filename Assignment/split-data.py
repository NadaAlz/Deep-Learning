import random, shutil
from pathlib import Path

def split_dataset(src="data/raw", dst="data/split", train=20, val=5, test=10, seed=42):
    random.seed(seed)
    src, dst = Path(src), Path(dst)
    for s in ["train","val","test"]: (dst / s).mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        imgs = sorted(cls_dir.glob("*.jpg")); random.shuffle(imgs)
        need = train + val + test
        if len(imgs) < need:
            print(f"ERROR: {cls} has {len(imgs)} < {need}. Add more and re-run."); continue
        splits = {
            "train": imgs[:train],
            "val":   imgs[train:train+val],
            "test":  imgs[train+val:train+val+test],
        }
        for split, files in splits.items():
            out = dst / split / cls; out.mkdir(parents=True, exist_ok=True)
            for f in files: shutil.copy2(f, out / f.name)
        print(f"{cls:>12} → train={len(splits['train']):3d}  val={len(splits['val']):3d}  test={len(splits['test']):3d}")
    print(f"\nDone. Splits at: {Path(dst).resolve()}")

if __name__ == "__main__":
    split_dataset()
