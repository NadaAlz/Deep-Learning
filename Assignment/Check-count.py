from pathlib import Path
ROOT = Path("data/raw")

if __name__ == "__main__":
    ok = True
    for cls in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        n = len(list(cls.glob("*.jpg")))
        print(f"{cls.name:>12}: {n:3d}  -> {'OK' if n>=35 else 'NEED MORE'}")
        ok &= (n>=35)
    if not ok: print("\nSome classes < 35. Capture a few more, then re-run this.")
