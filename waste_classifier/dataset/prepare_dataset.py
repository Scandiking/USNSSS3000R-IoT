###
# Fixes the dataset split and enriches underrepresented classes.
#
# Problems solved:
#   1. glass and metal have 0 images in validation/ and test/ (they were
#      removed in a previous commit). This script moves 15%/15% of the
#      train images for those two classes into validation/ and test/.
#   2. glass (552 train) and metal (1064 train) are underrepresented vs.
#      paper (2069) and plastic (1754).  Archive2 contains 1002 extra glass
#      images and 820 extra metal images that map directly to our categories,
#      so they are copied into train/glass and train/metal.
#
# Run ONCE before training:
#   cd waste_classifier/dataset
#   python prepare_dataset.py
###

import os
import shutil
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE2 = BASE_DIR / "archive2" / "Garbage classification"

SPLIT_VAL  = 0.15
SPLIT_TEST = 0.15

# ---------------------------------------------------------------------------
# 1. Fix missing validation/test for glass and metal
# ---------------------------------------------------------------------------
print("=== Step 1: Fix glass/metal validation & test split ===")

for cls in ["glass", "metal"]:
    train_dir = BASE_DIR / "train" / cls
    val_dir   = BASE_DIR / "validation" / cls
    test_dir  = BASE_DIR / "test" / cls

    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    existing_val  = list(val_dir.glob("*"))
    existing_test = list(test_dir.glob("*"))

    if existing_val or existing_test:
        print(f"  {cls}: validation/test already populated, skipping.")
        continue

    files = [f for f in train_dir.iterdir() if f.is_file()]
    random.shuffle(files)

    n_val  = max(1, int(len(files) * SPLIT_VAL))
    n_test = max(1, int(len(files) * SPLIT_TEST))

    val_files  = files[:n_val]
    test_files = files[n_val:n_val + n_test]

    for f in val_files:
        shutil.move(str(f), val_dir / f.name)
    for f in test_files:
        shutil.move(str(f), test_dir / f.name)

    remaining = len(list(train_dir.glob("*")))
    print(f"  {cls}: moved {len(val_files)} to validation, "
          f"{len(test_files)} to test, {remaining} remain in train")

# ---------------------------------------------------------------------------
# 2. Enrich train/glass and train/metal from archive2
# ---------------------------------------------------------------------------
print("\n=== Step 2: Enrich glass/metal from archive2 ===")

ARCHIVE2_SOURCES = {
    "glass": [
        ARCHIVE2 / "test" / "glass",
        ARCHIVE2 / "val"  / "glass",
    ],
    "metal": [
        ARCHIVE2 / "test" / "metal",
        ARCHIVE2 / "val"  / "metal",
    ],
}

for cls, source_dirs in ARCHIVE2_SOURCES.items():
    dest = BASE_DIR / "train" / cls
    copied = 0
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found, skipping")
            continue
        for f in src_dir.iterdir():
            if not f.is_file():
                continue
            dst_name = f"arc2_{cls}_{f.name}"
            dst_path = dest / dst_name
            if not dst_path.exists():
                shutil.copy2(str(f), dst_path)
                copied += 1
    total = len(list(dest.glob("*")))
    print(f"  {cls}: copied {copied} images from archive2, train total = {total}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Final counts ===")
for split in ["train", "validation", "test"]:
    split_dir = BASE_DIR / split
    if not split_dir.exists():
        continue
    print(f"  {split}:")
    for cls_dir in sorted(split_dir.iterdir()):
        if cls_dir.is_dir():
            n = len(list(cls_dir.glob("*")))
            print(f"    {cls_dir.name}: {n}")

print("\nDone! You can now run train_model.py.")
