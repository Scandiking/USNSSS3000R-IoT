###
# This Python script automates the process of looping through each waste category folder and 
# split the files (70/15/15) into the train/validation/test folders while maintaining the 
# category structure.
# This way TensorFlow can directly read from the folder structure, and your labels are 
# folder names.
###

import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_TRAIN = Path("train") # your current train folder with subfolders
OUTPUT_BASE = Path(".") # Where to create train/val/test folders
SPLIT_RATIO = (0.7, 0.15, 0.15) # train, val, test


# Get all categories
categories = [d for d in SOURCE_TRAIN.iterdir() if d.is_dir()]

print("Collecting categories...")
for category in categories:
    cat_name = category.name
    files = [f for f in category.rglob("*") if f.is_file()]
    random.shuffle(files)

    # Calculate split indices
    train_idx = int(len(files) * SPLIT_RATIO[0])
    val_idx = train_idx + int(len(files) * SPLIT_RATIO[1])

    train_files = files[:train_idx]
    val_files = files[train_idx:val_idx]
    test_files = files[val_idx:]

    # Create destination folders
    for split in ["train", "validation", "test"]:
        dest_dir = OUTPUT_BASE / split / cat_name
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Move (cut) files to appropriate folders
    for f in train_files:
        shutil.move(f, OUTPUT_BASE / "train" / cat_name / f.name)
    for f in val_files:
        shutil.move(f, OUTPUT_BASE / "validation" / cat_name / f.name)
    for f in test_files:
        shutil.copy2(f, OUTPUT_BASE / "test" / cat_name / f.name)

    print(f"{cat_name}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

print("Done!")