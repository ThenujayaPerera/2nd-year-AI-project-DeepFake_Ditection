"""
prepare_data.py
Duty: shared settings + loading images from dataset/ folders.
Other files import from here:  from prepare_data import IMG_SIZE, load_split
"""
from pathlib import Path

import tensorflow as tf
from PIL import Image

# ---------- Shared settings ----------
# Path(__file__) = this file. .parent = src/. .parent again = project root.
# Building paths this way means the code works no matter where you run it from.
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "apple_tomato_cnn.keras"

IMG_SIZE = (128, 128)               # every image is resized to 128 x 128
BATCH_SIZE = 32                     # the model sees 32 images at a time
CLASS_NAMES = ["apple", "tomato"]   # apple -> label 0, tomato -> label 1
SEED = 42                           # makes shuffling repeatable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------- Loading ----------
def load_split(split_name, shuffle):
    """
    Load one split ("train", "validation" or "test") as a tf.data.Dataset.
    Each item is a batch: (images, labels)
        images shape: (32, 128, 128, 3)  -> pixel values 0-255
        labels shape: (32, 1)            -> 0.0 = apple, 1.0 = tomato
    """
    split_dir = DATASET_DIR / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Folder not found: {split_dir}")
    validate_split(split_name)

    ds = tf.keras.utils.image_dataset_from_directory(
        str(split_dir),
        labels="inferred",        # label = name of the sub-folder
        label_mode="binary",      # 2 classes -> a single 0/1 label
        class_names=CLASS_NAMES,  # forces the order: apple=0, tomato=1
        image_size=IMG_SIZE,      # resize every image
        batch_size=BATCH_SIZE,
        shuffle=shuffle,          # True for training, False for validation/test
        seed=SEED,
    )
    # prefetch = prepare the next batch while the model works on the current one
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------- Small helper checks ----------
def count_images(split_name):
    """Count images per class in one split."""
    counts = {}
    for cls in CLASS_NAMES:
        folder = DATASET_DIR / split_name / cls
        if folder.exists():
            counts[cls] = sum(
                1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            counts[cls] = 0
    return counts


def validate_split(split_name):
    """Check that every class folder exists and contains supported images."""
    split_dir = DATASET_DIR / split_name
    counts = count_images(split_name)

    missing = [cls for cls in CLASS_NAMES if not (split_dir / cls).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folder(s) in {split_dir}: {', '.join(missing)}"
        )

    empty = [cls for cls, count in counts.items() if count == 0]
    if empty:
        raise ValueError(
            f"No supported images found for {', '.join(empty)} in {split_dir}"
        )


def find_bad_images():
    """Return a list of image files that Pillow cannot open (corrupt files)."""
    bad = []
    for path in DATASET_DIR.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(path) as img:
                    img.verify()          # checks the file without fully decoding it
            except Exception:
                bad.append(path)
    return bad


# ---------- Run this file directly to test it ----------
if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        print(f"{split:<11}", count_images(split))

    bad_files = find_bad_images()
    print("Corrupt images:", bad_files if bad_files else "none")

    train_ds = load_split("train", shuffle=True)
    images, labels = next(iter(train_ds))         # grab ONE batch
    print("images:", images.shape, images.dtype)
    print("labels:", labels.shape)
    print("pixel range:", float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))