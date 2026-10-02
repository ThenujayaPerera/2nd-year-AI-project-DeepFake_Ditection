"""
predict.py
Duty: classify one or more images using the saved model.
Run:  python src/predict.py test_images/my_apple.jpg
      python src/predict.py                      (predicts everything in test_images/)
      python src/predict.py my.jpg --show        (also displays the image)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from prepare_data import BASE_DIR, CLASS_NAMES, IMAGE_EXTENSIONS, IMG_SIZE, MODEL_PATH

TEST_IMAGES_DIR = BASE_DIR / "test_images"


def predict_image(model, image_path):
    """Return (class_name, confidence) for a single image file."""
    image_path = str(image_path)
    if not tf.io.gfile.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.lower().endswith(tuple(IMAGE_EXTENSIONS)):
        raise ValueError(f"Unsupported image format: {image_path}")

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)  # open + resize
    arr = tf.keras.utils.img_to_array(img)        # to numpy: (128, 128, 3), values 0-255
    arr = np.expand_dims(arr, axis=0)             # add batch dimension -> (1, 128, 128, 3)

    prob_tomato = float(model.predict(arr, verbose=0)[0][0])   # number between 0 and 1

    if prob_tomato >= 0.5:
        return CLASS_NAMES[1], prob_tomato        # "tomato"
    return CLASS_NAMES[0], 1 - prob_tomato        # "apple"


def main():
    parser = argparse.ArgumentParser(description="Classify apple vs tomato images.")
    parser.add_argument("images", nargs="*", help="image path(s); default = test_images/")
    parser.add_argument("--show", action="store_true", help="display each image")
    args = parser.parse_args()

    if args.images:
        paths = args.images
    else:
        if not TEST_IMAGES_DIR.exists():
            print(f"Directory not found: {TEST_IMAGES_DIR}")
            return
        paths = sorted(
            str(p) for p in TEST_IMAGES_DIR.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not paths:
        print("No images found. Put some in test_images/ or pass a path.")
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    for path in paths:
        label, confidence = predict_image(model, path)
        print(f"{path}  ->  {label}  ({confidence * 100:.1f}% sure)")

        if args.show:
            plt.imshow(plt.imread(path))
            plt.title(f"{label} ({confidence * 100:.1f}%)")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()