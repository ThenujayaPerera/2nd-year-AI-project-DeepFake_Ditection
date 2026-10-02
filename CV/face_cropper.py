"""Detect faces and prepare cropped images for a deepfake model.

Examples:
    python CV/face_cropper.py input.jpg --output CV/crops
    python CV/face_cropper.py Downloads --output CV/crops --size 224
"""

import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def crop_largest_face(
    image,
    detector,
    padding=0.25,
    output_size=224,
):
    """Return a square face crop and its bounding box, or (None, None)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None, None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    center_x = x + width / 2
    center_y = y + height / 2
    side = max(width, height) * (1 + 2 * padding)

    left = max(0, int(center_x - side / 2))
    top = max(0, int(center_y - side / 2))
    right = min(image.shape[1], int(center_x + side / 2))
    bottom = min(image.shape[0], int(center_y + side / 2))

    crop = image[top:bottom, left:right]
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop, (left, top, right - left, bottom - top)


def process_image(image_path, output_dir, detector, padding, output_size):
    image = cv2.imread(str(image_path))
    if image is None:
        return "unreadable"

    crop, _ = crop_largest_face(image, detector, padding, output_size)
    if crop is None:
        return "no face"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_face.jpg"
    if not cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        return "could not save"
    return str(output_path)


def input_images(input_path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Crop the largest face from images.")
    parser.add_argument("input", type=Path, help="An image file or folder of images")
    parser.add_argument("--output", type=Path, default=Path("CV/crops"))
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    if args.padding < 0 or args.padding > 2:
        parser.error("--padding must be between 0 and 2")
    if args.size < 32:
        parser.error("--size must be at least 32")

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {CASCADE_PATH}")

    images = input_images(args.input)
    if not images:
        print("No supported images found.")
        return

    for image_path in images:
        result = process_image(
            image_path, args.output, detector, args.padding, args.size
        )
        print(f"{image_path} -> {result}")


if __name__ == "__main__":
    main()