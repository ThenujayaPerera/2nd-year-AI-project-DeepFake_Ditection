"""Image preparation used by the inference application."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreparedImage:
    """The model input and the image shown to the user."""

    model_input: np.ndarray
    prepared_image: Image.Image
    face_detected: bool
    face_count: int
    detector_name: str


class FaceNotDetectedError(ValueError):
    """Raised when an image does not contain a detectable human face."""


def _detect_faces(image: Image.Image) -> list[tuple[int, int, int, int, float]]:
    """Detect faces with MediaPipe and return pixel boxes plus confidence."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    detections = []
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    ) as detector:
        result = detector.process(rgb)
    for detection in result.detections or []:
        box = detection.location_data.relative_bounding_box
        left = max(0, int(box.xmin * width))
        top = max(0, int(box.ymin * height))
        right = min(width, int((box.xmin + box.width) * width))
        bottom = min(height, int((box.ymin + box.height) * height))
        if right > left and bottom > top:
            confidence = float(detection.score[0]) if detection.score else 0.0
            detections.append((left, top, right - left, bottom - top, confidence))
    return detections


def _clearest_face(
    image: Image.Image,
    faces: list[tuple[int, int, int, int, float]],
) -> tuple[int, int, int, int, float]:
    """Choose the clearest useful face, not simply the first or largest face."""
    pixels = np.asarray(image.convert("RGB"))
    image_area = pixels.shape[0] * pixels.shape[1]
    scored = []
    for face in faces:
        x, y, width, height, detector_confidence = face
        crop = pixels[y : y + height, x : x + width]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(sharpness / 500.0, 1.0)
        size_score = min((width * height) / (image_area * 0.35), 1.0)
        clarity_score = (
            0.50 * sharpness_score
            + 0.30 * size_score
            + 0.20 * detector_confidence
        )
        scored.append((clarity_score, face))
    return max(scored, key=lambda item: item[0])[1]


def prepare_image(image: Image.Image, image_size: tuple[int, int] = (128, 128)) -> PreparedImage:
    """Detect/crop a face, resize it, and return a float32 model batch."""
    rgb = image.convert("RGB")
    faces = _detect_faces(rgb)
    if not faces:
        raise FaceNotDetectedError(
            "No human face was detected. Upload an image containing a clear human face; "
            "the image was not evaluated by the model."
        )
    if len(faces) == 1 and abs((rgb.width / rgb.height) - (image_size[0] / image_size[1])) < 0.01:
        cropped = rgb
        face_detected = True
        detector_name = "MediaPipe Face Detection (raw image used)"
    else:
        x, y, width, height, _ = _clearest_face(rgb, faces)
        padding = int(max(width, height) * 0.20)
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(rgb.width, x + width + padding)
        bottom = min(rgb.height, y + height + padding)
        cropped = rgb.crop((left, top, right, bottom))
        face_detected = True
        detector_name = "MediaPipe Face Detection"

    resized = cropped.resize(image_size, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    return PreparedImage(
        model_input=np.expand_dims(array, axis=0),
        prepared_image=resized,
        face_detected=face_detected,
        face_count=len(faces),
        detector_name=detector_name,
    )
