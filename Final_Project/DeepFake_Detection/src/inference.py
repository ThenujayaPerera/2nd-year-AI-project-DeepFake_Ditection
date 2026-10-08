"""Model loading and single-image prediction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from PIL import Image

try:
    from .preprocessing import PreparedImage, prepare_image
except ImportError:  # Streamlit executes app.py as a script.
    from preprocessing import PreparedImage, prepare_image


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "mediapipe_e08_best.keras"
DEFAULT_THRESHOLD = 0.40


@dataclass(frozen=True)
class Prediction:
    """A user-facing prediction result."""

    label: str
    confidence: float
    fake_probability: float
    prepared: PreparedImage


def model_path() -> Path:
    """Resolve the model path from MODEL_PATH or the checked-in project layout."""
    configured = os.getenv("MODEL_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_PATH


def load_model(model_file: Path | None = None) -> tf.keras.Model:
    """Load the trained Keras model and fail clearly if it is unavailable."""
    path = model_file or model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. Set MODEL_PATH to a .keras model file."
        )
    return tf.keras.models.load_model(path, compile=False)


def predict_image(
    model: tf.keras.Model,
    image: Image.Image,
    threshold: float = DEFAULT_THRESHOLD,
) -> Prediction:
    """Prepare an image and classify it as real or fake."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    prepared = prepare_image(image)
    output = model.predict(prepared.model_input, verbose=0)
    # Training encoded Fake as class 0 and Real as class 1.
    real_probability = float(output[0][0])
    fake_probability = 1.0 - real_probability
    is_fake = fake_probability >= threshold
    confidence = fake_probability if is_fake else 1.0 - fake_probability
    return Prediction(
        label="FAKE" if is_fake else "REAL",
        confidence=confidence,
        fake_probability=fake_probability,
        prepared=prepared,
    )
