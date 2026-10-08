"""Flask backend for the DeepFake Detection web interface."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from src.inference import DEFAULT_THRESHOLD, load_model, model_path, predict_image
from src.preprocessing import FaceNotDetectedError


BASE_DIR = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
app = Flask(__name__, template_folder="templates", static_folder="static")


def _active_model_path() -> Path:
    return model_path().expanduser().resolve()


def _model_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


_loaded_model_path: Path | None = None
_loaded_model_signature: tuple[int, int] | None = None
_loaded_model = None


def get_model():
    """Load the configured model once and reload it when the file changes."""
    global _loaded_model, _loaded_model_path, _loaded_model_signature
    path = _active_model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. Set MODEL_PATH to a .keras model file."
        )
    signature = _model_signature(path)
    if (
        _loaded_model is None
        or path != _loaded_model_path
        or signature != _loaded_model_signature
    ):
        _loaded_model = load_model(path)
        _loaded_model_path = path
        _loaded_model_signature = signature
    return _loaded_model


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
def index():
    path = _active_model_path()
    return render_template(
        "index.html",
        model_name=path.name,
        threshold=DEFAULT_THRESHOLD,
    )


@app.post("/api/predict")
def predict():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Please choose an image to evaluate."), 400

    extension = Path(uploaded.filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify(error="Supported image types are JPG, JPEG, PNG, and WEBP."), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        prediction = predict_image(get_model(), image, DEFAULT_THRESHOLD)
    except FaceNotDetectedError as error:
        return jsonify(error=str(error), evaluated=False), 422
    except (OSError, UnidentifiedImageError) as error:
        return jsonify(error=f"Could not read the image: {error}"), 400
    except (FileNotFoundError, ValueError) as error:
        return jsonify(error=str(error)), 500

    return jsonify(
        evaluated=True,
        label=prediction.label,
        confidence=prediction.confidence,
        fake_probability=prediction.fake_probability,
        face_count=prediction.prepared.face_count,
        detector=prediction.prepared.detector_name,
        model_name=_active_model_path().name,
        processed_image=_image_data_url(prediction.prepared.prepared_image),
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
