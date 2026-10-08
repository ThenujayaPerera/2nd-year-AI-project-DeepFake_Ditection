"""Streamlit user interface for DeepFake Detection."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

try:
    from .inference import DEFAULT_THRESHOLD, load_model, model_path, predict_image
    from .preprocessing import FaceNotDetectedError
except ImportError:  # Streamlit executes this file directly.
    from inference import DEFAULT_THRESHOLD, load_model, model_path, predict_image
    from preprocessing import FaceNotDetectedError


st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🔍",
    layout="centered",
)

st.markdown(
    """
    <style>
    .app-header {
        padding: 1.25rem 1.5rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #172554 0%, #1d4ed8 100%);
        color: white;
        margin-bottom: 1.25rem;
    }
    .app-header h1 {
        margin: 0;
        color: white;
    }
    .app-header p {
        margin: 0.35rem 0 0;
        color: #dbeafe;
    }
    .result-card {
        padding: 1.25rem;
        border-radius: 1rem;
        border: 1px solid #dbeafe;
        background: #f8fafc;
        text-align: center;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .confidence-track {
        height: 0.75rem;
        border-radius: 1rem;
        background: #e2e8f0;
        overflow: hidden;
        margin: 0.75rem 0 0.35rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 1rem;
    }
    .upload-help {
        color: #475569;
        font-size: 0.95rem;
        margin: 0.25rem 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>DeepFake Detection</h1>
        <p>Upload a face image and check whether it appears real or AI-generated.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("E8 deeper 128×128 CNN with BatchNorm · Decision threshold: 0.40")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown(
        '<p class="upload-help">Drag and drop an image into the box below, or click '
        "<b>Browse files</b>. JPG, JPEG, PNG, and WEBP are supported.</p>",
        unsafe_allow_html=True,
    )
with header_right:
    if st.button("Clear", use_container_width=True):
        st.session_state.uploader_key += 1
        st.rerun()

@st.cache_resource
def get_model(model_file: str, file_signature: tuple[int, int]):
    """Load the selected file and reload it when the file changes."""
    del file_signature
    return load_model(Path(model_file))


configured_model_path = model_path().expanduser().resolve()
if configured_model_path.is_file():
    model_signature = (
        configured_model_path.stat().st_mtime_ns,
        configured_model_path.stat().st_size,
    )
else:
    model_signature = (0, 0)
st.caption(f"Active model: `{configured_model_path.name}`")


uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a clear face image for the best result.",
    key=f"uploaded_image_{st.session_state.uploader_key}",
)

if uploaded is None:
    st.info("Choose an image to begin.")
else:
    try:
        image = Image.open(uploaded)
        prediction = predict_image(
            get_model(str(configured_model_path), model_signature),
            image,
            DEFAULT_THRESHOLD,
        )
    except FaceNotDetectedError as error:
        st.warning(str(error))
    except (OSError, ValueError, FileNotFoundError) as error:
        st.error(str(error))
    else:
        if not prediction.prepared.face_detected:
            st.error("Image not clear or can't identify a human.")
            st.stop()

        left, right = st.columns(2)
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            st.image(
                prediction.prepared.prepared_image,
                caption="Model input (cropped and resized to 128×128)",
                use_container_width=True,
            )

        if prediction.label == "FAKE":
            result_color = "#dc2626"
            result_background = "#fef2f2"
        else:
            result_color = "#16a34a"
            result_background = "#f0fdf4"

        st.markdown(
            f"""
            <div class="result-card" style="background:{result_background}">
                <div style="color:{result_color};font-size:0.9rem;font-weight:600">
                    MODEL PREDICTION
                </div>
                <div class="result-label" style="color:{result_color}">
                    {prediction.label}
                </div>
                <div style="font-size:1.1rem;margin-top:0.35rem">
                    Confidence: <b>{prediction.confidence:.2%}</b>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill"
                         style="width:{prediction.confidence:.2%};background:{result_color}">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Fake probability: **{prediction.fake_probability:.2%}**")
        st.caption(
            f"{prediction.prepared.detector_name} selected the clearest "
            f"of {prediction.prepared.face_count} detected face(s)."
        )
