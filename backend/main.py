import asyncio
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / "CNN" / "models" / "apple_tomato_cnn.keras"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ("apple", "tomato")
model = tf.keras.models.load_model(MODEL_PATH) if MODEL_PATH.exists() else None

# Allow CORS so the frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_bytes()

        await websocket.send_json({
            "status": "received",
            "message": "Image received by server. Classifying image..."
        })

        if model is None:
            raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

        image = tf.keras.utils.load_img(io.BytesIO(data), target_size=IMAGE_SIZE)
        image_array = tf.keras.utils.img_to_array(image)
        prediction = float(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0][0])
        class_index = 1 if prediction >= 0.5 else 0
        confidence = prediction if class_index == 1 else 1 - prediction

        await websocket.send_json({
            "status": "complete",
            "prediction": CLASS_NAMES[class_index],
            "confidence": confidence,
        })
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except:
            pass
