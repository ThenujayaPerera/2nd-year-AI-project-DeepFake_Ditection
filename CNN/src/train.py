"""
train.py
Duty: build the CNN, train it, save it to models/, plot the training curves.
Run:  python src/train.py
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from prepare_data import (
    CLASS_NAMES,
    IMG_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    count_images,
    load_split,
)

EPOCHS = 20   # max passes over the training data (early stopping may end sooner)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


def build_model():
    """The CNN. Layers run top to bottom, like an assembly line."""
    model = keras.Sequential(
        [
            keras.Input(shape=IMG_SIZE + (3,)),          # (128, 128, 3) = height, width, RGB

            # --- Data augmentation: random tweaks so the model sees "new" images.
            # These are ONLY active during training, automatically switched off later.
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),

            # --- Scale pixels from 0-255 down to 0-1 (neural nets train better).
            keras.layers.Rescaling(1.0 / 255),

            # --- Convolution blocks: find patterns (edges -> shapes -> "apple-ness").
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),                 # shrink the image by half

            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),

            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.MaxPooling2D(),

            # --- Classifier part: turn the feature maps into one decision.
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(1e-4),
            ),
            keras.layers.Dense(1, activation="sigmoid"), # output 0-1: probability of "tomato"
        ],
        name="apple_tomato_cnn",
    )
    return model


def plot_history(history):
    """Draw accuracy and loss curves, save them as a picture, and show them."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_curves.png")
    plt.close(fig)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_counts = count_images("train")
    total_images = sum(train_counts.values())
    class_weight = {
        index: total_images / (len(CLASS_NAMES) * train_counts[name])
        for index, name in enumerate(CLASS_NAMES)
    }

    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("validation", shuffle=False)

    model = build_model()
    model.summary()                                      # prints layers + parameter counts

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # how the model learns
        loss="binary_crossentropy",                            # how wrong it is (2-class)
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    # Stop if validation loss hasn't improved for 5 epochs,
    # and go back to the best weights seen.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight,
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    plot_history(history)


if __name__ == "__main__":
    main()