"""
evaluate.py
Duty: load the saved model, score it on the test set, show a confusion matrix.
Run:  python src/evaluate.py
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from prepare_data import CLASS_NAMES, MODEL_PATH, MODELS_DIR, load_split


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    test_ds = load_split("test", shuffle=False)   # NO shuffle: keeps order stable

    # 1) Overall score
    results = model.evaluate(test_ds, verbose=0, return_dict=True)
    print(f"Test loss:     {results['loss']:.4f}")

    # 2) Real labels vs predicted labels
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_true = y_true.ravel().astype(int)           # [[0.],[1.]] -> [0, 1]

    probs = model.predict(test_ds, verbose=0).ravel()   # probability of "tomato"
    y_pred = (probs >= 0.5).astype(int)                # 0.5 or above -> tomato (1)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%\n")

    # 3) Precision / recall / F1 per class
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # 4) Confusion matrix picture
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.savefig(MODELS_DIR / "confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()