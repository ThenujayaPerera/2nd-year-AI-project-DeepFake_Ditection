from pathlib import Path
from PIL import Image
import os
import sys

# Ensure src is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference import load_model, predict_image

def test():
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully!")
    
    # Check a Real image
    real_path = Path("sample_Dataset/Train/Real/real_0.jpg")
    if real_path.exists():
        print(f"Testing Real image: {real_path}")
        img = Image.open(real_path)
        pred = predict_image(model, img)
        print(f"Result -> Label: {pred.label}, Confidence: {pred.confidence:.2%}, Fake Prob: {pred.fake_probability:.2%}")
    
    # Check a Fake image
    fake_path = Path("sample_Dataset/Train/Fake/fake_0.jpg")
    if fake_path.exists():
        print(f"Testing Fake image: {fake_path}")
        img = Image.open(fake_path)
        pred = predict_image(model, img)
        print(f"Result -> Label: {pred.label}, Confidence: {pred.confidence:.2%}, Fake Prob: {pred.fake_probability:.2%}")

if __name__ == "__main__":
    test()
