import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------
# Always resolve paths relative to this file
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "baseline_skin_cancer_model.h5")
IMAGE_PATH = os.path.join(BASE_DIR, "sample.jpg")

def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    model = load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    threshold = 0.5
    label = "Malignant" if prediction >= threshold else "Benign"

    print(f"Prediction score: {prediction:.4f}")
    print(f"Predicted class: {label}")

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    predict_image(IMAGE_PATH)
