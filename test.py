import cv2
import numpy as np
import joblib
import sys

IMAGE_SIZE = (64, 64)
MODEL_PATH = 'model.pkl'

def predict_image(image_path):
    model = joblib.load(MODEL_PATH)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    features = img.flatten().reshape(1, -1)
    prediction = model.predict(features)
    print(f"Prediction: {prediction[0]} (1=True, 0=False)")

# Example usage
if __name__ == "__main__":
    image_path = sys.argv[1]
    predict_image(image_path)
