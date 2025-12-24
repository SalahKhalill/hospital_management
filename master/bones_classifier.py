import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from django.conf import settings
import os
from PIL import Image


def load_trained_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    input_shape = list(model.layers[0].input_shape)
    print('Bone Fracture Model loaded - Input shape:', input_shape)
    return model, input_shape[1:-1]


def classifier(path, model, shape) -> tuple:
    try:
        img = Image.open(path)
        img = img.resize(shape)
        img = img.convert('RGB')
        img = np.array(img) / 255.0
    except Exception as e:
        raise ValueError(f"Could not read image from path: {path}. Error: {e}")
    
    img_batch = np.expand_dims(img, axis=0)
    
    pred = model.predict(img_batch, verbose=0)
    
    prob_fractured = pred[0][0]
    idx = 1 if prob_fractured > 0.5 else 0
    
    pred_probs = np.array([[1 - prob_fractured, prob_fractured]])
    
    print(f'Bone Fracture Prediction - No Fracture: {(1-prob_fractured)*100:.2f}%, Fractured: {prob_fractured*100:.2f}%')
    
    return idx, pred_probs

model_path = os.path.join(settings.BASE_DIR, 'models', 'xray.h5')
if os.path.exists(model_path):
    model, input_layer = load_trained_model(model_path)
    print("Bone Fracture Detection Model (Custom CNN) loaded successfully!")
    print(f"Model accuracy: ~98.8% (Test)")
else:
    model = None
    input_layer = (128, 128)
    print(f"WARNING: Bone fracture model not found at {model_path}")
    print("Please train the model using f1-0-99-bone-fracture-x-ray-tf-cnn.ipynb")
classes = ['No fracture', 'Fractured']


