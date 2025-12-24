import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from django.conf import settings
import os

try:
    import imutils
except ImportError:
    imutils = None
    print("Warning: imutils not installed. Brain cropping may not work optimally.")


def load_trained_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    input_shape = list(model.layers[0].input_shape)
    print('Brain Tumor Model loaded - Input shape:', input_shape)
    return model, input_shape[1:-1]


def crop_brain_contour(img, add_pixels_value=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if imutils:
        cnts = imutils.grab_contours(cnts)
    else:
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    if len(cnts) == 0:
        return img
    
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = add_pixels_value
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, 
                  extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    
    return new_img if new_img.size > 0 else img


def classifier(path, model, shape) -> tuple:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image from path: {path}")
    img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC)
    img_cropped = crop_brain_contour(img)
    img_resized = cv2.resize(img_cropped, dsize=shape, interpolation=cv2.INTER_CUBIC)
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    pred = model.predict(img_batch, verbose=0)
    # For binary classification, convert probability to class index
    prob_tumor = pred[0][0]
    idx = 1 if prob_tumor > 0.5 else 0
    pred_probs = np.array([[1 - prob_tumor, prob_tumor]])
    print(f'Brain Tumor Prediction - No Tumor: {(1-prob_tumor)*100:.2f}%, Tumor: {prob_tumor*100:.2f}%')
    return idx, pred_probs


model_path = os.path.join(settings.BASE_DIR, 'models', 'brain.h5')
if os.path.exists(model_path):
    model, input_layer = load_trained_model(model_path)
    print("Brain Tumor Classification Model (VGG-16) loaded successfully!")
    print(f"Model accuracy: ~80% (Validation & Test)")
else:
    model = None
    input_layer = (224, 224)
    print(f"WARNING: Brain model not found at {model_path}")
    print("Please train the model using brain-tumor-detection-v1-0-cnn-vgg-16.ipynb")
classes = ['No Tumor', 'Tumor']
