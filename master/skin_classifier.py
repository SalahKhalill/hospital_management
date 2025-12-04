import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from django.conf import settings
import os


def load_trained_model(model_dir):
  model = tf.keras.models.load_model(model_dir)
  input_shape = list(model.layers[0].input_shape)
  print('input_shape',input_shape)
  return model,input_shape[1:-1]#, output_len[1:-1]


def classifier(path,model,shape)-> int:
    img=cv2.imread(path)
    img=cv2.resize(img,dsize=shape)
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img=tf.preprocess_input(img_rgb)
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img,verbose=0)
    idx=np.argmax(pred)
    print('model output: ',pred)

    # plt.imshow(img_rgb) - Removed to prevent blocking in web server
    # values={k:i for i,k in train_batches.class_indices.items()}
    # print(values[idx])
    return idx,pred


# Load model with proper path
model_path = os.path.join(settings.BASE_DIR, 'models', 'skin.h5')
model,input_layer=load_trained_model(model_path)  

# Test code removed - will be called from views when needed
classes=['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis \n ', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
