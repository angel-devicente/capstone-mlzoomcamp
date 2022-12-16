import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img

model = keras.models.load_model('train_model.h5')
#img = Image.open('../MIAS-data/all-mias/mdb001.jpg')

classes = [
    'abnormal',
    'normal'
]

correct = 0
total = 0

print("Mammographies with abnormalities")
print("================================")
for filename in os.listdir('../MIAS-data/val_data/abnormal'):
    f = os.path.join('../MIAS-data/val_data/abnormal', filename)
    if os.path.isfile(f):
        img = load_img(f)

        x = np.array(img,dtype=np.float32)
        X = np.array([x])

        X /= 255.

        preds = model.predict(X)
        res = dict(zip(classes, preds[0]))
        print(f'{filename} -> {res}\n')

        total += 1
        if res['abnormal'] >= 0.5:
            correct += 1

        
print("Mammographies without abnormalities")
print("===================================")
for filename in os.listdir('../MIAS-data/val_data/normal'):
    f = os.path.join('../MIAS-data/val_data/normal', filename)
    if os.path.isfile(f):
        img = load_img(f)

        x = np.array(img,dtype=np.float32)
        X = np.array([x])

        X /= 255.

        preds = model.predict(X)
        res = dict(zip(classes, preds[0]))
        print(f'{filename} -> {res}\n')
        
        total += 1
        if res['normal'] >= 0.5:
            correct += 1


print("=========================")            
print(f'Accuracy: {correct/total}')            
print("=========================")
