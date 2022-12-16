#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

interpreter = tflite.Interpreter(model_path='train_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'abnormal',
    'normal'
]

#### CHANGE THIS!!!
url = 'https://i.ibb.co/ZXs9SJN/mdb182.jpg'

def preprocess_image_from_url(url):

    # download from URL
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    x = np.array(img, dtype=np.float32)
    X = np.array([x])
    X /= 255.
    
    return X

def predict(url): 
    X = preprocess_image_from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

