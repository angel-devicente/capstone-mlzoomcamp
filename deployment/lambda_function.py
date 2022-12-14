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

def preprocess_image_from_url(url):

    # download from URL
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    im = Image.open(stream)
    img = im.resize((256,256))
    
    x = np.array(img, dtype=np.float32)
    X = np.array([x])
    X = np.expand_dims(X, axis=3)
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

