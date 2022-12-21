# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: ml_zoomcamp
#     language: python
#     name: ml_zoomcamp
# ---

# ## Required modules

# +
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

from sklearn.model_selection import train_test_split

import os
from tqdm import tqdm
import shutil

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img

import tensorflow.lite as tflite_tf
import tflite_runtime.interpreter as tflite
# -

# ## 1. Data preparation

# The original data found in the Kaggle dataset had redundant files, and data in a format (`.pgm`) that Keras would not be able to use, so I pre-processed them (and stored them in the directory `MIAS-data`) in the following way:           
# + I deleted the files `all-mias.tar` and `all_mias_scans.h5`
# + I converted all `.pgm` files in directory `all-mias` to `.jpg` format. The code used to do this conversion was simply:                                   

# + active=""
# import os 
# from PIL import Image                                                                                                                                                   
# directory = 'MIAS-data/all-mias'                    
# for file in os.listdir(directory):                  
#     f = os.path.join(directory, file)               
#     if file.endswith('.pgm') and os.path.isfile(f):  
#         nfile = file.replace('.pgm','.jpg')     
#         nf = os.path.join(directory, nfile)     
#         img = Image.open(f)                    
#         img.save(nf)                                                                                               
# -

# ## 2. Exploratory Data Analysis

# The file `MIAS-data/all-mias/README` describes the dataset. and the data available for each of the mammogram images. The file `MIAS-data/Info.txt` lists all the available images, together with the relevant information for each of them. Thus, the first lines of `MIAS-data/Info.txt` are: 

# + active=""
# REFNUM BG CLASS SEVERITY X Y RADIUS 
# mdb001 G CIRC B 535 425 197
# mdb002 G CIRC B 522 280 69
# mdb003 D NORM 
# mdb004 D NORM 
# mdb005 F CIRC B 477 133 30
# mdb005 F CIRC B 500 168 26
# mdb006 F NORM 
# mdb007 G NORM 
# mdb008 G NORM 
# mdb009 F NORM 
# -

# The meaning of each field is described in `MIAS-data/all-mias/README`. I reproduce here the relevant part of the file for ease of reference:

# + active=""
# INFORMATION:
#
# This file lists the films in the MIAS database and provides
# appropriate details as follows:
#
# 1st column: MIAS database reference number.
#
# 2nd column: Character of background tissue:
#                 F - Fatty
#                 G - Fatty-glandular
#                 D - Dense-glandular
#
# 3rd column: Class of abnormality present:
#                 CALC - Calcification
#                 CIRC - Well-defined/circumscribed masses
#                 SPIC - Spiculated masses
#                 MISC - Other, ill-defined masses
#                 ARCH - Architectural distortion
#                 ASYM - Asymmetry
#                 NORM - Normal
#
# 4th column: Severity of abnormality;
#                 B - Benign
#                 M - Malignant
#
# 5th,6th columns: x,y image-coordinates of centre of abnormality.
#
# 7th column: Approximate radius (in pixels) of a circle enclosing
#             the abnormality.
#
# NOTES
# =====
# 1) The list is arranged in pairs of films, where each pair
#    represents the left (even filename numbers) and right mammograms
#    (odd filename numbers) of a single patient.
#
# 2) The size of ALL the images is 1024 pixels x 1024 pixels. The images
#    have been centered in the matrix.
#
# 3) When calcifications are present, centre locations and radii
#    apply to clusters rather than individual calcifications.
#    Coordinate system origin is the bottom-left corner.
#
# 4) In some cases calcifications are widely distributed throughout
#    the image rather than concentrated at a single site. In these
#    cases centre locations and radii are inappropriate and have
#    been omitted.
# -

# As an example, we can see the first image in the dataset (resized to 300x300 pixels)

im = Image.open('MIAS-data/all-mias/mdb001.jpg')
im.resize((300,300))

# ### 2.1. Collecting image information in a DataFrame

df = pd.read_csv('MIAS-data/Info.txt',sep=' ')

# We can `describe` the generated DataFrame, and we see that a column 'Unnamed: 7' was added with no real data.

df.describe(include='all')

# We can delete the bogus 'Unnamed: 7' column and inspect the DataFrame data types with the command `df.dtypes`. Everything looks fine: the first four columns are of string type, while the last three are floats. 

del df['Unnamed: 7']

df.dtypes

# ### 2.2. Studying data distribution

# We can first see how many of the mammogram images have an abnormality and what is their distribution. Given the small number of samples, it will be important to try and maintain these proportions later when we divide our dataset in training and validation sets.

print(f'Number of abnormalities: {df.CLASS[df.CLASS != "NORM"].count()}')

# Out of the abnormal mammograms, what is the distribution of benign and malignant samples? As we can see below, the number is very similar for both categories.

# Another feature whose distribution we should also probably try to maintain for the training and validation tests is the radius size of the abnormality. We can also see its distribution below.

f, axs = plt.subplots(1, 3, figsize=(21, 7))
sns.histplot(df.CLASS, ax=axs[0])
sns.histplot(df.SEVERITY, ax=axs[1])
sns.histplot(df.RADIUS, ax=axs[2])

# Let's make number of normal and abnormal similar

df_norm_a, df_norm = train_test_split(df[df.CLASS == 'NORM'], test_size=123./207., random_state=42)
df_abnorm = df[df.CLASS != 'NORM']
df_new = pd.concat([df_norm,df_abnorm])

f, axs = plt.subplots(1, 3, figsize=(21, 7))
sns.histplot(df_new.CLASS, ax=axs[0])
sns.histplot(df_new.SEVERITY, ax=axs[1])
sns.histplot(df_new.RADIUS, ax=axs[2])

# ## 3. Building the detection CNN

# ### 3.1. Splitting the data: training and validation datasets

df_train, df_val = train_test_split(df_new, test_size=0.2, random_state=42)
df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)

# If we plot `CLASS`, `SEVERITY` and `RADIUS` we see that the distribution for both training and validations sets is very similar to that of the full dataset.

# **Training set distribution**

f, axs = plt.subplots(1, 3, figsize=(21, 7))
sns.histplot(df_train.CLASS, ax=axs[0])
sns.histplot(df_train.SEVERITY, ax=axs[1])
sns.histplot(df_train.RADIUS, ax=axs[2])
#f.tight_layout()

# **Validation set distribution**

f, axs = plt.subplots(1, 3, figsize=(21, 7))
sns.histplot(df_val.CLASS, ax=axs[0])
sns.histplot(df_val.SEVERITY, ax=axs[1])
sns.histplot(df_val.RADIUS, ax=axs[2])
#f.tight_layout()

# ### 3.2 Preparing training and validation datasets for Keras

# Since the easiest way to load images for Keras is to have them categorized by directories, I create training and validation directories, with the mammographies simply categorized as `normal` or `abnormal`. For the moment, I only categorize in `normal` or `abnormal`, but for the last course project I would like to consider all abnormality categories, so the following deep network will assume categorical classification, though for this particular case binary classification would have been sufficient.

# Let's create the directories if they don't exist 
for dirn in ['train_data','val_data']:
    if not os.path.exists('MIAS-data/' + dirn):
        os.mkdir('MIAS-data/' + dirn)
    for cat in ['/normal','/abnormal']:
        if not os.path.exists('MIAS-data/' + dirn + cat):
            os.mkdir('MIAS-data/' + dirn + cat)

# +
# Let's move all the train data to normal|abnormal directories
for i in tqdm(range(df_train.shape[0])):
    sourceimg = 'MIAS-data/all-mias/' + df_train.REFNUM[i] + '.jpg'    
    if df_train.CLASS[i] != 'NORM':
        destdir = 'MIAS-data/train_data/abnormal/'
    else:
        destdir = 'MIAS-data/train_data/normal/'
    shutil.copy2(sourceimg, destdir) 
    
# Let's move all the validation data to normal|abnormal directories
for i in tqdm(range(df_val.shape[0])):
    sourceimg = 'MIAS-data/all-mias/' + df_val.REFNUM[i] + '.jpg'
    if df_val.CLASS[i] != 'NORM':
        destdir = 'MIAS-data/val_data/abnormal/'
    else:
        destdir = 'MIAS-data/val_data/normal/'
    shutil.copy2(sourceimg, destdir)     
# -

# ### 3.3 Building and training a basic CNN

# **Batch and image size parameters**

batch_size = 20
img_height = 1024
img_width = 1024

# I define the `train_ds` and the `val_ds`. I use `image_dataset_from_directory` instead of the routines seen during the course, since this one doesn't have such a large I/O bottleneck, and the GPU can be utilized much more efficiently. 

# +
train_ds = tf.keras.utils.image_dataset_from_directory(
  "MIAS-data/train_data",
  label_mode='categorical',
  color_mode='grayscale',  
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "MIAS-data/val_data",
  label_mode='categorical',
  color_mode='grayscale',      
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# -

# We are going to do *Data Augmentation*. For this, be `repeat` the train dataset three times, and then apply a number of transformations to these data, like *rotation*, *translation*, *zoom* and *horizontal flip* (following https://www.tensorflow.org/tutorials/images/data_augmentation)

train_ds = train_ds.repeat(count=3)

# + tags=[]
# %%capture
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.111),    # given as a fraction of 2*Pi  (0.111 * 2 * Pi = 40 degrees)
    layers.RandomTranslation(height_factor=0.2,width_factor=0.2),
    layers.RandomZoom(height_factor=0.2,width_factor=0.2),
    layers.RandomFlip("horizontal")
])

# %%capture
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE);
# -

# We define a basic CNN, where we first normalize the data by dividing the input by 255, with `input_shape` of [1024,1024,1], since the given data is only one channel (grayscale), and then with a number of smaller convolution layers, and finally some dense layers, followed by a `softmax` final layer which will categorize the output in the given two categories. 
#
# For this project, given that the emphasis was on reproducibility and deployment, I don't consider the effect that the number and characteristics of the different layers have on the quality of the prediction and I just design a more or less basic standard network, but for the final course project I will consider different designs and layer parameters in order to improve the accuracy of the predictions.

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1024,1024,1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (4, 4), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),    
    layers.Conv2D(32, (2, 2), activation='relu'),    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),        
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),        
    layers.Dense(2,activation='softmax')
])

# For this project, given that the emphasis was on reproducibility and deployment, I fix the `learning_rate` and `momentum` to some more or less random values, but for the final course project I will consider these as hyperparameters in order to improve the accuracy of the predictions.

model.compile(
  optimizer=SGD(learning_rate=0.01, momentum=0.3),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])

# I want to save the model that produces the best predictions in `model_checkpoints` directory, so we can load and use this model at a later point. 

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model_checkpoints/mias_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# And finally we train the model. As it is, the model seems very stable but with a not so good predicition rate of about 68%. In my local machine, with a Tesla K40c NVidia GPU, each training epoch took about one minute. As mentioned above, for this part of the project we also fix the number of epochs "randomly" to five, but for the final course project I will study how many epochs we should train the network for best performance.

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint]
)

# We can see the summary of the network created, which has a large number of total parameters. For the final project of the course, I will also try to create a less complicated model while trying to at least keep the current accuracy.

model.summary()

# ### 3.4 Training and testing the model locally

# *The scripts mentioned in this section are located and should be run inside the directory `training_testing_locally`*. 
#
# *Remember to run `IPython` inside the `capstone_pr` conda environment, and inside the `capstone-mlzoomcamp` virtual environment (see file `README.md`).*

# #### 3.4.1 Model training 

# Based on the previous analysis, the Python script `train.py` performs all the necessary steps to go from the original data to a saved model in the `.h5` Keras format. For example, if we run it inside IPython:

# + active=""
# In [1]: %run train.py
# ===========================================================
# Collecting image information in a DataFrame
# ===========================================================                                                                                                           
#
# ===========================================================
# Deleting the bogus 'Unnamed: 7' column
# ===========================================================
#
# ===========================================================
# Splitting the data: training and validation datasets
# ===========================================================                                                                                                           
#
# ===========================================================
# Preparing training and validation datasets for Keras
# ===========================================================
#
# ===========================================================
# Building and training a basic CNN
# ===========================================================
#
# Found 259 files belonging to 2 classes.
# Found 66 files belonging to 2 classes.
# ===========================================================
# Performing Data Augmentation
# ===========================================================
#
# [...]
#
# ===========================================================
# Defining basic CNN
# ===========================================================
#
# ===========================================================
# Compiling model
# ===========================================================
#
# ===========================================================
# Training the model
# ===========================================================
#
# [...]
#
# ===========================================================
# Saving model as a train_model.h5
# ===========================================================
# -

# #### 3.4.2 Model testing 

# The script `predict.py` can be used just to verify that the saved model can indeed be used to predict the presence of an abnormality in a mammography.
#
# As it is, it just goes through all the images in the validation set, and for each one it prints the probabilities for 'abnormal' and 'normal' categories, and at the end it prints the total accuracy.

# + active=""
# In [85]: %run predict.py                                               
# Mammographies with abnormalities                            
# ================================                            
# 1/1 [==============================] - 0s 81ms/step                    
# mdb145.jpg -> {'abnormal': 0.4730868, 'normal': 0.5269132} 
#                                                                        
# 1/1 [==============================] - 0s 24ms/step                    
# mdb171.jpg -> {'abnormal': 0.47308245, 'normal': 0.5269175}
#                                                                        
# 1/1 [==============================] - 0s 24ms/step                    
# mdb249.jpg -> {'abnormal': 0.47307074, 'normal': 0.5269293}
#                                      
# [...] 
#
# 1/1 [==============================] - 0s 24ms/step
# mdb303.jpg -> {'abnormal': 0.47307563, 'normal': 0.5269244}
#
# 1/1 [==============================] - 0s 23ms/step
# mdb320.jpg -> {'abnormal': 0.47300902, 'normal': 0.526991}
#
# 1/1 [==============================] - 0s 25ms/step
# mdb180.jpg -> {'abnormal': 0.47306326, 'normal': 0.5269367}
#
# =========================
# Accuracy: 0.6818181818181818
# =========================
# -

# ## 4. Deploying the model

# For the deployment of this model, I used the `AWS Lambda` service, which is very
# interesting since one can imagine a nice workflow with this, in which a
# hospital needs to check mammogram images for abnormalities, and `AWS Lambda` would
# provide a nice platform to offer such a service, without the need for the
# hospital to install a local server to run the model, but at the same time being
# very cost effective, since charges would be incurred only when a prediction on a
# mammogram image needs to be generated. For this, we will follow the workflow
# learnt in Week 9 of the ML-Zoomcamp course.

# ### 4.1. Preparing the deployment locally

# We will first prepare everything locally, since problems can be debugged much
# easier this way.

# *The scripts mentioned in this section are located and should be run inside the directory `deployment`*. 
#
# *Remember to run `IPython` inside the `capstone_pr` conda environment, and inside the `capstone-mlzoomcamp` virtual environment (see file `README.md`).*

# + [markdown] tags=[]
# #### 4.1.1 Convert Keras model to TF-Lite
# -

# Full TensorFlow is too large to use it with the `AWS Lambda` service, so we need to convert it to the `TF-Lite` format, so let's first create the `train_model.tflite` model.

# +
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('deployment/train_model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
# -

# And now let's test that we can use it to make a prediction.

# +
# We will just use one of the images in the 'abnormal' directory

filename = os.listdir('MIAS-data/val_data/abnormal')[0]
f = os.path.join('MIAS-data/val_data/abnormal', filename)
img = load_img(f)

x = np.array(img,dtype=np.float32)
X = np.array([x])

X /= 255.        

classes = [
    'abnormal',
    'normal'
]
# -

# Let's make the prediction, though still using the `TF-Lite` bundled with the full `TensorFlow` (imported as `tflite_tf`)

# +
interpreter = tflite_tf.Interpreter(model_path='deployment/train_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# +
# Let's make the prediction

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

# +
# Let's print the prediction probabilities for the given mammography

res = dict(zip(classes, preds[0]))
print(f'{f} -> {res}\n')

# + [markdown] tags=[]
# #### 4.1.2 Remove the TensorFlow dependency
# -

# Instead of using `TF-Lite` fron `TensorFlow` let's make sure that it works fine with `TFLite_runtime` (imported here as `tflite`), using the same image loaded above.
#

# +
interpreter = tflite.Interpreter(model_path='deployment/train_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# +
# Let's make the prediction

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

# +
# Let's print the prediction probabilities for the given mammography

res = dict(zip(classes, preds[0]))
print(f'{f} -> {res}\n')
# -

# #### 4.1.3 Prepare a Docker image 

# `AWS Lambda` can be used to run simple programs in several languages, but it can also be used to run code that has been packaged inside a full `Docker` container, and that is the route I follow here, since we need to be able to specify a number of libraries to run the model. So the first step is to locally build a `Docker` container with all dependencies and with the required function definitios for `AWS Lambda`. For this I follow the instructions in lesson 9.5 of the ML-Zoomcamp course: https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/09-serverless/05-docker-image.md

# * First step is to create a `Dockerfile`, which can be seen below. In it we install the `Pillow` and `tflite_runtime` libraries, then we include the `train_model.tflite` model generated in section 4.1.1 above, and also the script `lambda_function.py`, which I show in next cell.

# + active=""
# FROM public.ecr.aws/lambda/python:3.9
#
# RUN pip install Pillow 
# RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl
#
# COPY train_model.tflite .
# COPY lambda_function.py .
#
# CMD [ "lambda_function.lambda_handler" ]
# -

# * The `lambda_function.py` script defines the `lambda_handler` entry point, which calls the `predict` function. This will first preprocess the image (downloading the image given a URL, and then normalizing the pixel values). This is then passed to the model saved in the TF-Lite format, for inferring a prediction.

# + active=""
# #!/usr/bin/env python
# # coding: utf-8
#
# import tflite_runtime.interpreter as tflite
# import numpy as np
#
# from io import BytesIO
# from urllib import request
# from PIL import Image
#
# interpreter = tflite.Interpreter(model_path='train_model.tflite')
# interpreter.allocate_tensors()
#
# input_index = interpreter.get_input_details()[0]['index']
# output_index = interpreter.get_output_details()[0]['index']
#
# classes = [
#     'abnormal',
#     'normal'
# ]
#
# def preprocess_image_from_url(url):
#
#     # download from URL
#     with request.urlopen(url) as resp:
#         buffer = resp.read()
#     stream = BytesIO(buffer)
#     img = Image.open(stream)
#
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#
#     x = np.array(img, dtype=np.float32)
#     X = np.array([x])
#     X /= 255.
#     
#     return X
#
# def predict(url): 
#     X = preprocess_image_from_url(url)
#
#     interpreter.set_tensor(input_index, X)
#     interpreter.invoke()
#     preds = interpreter.get_tensor(output_index)
#
#     float_predictions = preds[0].tolist()
#
#     return dict(zip(classes, float_predictions))
#
#
# def lambda_handler(event, context):
#     url = event['url']
#     result = predict(url)
#     return result
# -

# The Docker image is simply built with:
#
# `(capstone-mlzoomcamp) $ docker build -t mammography-model .`

# To test that everything is working fine, we start by creating a container:

# + active=""
# (capstone-mlzoomcamp) $ docker run -it --rm -p 8080:8080 mammography-model:latest
# -

# and then we run the `test_local.py` script, which just runs the prediction for one image stored in `AWS S3` service (see below), against the local service running inside the `docker` container:

# + active=""
# (capstone-mlzoomcamp) $ python test_local.py
# -

# ### 4.2. Cloud deployment

# #### 4.2.1 Uploading Docker image to `ECR` service, creating an `AWS Lambda` function and a `Web service`

# The steps required to create a full cloud deployment are not easy to explain without taking several screenshots or a video of the full process, so I will only mention here the main points of the process. 

# + First, we need to upload the `Docker` image we created in section 4.1.3 to the `ECR` service, which involved the following commands

# + active=""
# (capstone-mlzoomcamp) $ aws ecr create-repository --repository-name mammography-repo
#
# (capstone-mlzoomcamp) $ aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 875176118356.dkr.ecr.eu-west-2.amazonaws.com
# Login Succeeded
#
# (capstone-mlzoomcamp) $ docker tag mammography-model:latest 875176118356.dkr.ecr.eu-west-2.amazonaws.com/mammography-repo:latest
#
# (capstone-mlzoomcamp) $ docker push 875176118356.dkr.ecr.eu-west-2.amazonaws.com/mammography-repo:latest

# + Second, we have to create the Lambda function that uses the previously uploaded container. This step was explained in detail in lesson 9.6 of the ML_Zoomcamp course (https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/09-serverless/06-creating-lambda.md)

# + And lastly we have to create the web service (with the `API Gateway` service) to the above lambda function, explained in detail in lesson 9.7 of the ML_Zoomcamp course (https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/09-serverless/07-api-gateway.md)

# *NOTE*: when creating the Lambda function, it is important to upgrade the default configuration, to have a larger RAM and longer timeout, since the TensorFlow model will require them (at least on the first run of the lambda function).
#
# *NOTE*: by default the web service is open to the public. If we want to restrict it, we could specify a `Resource policy`. For example, to allow access only to a certain range of IPs, we could specify it like:

# + active=""
# {
#     "Version": "2012-10-17",
#     "Statement": [
#         {
#             "Effect": "Allow",
#             "Principal": "*",
#             "Action": "execute-api:Invoke",
#             "Resource": "arn:aws:execute-api:eu-west-2:875176118356:kcri5johz4/*",
#             "Condition": {
#                 "IpAddress": {
#                     "aws:SourceIp": "161.27.206.0/24"
#                 }
#             }
#         }
#     ]
# }
# -

# #### 4.2.2 Testing the service

# To test the cloud services deployed in section 4.2.1, we can run the `test_aws.py` script, which is basically the same as the `test_local.py` script, but running the prediction against the cloud services deployed above instead of the local `docker` container.

# + active=""
# (capstone-mlzoomcamp) $ python test_aws.py
# -

# #### 4.2.2 Adding a user interface and a data storage

# At the beginning of section 4, I explained how `AWS Lambda` could be used in an imagined workflow, in which a hospital could upload mammogram images to some cloud service and then use the trained model deployed to `AWS ECR` for predictions of 'abnormalities`. A graphical description of the solution I developed for this workflow can be seen below:
#
# + The mammogram images for which we want predictions are stored in the service `AWS S3`
# + The User Interface can be created fully in the cloud, with the `Streamlit Cloud` service, which is very easy to set, just by linking the application to a GitHub repository. For this example, the `GitHub` repository I created that is linked to `Streamlit Cloud` is https://github.com/angel-devicente/streamlit-mammography, and the description of the User Interface is just given by the `streamlit_app.py` script (https://github.com/angel-devicente/streamlit-mammography/blob/main/streamlit_app.py). As it is, the User Interface is extremely simple, where we can just select the mammogram image we want to use by using a slider, and then asking for a prediction by pressing a simple button. When this button is pressed, `DNN Prediction` takes over the control.
# + When we press the `Predict` button in the User Interface, the `AWS API Gateway` is used to invoke the `AWS Lambda` function, which in turn uses the `Docker` image stored in the `AWS ECR` service.
#
# **NOTE**: For the moment, the images stored in `AWS S3` are all the images from the "MIAS Mammography" dataset. This is obviously not ideal, since these are the images that were used to train and validate the machine learning model, but as I said previously, the focus for this Capstone project was mainly to work on reproducibility and deployment, so this is not important at this stage. For the final project, I will create training, validation and test datasets, and upload to `AWS S3` only the test dataset.

display.Image("images/cloud_deployment.png")

# #### 4.2.3 Testing the full cloud deployment

# The solution described in section 4.2.2 can be accessed at: 
#
# https://angel-devicente-streamlit-mammography-streamlit-app-jtcdkq.streamlit.app/
#
# A sample screenshot of the user interface can be seen below.

display.Image("images/user_interface.png")

# #### 4.2.3 Cost estimation of the cloud deployment solution

# To calculate a cost estimate for the full solution, let's assume that we need to perform 1000 predictions per week.
#
# From the services described in the above diagram, the ones that would incur some charges are those in `AWS`:
#
# + `AWS S3`: With the current format and resolution, each image is about 40KB. So, if we assume that we need to work with 1000 images, we would need just about 40MB of space (we can assume that at the beginning of each week the old images are removed from `AWS S3`). This is virtually free in `AWS S3` (about `US$` 0.001 / month). *Pricing information can be found at*: https://aws.amazon.com/s3/pricing/?nc=sn&loc=4 
# + `AWS ECR`: The `Docker` image for this project is about 458MB in size, which would incur a cost of about `US$` 0.045 / month. *Pricing information can be found at*: https://aws.amazon.com/ecr/pricing/
# + `AWS Lambda`: Pricing for this service depends on the number of times the Lambda function is called, together with the amount of memory required to run the function and the amount of time it takes to completion. I allocated 1024MB RAM for each function call, and it takes approximately 2 seconds to run, so each function call would cost about `US$` 0.00003, so for 1000 predictions per week (or ~4000 predictions per month), this would be just `US$` 0.12 / month. *Pricing information can be found at*: https://aws.amazon.com/lambda/pricing/
#
# So, **in total** the whole cloud solution running 1000 predictions per week would cost about `US$` 0.17 / month. 

# ## Conclusions

# This work has been prepared for the Capstone Project of the ML_Zoomcamp 2022 course. The goal of the project is to use Machine Learning techniques to help predict abnormalities in mammogram images. Since the work to guarantee easy reproducibility of the project, good prediction accuracy and full cloud deployment was too much for one single project, I focused here on the reproducibility and the deployment parts. The prediction accuracy in this project is very low, which will be tackled in the final project for the ML_Zoomcamp 2022 course.
#
# Regarding reproducibility, I have provided detailed instructions for having a reproducible environment using `conda` and `pipenv` and verified that it indeed can be easily reproduced in a number of different computers (with several `Linux` distributions).
#
# Regarding cloud deployment, using `Streamlit Cloud`, `AWS S3`, `AWS ECR` and `AWS Lambda` I have provided a full cloud solution that could be very cost effective in a real prediction scenario where a few hundred predictions per day need to be performed. 
