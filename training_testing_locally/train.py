#!/usr/bin/env python
# coding: utf-8

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers

import pandas as pd
import os
import shutil

from sklearn.model_selection import train_test_split



#############  Functions

def pprint(str):
    print("===========================================================")
    print(str)
    print("===========================================================\n")

#############     



pprint("Collecting image information in a DataFrame")
df = pd.read_csv('../MIAS-data/Info.txt',sep=' ')



pprint("Deleting the bogus 'Unnamed: 7' column")
del df['Unnamed: 7']



pprint("Splitting the data: training and validation datasets")
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)



pprint("Preparing training and validation datasets for Keras")
# let's create the directories if they don't exist 
for dirn in ['train_data','val_data']:
    if not os.path.exists('../MIAS-data/' + dirn):
        os.mkdir('../MIAS-data/' + dirn)
    for cat in ['/normal','/abnormal']:
        if not os.path.exists('../MIAS-data/' + dirn + cat):
            os.mkdir('../MIAS-data/' + dirn + cat)

# let's move all the train data to normal|abnormal directories
for i in range(df_train.shape[0]):
    sourceimg = '../MIAS-data/all-mias/' + df_train.REFNUM[i] + '.jpg'    
    if df_train.CLASS[i] != 'NORM':
        destdir = '../MIAS-data/train_data/abnormal/'
    else:
        destdir = '../MIAS-data/train_data/normal/'
    shutil.copy2(sourceimg, destdir) 
    
# let's move all the validation data to normal|abnormal directories
for i in range(df_val.shape[0]):
    sourceimg = '../MIAS-data/all-mias/' + df_val.REFNUM[i] + '.jpg'
    if df_val.CLASS[i] != 'NORM':
        destdir = '../MIAS-data/val_data/abnormal/'
    else:
        destdir = '../MIAS-data/val_data/normal/'
    shutil.copy2(sourceimg, destdir)     


    
pprint("Building and training a basic CNN")
batch_size = 20
img_height = 256
img_width = 256

train_ds = tf.keras.utils.image_dataset_from_directory(
  "../MIAS-data/train_data",
  label_mode='categorical',
  color_mode='grayscale',  
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "../MIAS-data/val_data",
  label_mode='categorical',
  color_mode='grayscale',      
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



pprint("Performing Data Augmentation")
train_ds = train_ds.repeat(count=3)

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.111),    # given as a fraction of 2*Pi  (0.111 * 2 * Pi = 40 degrees)
    layers.RandomTranslation(height_factor=0.2,width_factor=0.2),
    layers.RandomZoom(height_factor=0.2,width_factor=0.2),
    layers.RandomFlip("horizontal")
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE);



pprint("Defining basic CNN")
model = tf.keras.Sequential([
    tf.keras.Input(shape=(256,256,1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (4, 4), activation='relu'),
    layers.MaxPooling2D(),    
    layers.Conv2D(16, (2, 2), activation='relu'),    
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),        
    layers.Dense(2,activation='softmax')
])



pprint("Compiling model")
model.compile(
  optimizer=SGD(learning_rate=0.01, momentum=0.3),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])



pprint("Training the model")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)



pprint("Saving model as a train_model.h5")
keras.models.save_model(model,"train_model.h5")

