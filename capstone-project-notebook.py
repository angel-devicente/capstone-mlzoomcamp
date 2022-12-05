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

# +
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import layers
# -

# ## Images

f = 'MIAS-data/all-mias/mdb125.jpg'
im = Image.open(f)

im.resize((300,300))

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

# PGM (Portable Gray Map) files store grayscale 2D images. Each pixel within the image contains only one or two bytes of information (8 or 16 bits).

batch_size = 20
img_height = 150
img_width = 150

train_ds = tf.keras.utils.image_dataset_from_directory(
  "MIAS-data/all-mias",
    label_mode='binary',
  image_size=(img_height, img_width),
  batch_size=batch_size)


# +

val_ds = tf.keras.utils.image_dataset_from_directory(
  "Homework_Dino_data/test",
    label_mode='binary',
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
