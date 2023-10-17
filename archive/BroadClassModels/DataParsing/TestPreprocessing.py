#!/usr/bin/env python
"""
This script is an attempt to ingest processed data into Tensorflow. Currently bunk

"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch", "Tim Garrett"]
__version__ = "1.0.1"

import pandas as pd
import tensorflow as tf
import cv2
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np


dfi = pd.read_csv('ImageLabelPairs.csv')
onehot_encoder = OneHotEncoder(sparse=False)
labels_encoded = onehot_encoder.fit_transform(dfi[['Label']])

images = []
for item in dfi['Image'][0:10]:
    im = cv2.imread('Images/' + item)
    images.append(im)
    print(len(images))


ds = tf.data.Dataset.from_tensors(images)
size = (200,200)
ds = ds.map(lambda images: tf.keras.preprocessing.image.smart_resize(images, size))
images = tf.keras.layers.CenterCrop(height=200, width=200)(images)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)
print(images.shape)

# test out a single convolutional layer to get an idea of what it is doing
tf.random.set_seed(42)  # extra code – ensures reproducibility
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
# these are the feature maps and their associated weights (which will be tuned during training)
fmaps = conv_layer(images)

plt.figure(figsize=(15, 9))
for image_idx in (0, 1):
    for fmap_idx in (0, 1):
        plt.subplot(2, 2, image_idx * 2 + fmap_idx + 1)
        plt.imshow(fmaps[image_idx, :, :, fmap_idx], cmap="gray")
        plt.axis("off")

plt.show()