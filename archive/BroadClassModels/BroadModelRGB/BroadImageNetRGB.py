#!/usr/bin/env python
"""
This script is an attempt to ingest processed data into Tensorflow and then develop a model

"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch", "Tim Garrett"]
__version__ = "1.0.1"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = 'RGBImages/'
batch_size = 32
img_height = 224
img_width = 224

# Creates testiing dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=43,
    image_size=(img_height, img_width))

test_ds.shape

# Model structure
model = tf.keras.applications.ResNet50(weights="imagenet")
images_resized = tf.keras.layers.Resizing(height=224, width=224,
                                          crop_to_aspect_ratio=True)(test_ds)
inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)
Y_proba = model.predict(inputs)
top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)

for image_index in range(len(test_ds)):
    print(f"Image #{image_index}")
for class_id, name, y_proba in top_K[image_index]:
    print(f" {class_id} - {name:12s} {y_proba:.2%}")
