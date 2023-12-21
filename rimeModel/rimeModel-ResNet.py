#!/usr/bin/env python
"""
This script is an attempt to ingest processed data into Tensorflow and then develop a model

"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch", "Tim Garrett"]
__version__ = "1.0.1"

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


data_dir = './trainingData/'
batch_size = 32
img_height = 224
img_width = 224
channels = 1

# Finds class weights
df = pd.read_csv('../preprocessing/imageMetadataCleaned.csv',
                 usecols=['Rime Label'])

# Handles class weights
# Count samples per class
classes_one = df[df['Rime Label'] == 1]
classes_two = df[df['Rime Label'] == 2]
classes_three = df[df['Rime Label'] == 3]
classes_four = df[df['Rime Label'] == 4]
classes_five = df[df['Rime Label'] == 5]
# Convert parts into NumPy arrays for weight computation
one_numpy = classes_one['Rime Label'].to_numpy()
two_numpy = classes_two['Rime Label'].to_numpy()
three_numpy = classes_three['Rime Label'].to_numpy()
four_numpy = classes_four['Rime Label'].to_numpy()
five_numpy = classes_five['Rime Label'].to_numpy()
# Compute class weights
all_together = np.concatenate((one_numpy, two_numpy, three_numpy, four_numpy, five_numpy))
unique_classes = np.unique(all_together)
num_classes = len(unique_classes)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=all_together)
# dict mapping for weights
class_weights = {i : class_weights[i] for i, label in enumerate(sorted(np.unique(all_together)))}

# Creates and preprocesses both training and validation sets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=43,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    crop_to_aspect_ratio=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=43,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    crop_to_aspect_ratio=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Gets test set
val_batches = tf.data.experimental.cardinality(val_ds)
test_dataset = val_ds.take(val_batches // 5)
validation_dataset = val_ds.skip(val_batches // 5)

# Adjust buffer size
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Rescales pixels values from 0-255 to 0-1
rescale = tf.keras.layers.Rescaling(1./255)

# Augments data
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomContrast(factor=0.2, seed=42),
])


# Transfer learning model architecture
base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                                     include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
# Freezes weights of pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# Compiles Model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# Begins Training
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weights)


# Plots model performance over epochs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('ModelPerformance.png')
