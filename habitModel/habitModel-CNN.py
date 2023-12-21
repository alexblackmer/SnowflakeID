#!/usr/bin/env python
"""
This script defines a snowflake habit classification model using a CNN.
The hyperparameters were found via hpSearch-Habit-CNN.py.
"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer"]

import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

dataDir = './trainingData/'
outputDir = './modelOutput/'
batchSize = 32
imgDim = 224
channels = 3

# Finds class weights
df = pd.read_csv('../preprocessing/imageMetadata.csv',
                 usecols=['Habit Label'])

# Handles class weights
# Count samples per class
classes_one = df[df['Habit Label'] == 'AGG']
classes_two = df[df['Habit Label'] == 'CC']
classes_three = df[df['Habit Label'] == 'GR']
classes_four = df[df['Habit Label'] == 'PC']
# Convert parts into NumPy arrays for weight computation
one_numpy = classes_one['Habit Label'].to_numpy()
two_numpy = classes_two['Habit Label'].to_numpy()
three_numpy = classes_three['Habit Label'].to_numpy()
four_numpy = classes_four['Habit Label'].to_numpy()
# Compute class weights
all_together = np.concatenate((one_numpy, two_numpy, three_numpy, four_numpy))
unique_classes = np.unique(all_together)
num_classes = len(unique_classes)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=all_together)
# dict mapping for weights
class_weights = {i: class_weights[i] for i, label in enumerate(sorted(np.unique(all_together)))}

# Creates and preprocesses both training and validation sets
train_ds = keras.utils.image_dataset_from_directory(
    dataDir,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    crop_to_aspect_ratio=True,
    image_size=(imgDim, imgDim),
    batch_size=batchSize)
val_ds = keras.utils.image_dataset_from_directory(
    dataDir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    crop_to_aspect_ratio=True,
    image_size=(imgDim, imgDim),
    batch_size=batchSize)

# Resizes and rescales dat
resize_and_rescale = keras.Sequential([
    keras.layers.Rescaling(1. / 255)])
# Augments data
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip(mode="horizontal", seed=42),
    keras.layers.RandomRotation(factor=0.05, seed=42),
    keras.layers.RandomContrast(factor=0.2, seed=42)])

# create model object
model = keras.Sequential([
    # Preprocesses training and val data
    resize_and_rescale,
    data_augmentation,
    # Adds first convolutional layer
    keras.layers.Conv2D(
        # Adds filter
        filters=64,
        # Adds filter size or kernel size
        kernel_size=7,
        # activation function
        activation='relu',
        input_shape=(imgDim, imgDim, channels)),
    # Adds second convolutional layer
    keras.layers.Conv2D(
        # Adds filter
        filters=64,
        # Adds filter size or kernel size
        kernel_size=7,
        # activation function
        activation='relu'),
    # Adds flatten layer
    keras.layers.Flatten(),
    # Adds dense layer
    keras.layers.Dense(
        units=32,
        activation='relu'),
    # Output layer
    keras.layers.Dense(num_classes, activation='softmax')
])
# Compiles model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Begins Training
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weights)

# Plots model performance over epochs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
# Plots Accuracy
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
# Plots Loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.suptitle("Habit Prediction Performance Using CNN", fontsize = 24)
plt.savefig(outputDir + 'Habit-CNN-Performance.png')
