#!/usr/bin/env python
"""
This script defines a snowflake melt classification model using from transfer learning derived from
the ResNet Model trained on ImageNet.
"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer"]

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.applications.resnet50 import ResNet50

dataDir = './trainingData/'
outputDir = './modelOutput/'
batchSize = 128
imgDim = 224
channels = 3
num_classes = 2


# Creates and preprocesses both training and validation sets
train_ds = keras.utils.image_dataset_from_directory(
    dataDir,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode="binary",
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
    label_mode="binary",
    color_mode="rgb",
    shuffle=True,
    crop_to_aspect_ratio=True,
    image_size=(imgDim, imgDim),
    batch_size=batchSize)

# Resizes and rescales data
resize_and_rescale = keras.Sequential([
    keras.layers.Rescaling(1. / 255)])
# Augments data
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip(mode="horizontal", seed=42),
    keras.layers.RandomRotation(factor=0.05, seed=42),
    keras.layers.RandomContrast(factor=0.2, seed=42)])

# Transfer learning model architecture
base_model = ResNet50(weights="imagenet", input_shape=(imgDim, imgDim, channels), include_top=False)
# Applies rescaling and data augmentation
inputs = keras.Input(shape=(imgDim, imgDim, channels))
x = resize_and_rescale(inputs)
x = data_augmentation(x)
outputs = base_model(x)
# Adds top layers to pretrained model
avg = keras.layers.GlobalAveragePooling2D()(outputs)
output = keras.layers.Dense(1, activation='sigmoid')(avg)
model = keras.Model(inputs=inputs, outputs=output)
# Freezes weights of pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# Compiles Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Begins Training
epochs = 40
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

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

plt.suptitle("Melt Prediction Performance Using ResNet", fontsize = 24)
plt.savefig(outputDir + 'Melt-ResNet-Performance.png')
