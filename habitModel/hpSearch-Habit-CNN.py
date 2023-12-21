#!/usr/bin/env python
"""
This script searches for the best hyperparameters for a 3-layer Convolutional Neural Network.
"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer"]

from tensorflow import keras
import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

data_dir = './trainingData/'
batch_size = 32
img_dim = 224
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
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    crop_to_aspect_ratio=True,
    image_size=(img_dim, img_dim),
    batch_size=batch_size)
val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    crop_to_aspect_ratio=True,
    image_size=(img_dim, img_dim),
    batch_size=batch_size)


# Resizes and rescales data
resize_and_rescale = keras.Sequential([
    keras.layers.Rescaling(1. / 255)])
# Augments data
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip(mode="horizontal", seed=42),
    keras.layers.RandomRotation(factor=0.05, seed=42),
    keras.layers.RandomContrast(factor=0.2, seed=42)])


def model_builder(hp):
    # create model object
    model = keras.Sequential([
        # Preprocesses training and val data
        resize_and_rescale,
        data_augmentation,
        # Adds first convolutional layer
        keras.layers.Conv2D(
            # Adds filter
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
            # Adds filter size or kernel size
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5, 7]),
            # activation function
            activation='relu',
            input_shape=(img_dim, img_dim, channels)),
        # Adds second convolutional layer
        keras.layers.Conv2D(
            # Adds filter
            filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
            # Adds filter size or kernel size
            kernel_size=hp.Choice('conv_2_kernel', values=[3, 5, 7]),
            # activation function
            activation='relu'),
        # Adds flatten layer
        keras.layers.Flatten(),
        # Adds dense layer
        keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
            activation='relu'),
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    # Compiles model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Defines tuning parameters
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hyperModels',
                     project_name='habit-CNN-weighted')
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Executes search
tuner.search(train_ds, epochs=50, validation_data=val_ds, callbacks=[stop_early], class_weight=class_weights)

# Get the best model
model = tuner.get_best_models(num_models=1)[0]
# Summary of best model
model.summary()

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal parameters are Learning rate: {best_hps.get('learning_rate')}, 
first conv layer filter:{best_hps.get('conv_1_filter')} and kernel size:{best_hps.get('conv_1_kernel')},
second conv layer filter:{best_hps.get('conv_2_filter')} and kernel size:{best_hps.get('conv_2_kernel')},
and number of dense units{best_hps.get('dense_1_units')}: 
""")
