#!/usr/bin/env python
"""
This script is used to assign labels and temperatures to each image in finals2. Uses flakeManifest2.csv and parsedClassifications.csv
for temp and labels.

"""
__author__ = "Alex Blackmer", "Alex Garrett"
__credits__ = ["Alex Blackmer", "Alex Garrett", "Kyle Fitch"]
__version__ = "1.0.1"

import cv2
import pandas as pd
from pathlib import Path


def isBlurry(image: str):
    ''' calculates laplacian variance of image
        arg "image": str path of image
        returns sharpness value; higher value = sharper image
    '''
    if Path(image).is_file():
        img = cv2.imread(image)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # calculate laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap


path = '../data/finals2'
dfClass = pd.read_csv('./parsedClassifications.csv',
                       usecols=['img1', 'img2', 'img3', 'label'])
dfManifest = pd.read_csv('../data/finals2/flakeManifest2.csv',
                     usecols=['img1', 'img2', 'img3', 'temperature'])

images = []
labels = []
temps = []

# For each unique image in manifest, find each associated classification label
for img in dfManifest.index:
    print(f'Finding label and temp of Image {img}/{dfManifest.shape[0]}')
    labelSet = []

    for cla in dfClass.index:
        if dfManifest['img1'][img] == dfClass['img1'][cla]:
            labelSet.append(dfClass['label'][cla])
        elif dfManifest['img2'][img] == dfClass['img2'][cla]:
            labelSet.append(dfClass['label'][cla])
        elif dfManifest['img3'][img] == dfClass['img3'][cla]:
            labelSet.append(dfClass['label'][cla])

    if len(labelSet) != 0:
        # Adds image 1, 2, and 3 to
        images.append(dfManifest['img1'][img])
        images.append(dfManifest['img2'][img])
        images.append(dfManifest['img3'][img])
        temps.append(dfManifest['temperature'][img])
        temps.append(dfManifest['temperature'][img])
        temps.append(dfManifest['temperature'][img])
        labels.append(labelSet)
        labels.append(labelSet)
        labels.append(labelSet)


# Use for output dataframe  when ready
dfo = pd.DataFrame(data={'image': images,
                         'label': labels,
                         'temp': temps})

# This section computes the blurriness of each image
blurList = []
for ind in dfo.index:
    print(f'Finding blurriness of Image {ind}/{dfo.shape[0]}')
    blur = isBlurry(path+dfo['image'][ind])
    blurList.append(blur)
dfo['blur'] = blurList

# Output file
dfo.to_csv('imageMetadata.csv')
