#!/usr/bin/env python
"""
This script takes images-label pairs and preprocesses them for tensorflow model development.
It takes each image, converts it to grayscale, and then places it in a label folder.
This is done to prepare data to be ingested into Tensorflow
"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch", "Tim Garrett"]
__version__ = "1.0.1"

import shutil

import pandas as pd
import os

dfi = pd.read_csv('../preprocessing/imageMetadata.csv', usecols=['Image Filename', 'Melt Label'])
sourceDir = '../data/images/'
outDir = './trainingData/'

for ind in dfi.index:
    filename = dfi['Image Filename'][ind]
    label = str(dfi['Melt Label'][ind])
    imageFileExists = os.path.exists(sourceDir + filename) 
    if label != 'N/A' and imageFileExists:
        labelDir = label + '/'
        classFolderExists = os.path.exists(outDir + labelDir)
        if not classFolderExists:
            # Create a new directory because it does not exist
            os.makedirs(outDir + labelDir)
        shutil.copy(sourceDir + filename, outDir + labelDir + filename)
    print(ind)
