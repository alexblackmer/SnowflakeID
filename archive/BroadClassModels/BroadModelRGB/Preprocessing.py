#!/usr/bin/env python
"""
This script takes images-label pairs and preprocesses them for tensorflow model development.
It takes each image, converts it to grayscale, and then places it in a label folder.
This is done to prepare data to be ingested into Tensorflow
"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch", "Tim Garrett"]
__version__ = "1.0.1"

import pandas as pd
import shutil

dfi = pd.read_csv('ImageLabelPairs.csv')
sourceDir = 'Images/'
outDir = '../ModelDev/ProcessedImages/'

for ind in dfi.index:
    imagePath = dfi['Image'][ind]
    labelDir = dfi['Label'][ind] + '/'
    shutil.copy(sourceDir + imagePath, outDir + labelDir + imagePath)
    print(ind)