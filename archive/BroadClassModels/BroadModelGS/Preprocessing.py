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
import cv2

dfi = pd.read_csv('../DataParsing/ImageLabelPairs.csv')
sourceDir = '../../RawImages/'
outDir = './GSImages/'

for ind in dfi.index:
    imagePath = dfi['Image'][ind]
    labelDir = dfi['Label'][ind] + '/'
    im = cv2.imread(sourceDir + imagePath)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(outDir + labelDir + imagePath, imGray)
    print(ind)