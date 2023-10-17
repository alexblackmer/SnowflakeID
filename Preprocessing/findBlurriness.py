#!/usr/bin/env python
"""
This script is uses laplacian variance to find the blurriness of an image

"""
__author__ = "Alex Blackmer, Alex Garrett"
__credits__ = ["Alex Blackmer", "Kyle Fitch"]
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


path = '../RawImages/'

dfi = pd.read_csv('./imageLabelTempPairs.csv',
                       usecols=['image', 'label', 'temp'])

blurList = []

for ind in dfi.index:
    print(f'Image {ind} of {dfi.shape[0]}')
    blur = isBlurry(path+dfi['image'][ind])
    blurList.append(blur)

dfi['blur'] = blurList


# Output file
dfi.to_csv('imageLabelTempBlur.csv')