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

path = '../data/finals2/'
dfManifest = pd.read_csv(path + 'flakeManifest2.csv',
                         usecols=['img1', 'img2', 'img3'])
images = []
exist = []
trueCount = 0
for ind in dfManifest.index:
    print(f'Checking subject {ind}/{dfManifest.shape[0]}')
    img1 = dfManifest['img1'][ind]
    img2 = dfManifest['img2'][ind]
    img3 = dfManifest['img3'][ind]
    images.append(img1)
    if Path(path + img1).is_file():
        exist.append("TRUE")
        trueCount = trueCount + 1
    else:
        exist.append("FALSE")
    images.append(img2)
    if Path(path + img2).is_file():
        exist.append("TRUE")
        trueCount = trueCount + 1
    else:
        exist.append("FALSE")
    images.append(img3)
    if Path(path + img3).is_file():
        exist.append("TRUE")
        trueCount = trueCount + 1
    else:
        exist.append("FALSE")

percentTrue = trueCount/len(images) * 100
print(f'{percentTrue}% of images in manifest are present in {path}')
# Use for output dataframe  when ready
dfo = pd.DataFrame(data={'images': images,
                         'exist': exist})

# Output file
dfo.to_csv('imageExistance.csv')
