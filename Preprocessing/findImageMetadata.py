#!/usr/bin/env python
"""
This script is used to assign labels and temperatures to each image in images. Uses flakeManifest2.csv and parsedClassifications.csv
for temp and labels.

"""
__author__ = "Alex Blackmer", "Alex Garrett"
__credits__ = ["Alex Blackmer", "Alex Garrett", "Kyle Fitch"]
__version__ = "1.0.1"

import cv2
import pandas as pd
from pathlib import Path
import scipy.stats
from scipy.special import softmax
import ast


def get_habit_label(lab):
    """
    This function gets one of 7 habit-based labels from categories defined by sub labels.

    Parameters:
    lab (Str): sub label
    Returns:
    str: The habit label
    """
    # Sub label categories
    SmallParticle = ['SP']
    Graupel = ['SN-GL', 'GR']
    Aggregate = ['SN-AGG1', 'SN-AGG2', 'SN-AGG3']
    Plate = ['SN-P1', 'SN-P2', 'SN-P3']
    Column = ['SN-C1', 'SN-C2', 'SN-C3']

    if lab in SmallParticle:
        return 'SP'
    elif lab in Graupel:
        return 'GR'
    elif lab in Aggregate:
        return 'AGG'
    elif lab in Plate:
        return 'PC'
    elif lab in Column:
        return 'CC'
    else:
        return 'N/A'


def get_rime_label(lab):
    """
    This function gets one of 6 rime-based labels from categories defined by sub labels.

    Parameters:
    lab (Str): sub label
    Returns:
    str: The rime label
    """
    # Sub label categories
    R1 = ['SN-AGG1', 'SN-P1', 'SN-C1', 'SN-PC1', 'SN-NS1', 'SN-NSPC1']
    R2 = ['SN-AGG2', 'SN-P2', 'SN-C2', 'SN-PC2', 'SN-NS2', 'SN-NSPC2']
    R3 = ['SN-AGG3', 'SN-P3', 'SN-C3', 'SN-PC3', 'SN-NS3', 'SN-NSPC3']
    R4 = ['SN-GL']
    R5 = ['GR']

    if lab in R1:
        return 'R1'
    elif lab in R2:
        return 'R2'
    elif lab in R3:
        return 'R3'
    elif lab in R4:
        return 'R4'
    elif lab in R5:
        return 'R5'
    else:
        return 'N/A'


def get_melt_label(lab):
    """
    This function gets one of 2 melt-based labels from categories defined by sub labels.

    Parameters:
    lab (Str): sub label
    Returns:
    str: The melt label
    """
    # Sub label categories
    meltLabels = ['PM', 'MM']
    dryLabels = ['SN-AGG1', 'SN-P1', 'SN-C1', 'SN-PC1', 'SN-NS1', 'SN-NSPC1',
                 'SN-AGG2', 'SN-P2', 'SN-C2', 'SN-PC2', 'SN-NS2', 'SN-NSPC2',
                 'SN-AGG3', 'SN-P3', 'SN-C3', 'SN-PC3', 'SN-NS3', 'SN-NSPC3',
                 'SN-GL', 'GR']

    if lab in meltLabels:
        return 'Melted'
    elif lab in dryLabels:
        return 'Dry'
    else:
        return 'N/A'


def find_agreement(labs):
    """
    This function finds agreement between a set of labels

    Parameters:
    labs (Str): Set of labels
    Returns:
    agr[ind] (int): The agreement metric
    agr_labels[ind] (str): The corresponding label
    """

    # Checks agreement of individual sub labels
    agr_labels = ['RA', 'PM', 'MM', 'SN-GL', 'SN-AGG1', 'SN-AGG2', 'SN-AGG3', 'SN-NS1', 'SN-NS2', 'SN-NS3',
          'SN-P2', 'SN-P3', 'SN-C2', 'SN-C3', 'SN-PC2', 'SN-PC3', 'SN-NSPC2', 'SN-NSPC3', 'GR','SN-P1', 'SN-C1', 'SN-PC1', 'SN-NSPC1']
    # Create list matrix to track number of labels
    agr = []
    for index in range(len(agr_labels)):
        agr.append(0)
    # Iterate over each label for agreement
    for lab in labs:
        # Check if label matches any defined agreement labels above
        for index, label in enumerate(agr_labels):
            if lab == label:
                agr[index] = agr[index] + 1


    # Uses softmax probability method to find which label is most prevalent.
    sM = softmax(agr)
    # Finds Shannon entropy over probability distribution to describe the quality of agreement
    en = scipy.stats.entropy(sM)
    # Index that defines blurriness

    _ind = 0  # ind variable to store the index of maximum value in the list
    max_element = sM[0]
    for _i in range(1, len(sM)):  # iterate over array
        if sM[_i] > max_element:  # to check max value
            max_element = sM[_i]
            _ind = _i

    return agr_labels[_ind], en

    # Uses softmax probability method to find which label is most prevalent.
    sM = softmax(agr)
    # Finds Shannon entropy over probability distribution to describe the quality of agreement
    en = scipy.stats.entropy(sM)
    # Index that defines blurriness
    blurry_ind = blurry / len(labs)

    _ind = 0  # ind variable to store the index of maximum value in the list
    max_element = sM[0]
    for i in range(1, len(sM)):  # iterate over array
        if sM[i] > max_element:  # to check max value
            max_element = sM[i]
            _ind = i

    return agr_labels[_ind], en


def find_lp_blur(image: str):
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


path = '../data/images/'
# Gets classification data
dfClasses = pd.read_csv('../data/classifications/parsedClassificationsDict.csv', header=None, names=['File', 'Labels'])
dfClasses['Labels'] = dfClasses['Labels'].apply(ast.literal_eval)
classDict = dfClasses.set_index('File')['Labels'].to_dict()
# Gets manifest data
dfManifest = pd.read_csv('../data/filesInManifest.csv')

labeledImages = []
unlabeledImages = []
habitLabels = []
rimeLabels = []
meltLabels = []
labelSets = []
labelSetEntropys = []
temps = []
blur = []

# For each unique image in manifest, find each associated classification label
for img in dfManifest.index:
    print(f'Finding metadata of Image {img}/{dfManifest.shape[0]}')
    imageFilename = dfManifest.loc[img, 'Files']
    labels = classDict.get(dfManifest.loc[img, 'Files'])

    # Compiles metadata if image is present in classification list
    try:
        labels = classDict[dfManifest.loc[img, 'Files']]  # Attempt to retrieve the value

        # Compiles metadata if image is present in classification list
        mostCommonLabel, setEntropy = find_agreement(labels)
        rLabel = get_rime_label(mostCommonLabel)  # Rime Label
        hLabel = get_habit_label(mostCommonLabel)  # Habit Label
        mLabel = get_melt_label(mostCommonLabel)    # Melt Label
        imageBlur = find_lp_blur(path + imageFilename)  # Blurriness
        # Compiles image metadata
        blur.append(imageBlur)
        labeledImages.append(imageFilename)
        habitLabels.append(hLabel)
        rimeLabels.append(rLabel)
        meltLabels.append(mLabel)
        labelSets.append(labels)
        labelSetEntropys.append(setEntropy)
        temps.append(dfManifest.loc[img, 'Temperature'])

    except KeyError:
        # Image filename is not found in classifications
        unlabeledImages.append(imageFilename)

# Use for output dataframe  when ready
dfo = pd.DataFrame(data={'Image Filename': labeledImages,
                         'Habit Label': habitLabels,
                         'Rime Label': rimeLabels,
                         'Melt Label': meltLabels,
                         'Label Set': labelSets,
                         'Label Set Entropy': labelSetEntropys,
                         'Temperature': temps,
                         'Laplacian Blur': blur})
# Output file
dfo.to_csv('imageMetadata.csv')
