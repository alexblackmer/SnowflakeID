#!/usr/bin/env python
"""
This script is used to group processed input data into 5 broader classifications
and find the agreement for each image. Uses softmax to find probability distribution
of labels for each classification, then finds best label. Also computes entropy of each
distribution to quantify label agreement.

"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch", "Tim Garrett"]
__version__ = "1.0.2"

import pandas as pd
import scipy.stats
from scipy.special import softmax


def get_broad_label(lab):
    """
    This function gets one of 5 broader labels from categories defined by sub labels.

    Parameters:
    lab (Str): sub label
    Returns:
    str: The broader label
    """
    # Sub label categories
    RA = ['RA']
    RASN = ['PM', 'MM']
    SN = ['SN-GL', 'SN-AGG1', 'SN-AGG2', 'SN-AGG3', 'SN-NS1', 'SN-NS2', 'SN-NS3',
          'SN-P2', 'SN-P3', 'SN-C2', 'SN-C3', 'SN-PC2', 'SN-PC3', 'SN-NSPC2', 'SN-NSPC3']
    GR = ['GR']
    PR = ['SN-P1', 'SN-C1', 'SN-PC1', 'SN-NSPC1']

    if lab in RA:
        return 'RA'
    elif lab in RASN:
        return 'RASN'
    elif lab in SN:
        return 'SN'
    elif lab in GR:
        return 'GR'
    elif lab in PR:
        return 'PR'
    else:
        return 'Error'


def find_agreement(labs):
    """
    This function finds agreement between a set of labels

    Parameters:
    labs (Str): Set of labels
    Returns:
    agr[ind] (int): The agreement metric
    agr_labels[ind] (str): The corresponding label
    """

    agr = [0, 0, 0, 0, 0]
    agr_labels = ['RA', 'RASN', 'SN', 'GR', 'PR']
    blurry = 0  # Keeps rack of number of blurry labels assigned to subject
    for lab in labs:
        if lab == agr_labels[0]:
            agr[0] = agr[0] + 1
        elif lab == agr_labels[1]:
            agr[1] = agr[1] + 1
        elif lab == agr_labels[2]:
            agr[2] = agr[2] + 1
        elif lab == agr_labels[3]:
            agr[3] = agr[3] + 1
        elif lab == agr_labels[4]:
            agr[4] = agr[4] + 1
        elif lab == 'B':
            blurry = blurry + 1

    # Uses softmax probability method to find which label is most prevalent.
    sM = softmax(agr)
    # Finds Shannon entropy over probability distribution to describe the quality of agreement
    en = scipy.stats.entropy(sM)
    # Index that defines blurriness
    blurry_ind = blurry/len(labs)

    _ind = 0  # ind variable to store the index of maximum value in the list
    max_element = sM[0]
    for i in range(1, len(sM)):  # iterate over array
        if sM[i] > max_element:  # to check max value
            max_element = sM[i]
            _ind = i

    return agr_labels[_ind], en


# Reads input data to data frame
dfi = pd.read_csv('InputData.csv',
                  usecols=['Image', 'Subject ID', 'Classification ID', 'Temperature', 'Label'])
images = []
subjects = []
blabel = []
entropy = []

# Fills list of subjects
for ind in dfi.index:
    if ind == 0 or dfi['Subject ID'][ind] != subjects[-1]:
        images.append([dfi['Image'][ind]])
        subjects.append(dfi['Subject ID'][ind])
    if ind != 0 and dfi['Subject ID'][ind] == subjects[-1]:
        images[-1].append(dfi['Image'][ind])

f_images = []
f_labels = []
# Finds labels for each subject
for i in range(0, len(subjects)):
    tempLabels = []
    for ind in dfi.index:
        if subjects[i] == dfi['Subject ID'][ind]:
            tempLabels.append(get_broad_label(dfi['Label'][ind]))
    blabel.append(find_agreement(tempLabels)[0])
    entropy.append(find_agreement(tempLabels)[1])

    # Fills image and best label
    if entropy[-1] < 0.1:       # Entropy cut-off
        for item in images[i]:
            if len(f_images) == 0 or item != f_images[-1]:
                f_images.append(item)
                f_labels.append(blabel[-1])
                print(str(f_images[-1]) + " " + str(f_labels[-1]))

# Outputs entropy data for analysis
dfo1 = pd.DataFrame(data={'Subject ID': subjects,
                          'Label': blabel,
                          'Entropy': entropy})
# Output file
dfo1.to_csv('LabelAgreement.csv')

# Outputs entropy data for analysis
dfo2 = pd.DataFrame(data={'Image': f_images,
                          'Label': f_labels})
# Output file
dfo2.to_csv('ImageLabelPairs.csv')
