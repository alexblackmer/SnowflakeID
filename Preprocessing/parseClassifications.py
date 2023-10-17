#!/usr/bin/env python
"""
This script is used to parse the raw classifications in snowflake-id-classifications.csv to a usable attribute array

"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch"]
__version__ = "1.0.1"

import pandas as pd


def get_image_ids(subject_data):
    # Defines substrings to split input string at
    sub1 = '"img1":"'
    sub2 = '","img2":"'
    sub3 = '","img3":"'
    sub4 = '","subject_id":"'
    # Gets positional ids of each substring
    idsub1 = subject_data.find(sub1)
    idsub2 = subject_data.find(sub2)
    idsub3 = subject_data.find(sub3)
    idsub4 = subject_data.find(sub4)
    # Splits input string into camera ids
    cam0 = subject_data[idsub1 + len(sub1): idsub2]
    cam1 = subject_data[idsub2 + len(sub2): idsub3]
    cam2 = subject_data[idsub3 + len(sub3): idsub4]

    return cam0, cam1, cam2


def get_label(wf, wfid):
    """
    This function gets label from defined workflow.

    Parameters:
    wf (Str): The classification workflow with label data
    wfid (str): The workflow version ID
    imid (str) The image file ID
    Returns:
    str: The label defined by workflow
    """
    label = ''
    riming_degree = ''
    # Parses list into work flow lines
    wfParsed = [str(task) for task in wf.split(',{"task"')]
    for task in wfParsed:
        # Checks if all images are blurry
        if task.find('T1') > -1:
            if task.find('"Yes') > -1:
                label = 'B'

        # # Checks if one of the three images are blurry, and labels this image accordingly
        # elif task.find('T8') > -1:
        #     # If the left image is blurry and this image is the left one, then label blurry.
        #     if task.find('Left') > -1 and imid[-5] == '0':
        #         label = 'B'
        #     # Same as above for center
        #     elif task.find('Center') > -1 and imid[-5] == '1':
        #         label = 'B'
        #     # Same as above for right
        #     elif task.find('Right') > -1 and imid[-5] == '2':
        #         label = 'B'

        # Checks degree of melting
        elif task.find('T7') > -1:
            if task.find('"No"') > -1:
                break
            elif task.find('partially') > -1:
                label = 'PM'
            elif task.find('mostly') > -1:
                label = 'MM'
            elif task.find('completely') > -1:
                label = 'RA'

        # Checks degree of riming
        elif task.find('T2') > -1:
            if task.find('"None"') > -1:
                riming_degree = '1'
            elif task.find('"Lightly Rimed"') > -1:
                riming_degree = '2'
            elif task.find('"Densely Rimed"') > -1:
                riming_degree = '3'
            elif task.find('"Graupel-like"') > -1:
                label = 'SN-GL'
            elif task.find('"Graupel"') > -1:
                label = 'GR'

        # Checks if crystal or aggregate
        elif task.find('T4') > -1:
            if task.find('"Aggregate"') > -1:
                label = 'SN-AGG'
            elif task.find('"Not sure"') > -1:
                label = 'SN-NS'

        # Checks crystal type
        elif task.find('T5') > -1:
            if task.find('"Plate"') > -1:
                label = 'SN-P'
            elif task.find('"Column"') > -1:
                label = 'SN-C'
            elif task.find('"Combination plate and column"') > -1:
                label = 'SN-PC'
            elif task.find('"Not sure"') > -1:
                label = 'SN-NSPC'
    return label + riming_degree


# Initialized containers for output data. They are arrays because there may be multiple classifications for
# a single image, thus no data is omitted.
cam0Im = []
cam1Im = []
cam2Im = []
labels = []

# Reads input data to data frame
dfi = pd.read_csv('../data/snowflake-id-classifications.csv',
                  usecols=['subject_data', 'annotations', 'workflow_version'])

wfList = ['125.254', '125.258', '125.261', '125.263', '125.266']


for ind in dfi.index:
    if str(dfi['workflow_version'][ind]) in wfList:
        cams = get_image_ids(dfi['subject_data'][ind])
        if cams[0][0] != '2':
            # Computes output data
            label = get_label(dfi['annotations'][ind],str(dfi['workflow_version'][ind]))
            if label != '' and label != 'B':
                cam0Im.append(cams[0])
                cam1Im.append(cams[1])
                cam2Im.append(cams[2])
                labels.append(label)

# Use for output dataframe  when ready
dfo = pd.DataFrame(data={'img1': cam0Im,
                         'img2': cam1Im,
                         'img3': cam2Im,
                         'label': labels})

# Output file
dfo.to_csv('parsedClassifications.csv')
