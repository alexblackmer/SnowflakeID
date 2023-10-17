#!/usr/bin/env python
"""
This script is used to assign each image a label from classifications done on subjects.
The output data are the image filename, the subject ID, the workflow version, and the label.
The output data is then used by FindBroadLabelAgreement.py to find label agreement for each subject.

"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer", "Kyle Fitch"]
__version__ = "1.0.1"

import pandas as pd
import os
import re


def get_label(wf, wfid, imid):
    """
    This function gets label from defined workflow.

    Parameters:
    wf (Str): The classification workflow with label data
    wfid (str): The workflow version ID
    imid (str) The image file ID
    Returns:
    str: The label defined by workflow
    """
    wfList = ['114.251', '114.254', '125.254', '125.258', '125.261', '125.263', '125.266']       # Valid workflows
    label = ''
    riming_degree = ''
    if wfid in wfList:
        # Parses list into work flow lines
        wfParsed = [str(task) for task in wf.split(',{"task"')]
        for task in wfParsed:
            # Checks if all images are blurry
            if task.find('T1') > -1:
                if task.find('"Yes') > -1:
                    label = 'B'

            # Checks if one of the three images are blurry, and labels this image accordingly
            elif task.find('T8') > -1:
                # If the left image is blurry and this image is the left one, then label blurry.
                if task.find('Left') > -1 and imid[-5] == '0':
                    label = 'B'
                # Same as above for center
                elif task.find('Center') > -1 and imid[-5] == '1':
                    label = 'B'
                # Same as above for right
                elif task.find('Right') > -1 and imid[-5] == '2':
                    label = 'B'

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
                # Returns melted label for specific workflows to prevent degree of riming application
                if wfid == '114.251' or wfid == '114.254':
                    return label

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
    else:
        return 'Invalid Workflow'


def get_temp(sentence):
    """
    This function gets temperature from subject data.

    Parameters:
    sentence (Str): The subject_data string containing temp
    Returns:
    str: The temperature in C
    """
    temp_sen = sentence.split('"temperature":"', 1)
    temp = re.findall(r'-?\d+\.?\d*', temp_sen[1])
    return temp[0]


# Gets list of images
imageFilenames = []
for file in os.listdir('../../RawImages/'):
    if file.endswith('.png'):
        imageFilenames.append(file)

# Initialized containers for output data. They are arrays because there may be multiple classifications for
# a single image, thus no data is omitted.
imageID = []
subjectID = []
workflowID = []
classificationID = []
tempID = []
labelID = []
# Reads input data to data frame
dfi = pd.read_csv('snowflake-id-classifications.csv',
                  usecols=['subject_ids', 'subject_data', 'annotations', 'workflow_version', 'classification_id'])
numImages = len(imageFilenames)
# Iterates for each image
for i in range(numImages):
    # Iterates over each subject classification in input data
    print('Image ' + str(i) + "/" + str(numImages))    # For development
    for ind in dfi.index:
        # Checks if subject data contains the file image name
        if dfi['subject_data'][ind].find(imageFilenames[i]) > -1:
            # Computes output data
            label = get_label(dfi['annotations'][ind], str(dfi['workflow_version'][ind]), imageFilenames[i])
            if label != '':
                imageID.append(imageFilenames[i])
                subjectID.append(str(dfi['subject_ids'][ind]))
                classificationID.append(str(dfi['classification_id'][ind]))
                workflowID.append(str(dfi['workflow_version'][ind]))
                tempID.append(get_temp(dfi['subject_data'][ind]))
                labelID.append(label)

# Use for output dataframe  when ready
dfo = pd.DataFrame(data={'Image': imageID,
                         'Subject ID': subjectID,
                         'Classification ID': classificationID,
                         'Workflow': workflowID,
                         'Temperature': tempID,
                         'Label': labelID})
# Output file
dfo.to_csv('InputDataAllWF.csv')

