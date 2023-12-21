"""
Creates CSV of all image filenames present in /images/.
Used to compare which images are present to those listed in manifest

"""
__author__ = "Alex Blackmer"
__credits__ = ["Alex Blackmer"]

import os
import csv
import pandas as pd


def list_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    files = os.listdir(folder_path)
    # Filter files that are not directories and do not start with '.'
    files = [file for file in files if os.path.isfile(os.path.join(folder_path, file)) and not file.startswith('.')
             and not file.startswith('flake')]
    return files


folder_path = 'images'
# Creates DataFrame for image filenames present in images folder
dfImages = pd.DataFrame(data={'Files': list_files(folder_path)})
# Creates DataFrame for image filenames in manifest file
dfi = pd.read_csv('flakeManifest2b.csv',
                         usecols=['img1', 'img2', 'img3', 'temperature'])
images = []
temps = []
for ind in dfi.index:
    for i in range(1,4):
        images.append(dfi[f'img{i}'][ind])
        temps.append(dfi['temperature'][ind])
        
dfManifest = pd.DataFrame(data={'Files': images, 'Temperature': temps})

# Output file
dfImages.to_csv('filesInImageFolder.csv', index=False, header=False)
dfManifest.to_csv('filesInManifest.csv', index=False)


# Convert 'Files' column in each dataframe to sets
files_dfImages = set(dfImages['Files'])
files_dfManifest = set(dfManifest['Files'])

# Find common items (intersection)
common_files = list(files_dfImages.intersection(files_dfManifest))

# Find different items (symmetric difference)
different_files = list(files_dfImages.symmetric_difference(files_dfManifest))


print("Common Items:")
print(common_files)
print("\nDifferent Items:")
print(different_files)
