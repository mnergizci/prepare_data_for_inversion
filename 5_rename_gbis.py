#!/usr/bin/env python3

import os
import numpy as np
from scipy.io import savemat

def load_gbis_data(filepath):
    # Load the data from the .gbis file
    data = np.loadtxt(filepath)

    # Extract columns and reshape as necessary
    Lon = data[:, 0].reshape(-1, 1)
    Lat = data[:, 1].reshape(-1, 1)
    Phase_rad = data[:, 2].reshape(-1, 1)
    Inc = data[:, 3].reshape(-1, 1)
    Heading = data[:, 4].reshape(-1, 1)

    return Lon, Lat, Phase_rad, Inc, Heading

# Get the current working directory
homedir = os.getcwd()

# List of specific folders to search in
specific_folders = ["014A", "021D", "116A", "123D"]

# Loop through each specified folder
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):  # Check if the folder exists
        for gbis in os.listdir(folder_path):  # List all files in the folder
            if gbis.endswith('.gbis'):  # Check if the file ends with '.gbis'
                if gbis.startswith('20'):  # Check if the file name starts with '20'
                    print(folder, gbis)
                    # Define the source and destination paths
                    src_path = os.path.join(folder_path, gbis)
                    dest_path = os.path.join(folder_path, f'{folder}_{gbis.replace(".gbis", ".mat")}')

                    # Load your data from the .gbis file
                    Lon, Lat, Phase_rad, Inc, Heading = load_gbis_data(src_path)

                    # Save the data to a .mat file
                    savemat(dest_path, {'Lon': Lon, 'Lat': Lat, 'Phase': Phase_rad, 'Inc': Inc, 'Heading': Heading})
                    print(f'Saved {dest_path}')
                else:
                    print(folder, gbis)
                    # Define the source and destination paths
                    src_path = os.path.join(folder_path, gbis)
                    dest_path = os.path.join(folder_path, gbis.replace(".gbis", ".mat"))

                    # Load your data from the .gbis file
                    Lon, Lat, Phase_rad, Inc, Heading = load_gbis_data(src_path)

                    # Save the data to a .mat file
                    savemat(dest_path, {'Lon': Lon, 'Lat': Lat, 'Phase': Phase_rad, 'Inc': Inc, 'Heading': Heading})
                    print(f'Saved {dest_path}')
