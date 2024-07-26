#!/usr/bin/env python3

import os
import subprocess
import numpy as np
from scipy.io import savemat
import sys

homedir = os.getcwd()

# Maximum number of rows allowed after downsampling
MAX_ROWS = 999999

# List of specific folder names to check
#specific_folders = ["021D"]
specific_folders = ["014A", "021D", "116A", "123D"]

# Processing .tif files
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for tifs in os.listdir(folder_path):
            if (tifs.endswith('msk.azi.tif') or tifs.endswith('msk.rng.tif')) and tifs.startswith('20'):
                print(f'Processing {tifs} file')
                pair = tifs[:17]
                t_type = tifs[22:29]
                output_file = f"{pair}.{t_type}.txt"

                # Check if the output file already exists
                if os.path.exists(os.path.join(folder_path, output_file)):
                    print(f"{output_file} already exists, skipping...")
                    continue

                # Construct the command
                command = f'gdal_translate -of XYZ {tifs} {output_file}'

                # Change to the folder directory, run the command, then change back
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

# Processing .tif files for BOI
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for tifs in os.listdir(folder_path):
            if tifs.endswith('_boi.tif'):
                print(f'Processing {tifs} file')
                output_file = tifs.replace('.tif', '.txt')
                # Check if the output file already exists
                if os.path.exists(os.path.join(folder_path, output_file)):
                    print(f"{output_file} already exists, skipping...")
                    continue

                # Construct the command
                command = f'gdal_translate -of XYZ {tifs} {output_file}'

                # Change to the folder directory, run the command, then change back
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)



print("XYZ created, let's remove NaN values")

# Removing NaN values from the data
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for txts in os.listdir(folder_path):
            if txts.endswith('msk.azi.txt') or txts.endswith('msk.rng.txt') or txts.endswith('_boi.txt'):
                # Construct the command to remove NaN values
                output_cleaned_file = txts.replace('.txt', '.nonan.txt')
                command = f"awk '$3 != \"nan\" {{print $0}}' {txts} > {output_cleaned_file}"

                # Change to the folder directory, run the command, then change back
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

print("NaN values removed, let's process and downsample the data")

# Processing and downsampling the nonan.txt files
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for txts in os.listdir(folder_path):
            if txts.endswith('.nonan.txt'):
                input_file_path = os.path.join(folder_path, txts)
                downsampled_file_path = input_file_path.replace('.nonan.txt', '.downsampled.txt')

                # Read the data
                data = np.loadtxt(input_file_path)

                # Replace NaN and zero values in the third column with 0.0001
                data[:, 2][data[:, 2] == 0] = 0.0001
                data[:, 2][np.isnan(data[:, 2])] = 0.0001

                # Downsample the data (select every 10th row)
                downsampled_data = data[::10]

                # Remove rows with NaNs (after replacement there should be none, but this is a safety check)
                downsampled_data = downsampled_data[~np.isnan(downsampled_data).any(axis=1)]

                # Further downsample if the number of rows exceeds MAX_ROWS
                if len(downsampled_data) > MAX_ROWS:
                    downsample_factor = len(downsampled_data) // MAX_ROWS + 1
                    downsampled_data = downsampled_data[::downsample_factor]

                # Save the downsampled data
                np.savetxt(downsampled_file_path, downsampled_data, fmt='%f')

print("Data downsampled, let's add metadata values and save to .mat file")
def m2rad_s1(inm):
    speed_of_light = 299792458  # m/s
    radar_freq = 5.405e9  # for S1
    wavelength = speed_of_light / radar_freq  # meter
    coef_r2m = -wavelength / 4 / np.pi  # rad -> mm, positive is -LOS
    outrad = inm / coef_r2m
    return outrad

# range
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        downsampled_rng = None  # Initialize downsampled_rng to None
        for txts in os.listdir(folder_path):
            if txts.endswith('rng.downsampled.txt') and txts.startswith('20'):
               downsampled_rng = txts

        if downsampled_rng is None:
            print(f"No rng.downsampled.txt file found in {folder_path}")
            continue  # Skip this folder if no rng.downsampled.txt file is found

        for txts in os.listdir(folder_path):
            if txts.endswith('geo.E.tif') or txts.endswith('geo.N.tif') or txts.endswith('geo.U.tif'):
                print(f'Processing {txts} file')
                track = txts[:4]
                LoSv = txts[-5]

                downsample_point = os.path.join(folder_path, downsampled_rng)  # Define downsample_point
                output_file = f'{track}.downs.{LoSv}.txt'

                print(f'{track}.{LoSv}.txt')
                command = f'gmt grdtrack -G{txts} {downsample_point} > {output_file}'

                # Change to the folder directory, run the command, then process with awk, then change back
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                    
                    # Use awk to extract $1, $2, and $4 columns and overwrite the output file
                    awk_command = f'awk \'{{print $1, $2, $4}}\' {output_file} > {output_file}.tmp && mv {output_file}.tmp {output_file}'
                    subprocess.run(awk_command, shell=True, check=True)
                    
                finally:
                    os.chdir(homedir)

for folder in specific_folders:
    folder_path=os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
       for ENU in os.listdir(folder_path):
           if ENU.endswith('E.txt') or ENU.endswith('N.txt') or ENU.endswith('U.txt'):
              track = ENU[:4]
              command=f"paste {track}.downs.E.txt {track}.downs.N.txt {track}.downs.U.txt | awk '{{print $1, $2, $3, $6, $9}}' > {track}.downs.ENU.rng.txt"
              try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
              finally:
                    os.chdir(homedir)




# Creating InSAR input of slipBERI
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        # First, find the ENU file
        ENU_file = None
        for file in os.listdir(folder_path):
            if file.endswith('.downs.ENU.rng.txt'):
                ENU_file = file
                print(f"Found ENU file: {ENU_file}")
                break  # Exit the loop once the ENU file is found

        # Ensure we have found an ENU file before proceeding
        if not ENU_file:
            print(f"No ENU file found in {folder_path}")
            continue

        # Process rng.downs.txt files using the found ENU file
        for downs in os.listdir(folder_path):
            if downs.endswith('rng.downsampled.txt'):
                print(f"Processing rng file: {downs}")
                id = downs[:25]
                command = f"paste {downs} {ENU_file} | awk '$3 != \"NaN\" && $6 != 0 {{print $1, $2, ($3*-1), $6, $7, $8}}' > {id}.inp"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)



# azi
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        downsampled_rng = None  # Initialize downsampled_rng to None
        for txts in os.listdir(folder_path):
            if txts.endswith('azi.downsampled.txt') and txts.startswith('20'):
                downsampled_azi = txts

        if downsampled_azi is None:
            print(f"No rng.downsampled.txt file found in {folder_path}")
            continue  # Skip this folder if no rng.downsampled.txt file is found

        for txts in os.listdir(folder_path):
            if txts.endswith('azi.E.tif') or txts.endswith('azi.N.tif') or txts.endswith('azi.U.tif'):
                print(f'Processing {txts} file')
                track = txts[:4]
                LoSv = txts[-5]

                downsample_point = os.path.join(folder_path, downsampled_azi)  # Define downsample_point
                output_file = f'{track}.downs.{LoSv}.txt'

                print(f'{track}.{LoSv}.txt')
                command = f'gmt grdtrack -G{txts} {downsample_point} > {output_file}'

                # Change to the folder directory, run the command, then process with awk, then change back
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)

                    # Use awk to extract $1, $2, and $4 columns and overwrite the output file
                    awk_command = f'awk \'{{print $1, $2, $4}}\' {output_file} > {output_file}.tmp && mv {output_file}.tmp {output_file}'
                    subprocess.run(awk_command, shell=True, check=True)

                finally:
                    os.chdir(homedir)

for folder in specific_folders:
    folder_path=os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
       for ENU in os.listdir(folder_path):
           if ENU.endswith('E.txt') or ENU.endswith('N.txt') or ENU.endswith('U.txt'):
              track = ENU[:4]
              command=f"paste {track}.downs.E.txt {track}.downs.N.txt {track}.downs.U.txt | awk '{{print $1, $2, $3, $6, $9}}' > {track}.downs.ENU.azi.txt"
              try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
              finally:
                    os.chdir(homedir)


# Creating InSAR input of slipBERI
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        # First, find the ENU file
        ENU_file = None
        for file in os.listdir(folder_path):
            if file.endswith('.downs.ENU.azi.txt'):
                ENU_file = file
                print(f"Found ENU file: {ENU_file}")
                break  # Exit the loop once the ENU file is found

        # Ensure we have found an ENU file before proceeding
        if not ENU_file:
            print(f"No ENU file found in {folder_path}")
            continue

        # Process rng.downs.txt files using the found ENU file
        for downs in os.listdir(folder_path):
            if downs.endswith('azi.downsampled.txt'):
                print(f"Processing rng file: {downs}")
                id = downs[:25]
                command = f"paste {downs} {ENU_file} | awk '$3 != \"NaN\" && $6 != 0 {{print $1, $2, ($3*-1), $6, $7, $8}}' > {id}.inp"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)


###boi
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        downsampled_rng = None  # Initialize downsampled_rng to None
        for txts in os.listdir(folder_path):
            if txts.endswith('boi.downsampled.txt'):
                downsampled_azi = txts

        if downsampled_azi is None:
            print(f"No boi.downsampled.txt file found in {folder_path}")
            continue  # Skip this folder if no rng.downsampled.txt file is found

        for txts in os.listdir(folder_path):
            if txts.endswith('azi.E.tif') or txts.endswith('azi.N.tif') or txts.endswith('azi.U.tif'):
                print(f'Processing {txts} file')
                track = txts[:4]
                LoSv = txts[-5]
                downsample_point = os.path.join(folder_path, downsampled_azi)  # Define downsample_point
                output_file = f'{track}.downs.{LoSv}.txt'

                print(f'{track}.{LoSv}.txt')
                command = f'gmt grdtrack -G{txts} {downsample_point} > {output_file}'

                # Change to the folder directory, run the command, then process with awk, then change back
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)

                    # Use awk to extract $1, $2, and $4 columns and overwrite the output file
                    awk_command = f'awk \'{{print $1, $2, $4}}\' {output_file} > {output_file}.tmp && mv {output_file}.tmp {output_file}'
                    subprocess.run(awk_command, shell=True, check=True)

                finally:
                    os.chdir(homedir)

for folder in specific_folders:
    folder_path=os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
       for ENU in os.listdir(folder_path):
           if ENU.endswith('E.txt') or ENU.endswith('N.txt') or ENU.endswith('U.txt'):
              track = ENU[:4]
              command=f"paste {track}.downs.E.txt {track}.downs.N.txt {track}.downs.U.txt | awk '{{print $1, $2, $3, $6, $9}}' > {track}.downs.ENU.azi.txt"
              try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
              finally:
                    os.chdir(homedir)


# Creating InSAR input of slipBERI
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        # First, find the ENU file
        ENU_file = None
        for file in os.listdir(folder_path):
            if file.endswith('.downs.ENU.azi.txt'):
                ENU_file = file
                print(f"Found ENU file: {ENU_file}")
                break  # Exit the loop once the ENU file is found

        # Ensure we have found an ENU file before proceeding
        if not ENU_file:
            print(f"No ENU file found in {folder_path}")
            continue

        # Process rng.downs.txt files using the found ENU file
        for downs in os.listdir(folder_path):
            if downs.endswith('boi.downsampled.txt'):
                print(f"Processing rng file: {downs}")
                id = downs[:8]
                command = f"paste {downs} {ENU_file} | awk '$3 != \"NaN\" && $6 != 0 {{print $1, $2, ($3*-1), $6, $7, $8}}' > {id}.inp"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

