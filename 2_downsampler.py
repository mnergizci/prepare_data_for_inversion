#!/usr/bin/env python3

import os
import subprocess
import sys
homedir = os.getcwd()
downsample_point='down_3500s' #'downs_point_1.5_10_20' #downsample_point
downsample_point='down_buffered_4500'
downsample_point='down_sparse'
#downsample_point='down_9000'
# List of specific folder names to check
specific_folders = ["014A", "021D", "116A", "123D"]

# Processing .tif files
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for tifs in os.listdir(folder_path):
            if tifs.endswith('rng.tif'): #msk.
                if tifs.startswith('20'):
                    print(f'Processing {tifs} file')
                    pair = tifs[:17]
                    t_type = tifs[-11:-4]  # Extract from the 11th character from the end to the 5th character from the end

                    # Construct the command
                    command = f'gmt grdtrack -G{tifs} {downsample_point} > {pair}.{t_type}.downs.txt'

                    # Change to the folder directory, run the command, then change back
                    try:
                        os.chdir(folder_path)
                        subprocess.run(command, shell=True, check=True)
                    finally:
                        os.chdir(homedir)

            if tifs.endswith('geo.E.tif') or tifs.endswith('geo.N.tif') or tifs.endswith('geo.U.tif'):
                 print(f'Processing {tifs} file')
                 track = tifs[:4]
                 LoSv = tifs[-5]

                 print(f'{track}.{LoSv}.txt')
                 command = f'gmt grdtrack -G{tifs} {downsample_point} > {track}.downs.{LoSv}.txt'

                 # Change to the folder directory, run the command, then change back
                 try:
                     os.chdir(folder_path)
                     subprocess.run(command, shell=True, check=True)
                 finally:
                     os.chdir(homedir)

print('Datasets downsampled! So let’s merge los with ENU vectors')

# Merging los with ENU vectors
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for ENU in os.listdir(folder_path):
            if ENU.endswith('E.txt') or ENU.endswith('N.txt') or ENU.endswith('U.txt'):
                track = ENU[:4]
                command = f"paste {track}.downs.E.txt {track}.downs.N.txt {track}.downs.U.txt | awk '{{print $1, $2, $3, $6, $9}}' > {track}.downs.ENU.txt"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

print('ENU.txt created. Let’s create InSAR input of slipBERI')


# Creating InSAR input of slipBERI
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        # First, find the ENU file
        ENU_file = None
        for file in os.listdir(folder_path):
            if file.endswith('.downs.ENU.txt'):
                ENU_file = file
                print(f"Found ENU file: {ENU_file}")
                break  # Exit the loop once the ENU file is found

        # Ensure we have found an ENU file before proceeding
        if not ENU_file:
            print(f"No ENU file found in {folder_path}")
            continue

        # Process rng.downs.txt files using the found ENU file
        for downs in os.listdir(folder_path):
            if downs.endswith('rng.downs.txt') or downs.endswith('pha.downs.txt'):
                print(f"Processing rng file: {downs}")
                id = downs[:25]
                command = f"paste {downs} {ENU_file} | awk '$3 != \"NaN\" && $6 != 0 {{print $1, $2, ($3*-1), $6, $7, $8}}' > {id}.inp"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)


######################
downsample_point='down_7000'
downsample_point='down_buffered_4500'
downsample_point='down_sparse'
# List of specific folder names to check
#specific_folders = ["014A", "021D", "116A", "123D"]

# Processing .tif files
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for tifs in os.listdir(folder_path):
            if tifs.endswith('azi.tif'):
                if tifs.startswith('20'):
                    print(f'Processing {tifs} file')
                    pair = tifs[:17]
                    t_type = tifs[-11:-4]  # Extract from the 11th character from the end to the 5th character from the end
                    print(f'{pair}.{t_type}.downs.txt')
                    # Construct the command
                    command = f'gmt grdtrack -G{tifs} {downsample_point} > {pair}.{t_type}.downs.txt'

                    # Change to the folder directory, run the command, then change back
                    try:
                        os.chdir(folder_path)
                        subprocess.run(command, shell=True, check=True)
                    finally:
                        os.chdir(homedir)

            if tifs.endswith('azi.E.tif') or tifs.endswith('azi.N.tif') or tifs.endswith('azi.U.tif'):
                 print(f'Processing {tifs} file')
                 track = tifs[:4]
                 LoSv = tifs[-9:-4]

                 print(f'{track}.{LoSv}.txt')
                 command = f'gmt grdtrack -G{tifs} {downsample_point} > {track}.downs.{LoSv}.txt'

                 # Change to the folder directory, run the command, then change back
                 try:
                     os.chdir(folder_path)
                     subprocess.run(command, shell=True, check=True)
                 finally:
                     os.chdir(homedir)

print('Datasets downsampled! So let’s merge los with ENU vectors')
# Merging los with ENU vectors
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for ENU in os.listdir(folder_path):
            if ENU.endswith('azi.E.txt') or ENU.endswith('azi.N.txt') or ENU.endswith('azi.U.txt'):
                track = ENU[:4]
                command = f"paste {track}.downs.azi.E.txt {track}.downs.azi.N.txt {track}.downs.azi.U.txt | awk '{{print $1, $2, $3, $6, $9}}' > {track}.downs.azi.ENU.txt"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

print('ENU.txt created. Let’s create InSAR input of slipBERI')


# Creating InSAR input of slipBERI
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        # First, find the ENU file
        ENU_file = None
        for file in os.listdir(folder_path):
            if file.endswith('.downs.azi.ENU.txt'):
                ENU_file = file
                print(f"Found ENU file: {ENU_file}")
                break  # Exit the loop once the ENU file is found

        # Ensure we have found an ENU file before proceeding
        if not ENU_file:
            print(f"No ENU file found in {folder_path}")
            continue

        # Process rng.downs.txt files using the found ENU file
        for downs in os.listdir(folder_path):
            if downs.endswith('azi.downs.txt'):
                print(f"Processing rng file: {downs}")
                id = downs[:25]
                command = f"paste {downs} {ENU_file} | awk '$3 != \"NaN\" && $6 != 0 {{print $1, $2, ($3*-1), $6, $7, $8}}' > {id}.inp"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)


############################################################################
######################
downsample_point='down_for_boi_upd'
# List of specific folder names to check
#specific_folders = ["014A", "021D", "116A", "123D"]

# Processing .tif files
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for tifs in os.listdir(folder_path):
            if tifs.endswith('_boi.tif'):
                t_type = tifs[0:-4]  # Extract from the 11th character from the end to the 5th character from the end
                print(f'{t_type}.downs.txt')
                # Construct the command
                command = f'gmt grdtrack -G{tifs} {downsample_point} > {t_type}.downs.txt'

                # Change to the folder directory, run the command, then change back
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

            if tifs.endswith('azi.E.tif') or tifs.endswith('azi.N.tif') or tifs.endswith('azi.U.tif'):
                 print(f'Processing {tifs} file')
                 track = tifs[:4]
                 LoSv = tifs[-5]

                 print(f'{track}.{LoSv}.txt')
                 command = f'gmt grdtrack -G{tifs} {downsample_point} > {track}.downs.{LoSv}.boi.txt'

                 # Change to the folder directory, run the command, then change back
                 try:
                     os.chdir(folder_path)
                     subprocess.run(command, shell=True, check=True)
                 finally:
                     os.chdir(homedir)

print('Datasets downsampled! So let’s merge los with ENU vectors')
# Merging los with ENU vectors
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for ENU in os.listdir(folder_path):
            if ENU.endswith('E.boi.txt') or ENU.endswith('N.boi.txt') or ENU.endswith('U.boi.txt'):
                track = ENU[:4]
                command = f"paste {track}.downs.E.boi.txt {track}.downs.N.boi.txt {track}.downs.U.boi.txt | awk '{{print $1, $2, $3, $6, $9}}' > {track}.downs.boi.ENU.txt"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

print('ENU.txt created. Let’s create InSAR input of slipBERI')


# Creating InSAR input of slipBERI
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        # First, find the ENU file
        ENU_file = None
        for file in os.listdir(folder_path):
            if file.endswith('.downs.boi.ENU.txt'):
                ENU_file = file
                print(f"Found ENU file: {ENU_file}")
                break  # Exit the loop once the ENU file is found

        # Ensure we have found an ENU file before proceeding
        if not ENU_file:
            print(f"No ENU file found in {folder_path}")
            continue

        # Process rng.downs.txt files using the found ENU file
        for downs in os.listdir(folder_path):
            if downs.endswith('boi.downs.txt'):
                print(f"Processing rng file: {downs}")
                id = downs[:8]
                command = f"paste {downs} {ENU_file} | awk '$3 != \"NaN\" && $6 != 0 {{print $1, $2, ($3*-1), $6, $7, $8}}' > {id}.inp"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

#Creating .point files from .inp files
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for inp_file in os.listdir(folder_path):
            if inp_file.endswith('.inp'):
                point_file = inp_file.replace('.inp', '.point')
                command = f"awk '{{print 1}}' {inp_file} > {point_file}"
                try:
                    os.chdir(folder_path)
                    subprocess.run(command, shell=True, check=True)
                finally:
                    os.chdir(homedir)

print('.point files created from .inp files.')



