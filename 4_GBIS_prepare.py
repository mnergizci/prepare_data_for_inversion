#!/usr/bin/env python3

import os
import shutil
import numpy as np


def m2rad_s1(inm):
    speed_of_light = 299792458 #m/s
    radar_freq = 5.405e9  #for S1
    wavelength = speed_of_light/radar_freq #meter
    coef_r2m = -wavelength/4/np.pi #*1000 #rad -> mm, positive is -LOS
    outrad = inm/coef_r2m
    return outrad

###################################

def process_data(input_file, output_file, mode='azi'):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    track=os.path.basename(os.path.dirname(input_file))
    with open(output_file, 'w') as f_out:
        for line in lines:
            cols = line.strip().split()
            col2 = float(cols[1])
            col1 = float(cols[0])
            col3 = float(cols[2])
            col6 = float(cols[5])
            col5 = float(cols[4])
            
            #changing the meter to phase regarding 0.055(C-band) :)
            col3 = m2rad_s1(col3)

            incidence_rad = np.arccos(col6)
            if mode == 'azi':
                if track[-1] == 'A':
                    heading_rad = np.arcsin(col5 / np.sin(incidence_rad))
                    heading_deg = np.degrees(heading_rad)
                    f_out.write(f"{col1:.6f} {col2:.6f} {col3:.6f} {np.degrees(incidence_rad):.6f} {heading_deg:.6f}\n")
                elif track[-1] == 'D':
                    heading_rad = np.arcsin(-col5 / np.sin(incidence_rad))
                    heading_deg = np.degrees(heading_rad) * -1
                    f_out.write(f"{col1:.6f} {col2:.6f} {col3:.6f} {np.degrees(incidence_rad):.6f} {heading_deg:.6f}\n")

            elif mode == 'rng':
                if track[-1] == 'A':
                    heading_rad = np.arcsin(col5 / np.sin(incidence_rad))
                    heading_deg = np.degrees(heading_rad)
                    f_out.write(f"{col1:.6f} {col2:.6f} {col3:.6f} {np.degrees(incidence_rad):.6f} {heading_deg:.6f}\n")
                elif track[-1] == 'D':
                    heading_rad = np.arcsin(-col5 / np.sin(incidence_rad)) - np.pi
                    heading_deg = np.degrees(heading_rad)
                    f_out.write(f"{col1:.6f} {col2:.6f} {col3:.6f} {np.degrees(incidence_rad):.6f} {heading_deg:.6f}\n")

homedir = os.getcwd()
specific_folders = ["014A", "021D", "116A", "123D"]

# Processing .inp files for azimuth (azi) mode
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for inps in os.listdir(folder_path):
            if inps.endswith('azi.inp') or inps.endswith('boi.inp'):
                print(f'Processing {inps} file')
                input_file = os.path.join(folder_path, inps)
                output_file = input_file.replace('inp', 'gbis')
                process_data(input_file, output_file, mode='azi')
                
                # Debugging statements
                print(f'Created output file: {output_file}')
                
# Processing .inp files for range (rng) mode
for folder in specific_folders:
    folder_path = os.path.join(homedir, folder)
    if os.path.isdir(folder_path):
        for inps in os.listdir(folder_path):
            if inps.endswith('rng.inp'):
                print(f'Processing {inps} file')
                input_file = os.path.join(folder_path, inps)
                output_file = input_file.replace('inp', 'gbis')
                process_data(input_file, output_file, mode='rng')
                
                # Debugging statements
                print(f'Created output file: {output_file}')
                

