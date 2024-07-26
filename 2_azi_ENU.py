import os
import glob
import numpy as np
from modules_qi import OpenTif, export_tif

def rng_azi_NEU_from_NEU(path):
    frame = os.path.basename(path)
    frame_base = os.path.splitext(frame)[0]
    print("Processing path:", path)
    print("Frame:", frame)
    
    # Ensure frame has enough characters
    if len(frame) < 4:
        print("Skipping invalid frame:", frame)
        return
    
    orientation = frame[3]
    track = frame[:4]
    print("Frame:", frame, "Orientation:", orientation, "Track:", track)
    
    n_file = os.path.join(os.path.dirname(path), "{}.geo.N.tif".format(frame_base))
    e_file = os.path.join(os.path.dirname(path), "{}.geo.E.tif".format(frame_base))
    u_file = os.path.join(os.path.dirname(path), "{}.geo.U.tif".format(frame_base))
    n = OpenTif(n_file)
    e = OpenTif(e_file)
    u = OpenTif(u_file)

    if orientation == "A":
        inc_rad = np.arccos(u.data)
        head_rad = np.arcsin(n.data / np.sin(inc_rad))
        heading = np.degrees(head_rad)
        incidence = np.degrees(inc_rad)
    elif orientation == "D":
        inc_rad = np.arccos(u.data)
        head_rad = np.arcsin(-n.data / np.sin(inc_rad)) - np.pi
        heading = np.degrees(head_rad)
        incidence = np.degrees(inc_rad)
    else:
        raise ValueError("The 4th character of frameID is neither A nor D, please check your frame name.")

    azi_N = np.cos(head_rad)
    azi_E = np.sin(head_rad)
    azi_U = np.zeros(head_rad.shape)
    
    # Create the output directory if it doesn't exist
    output_directory = os.path.dirname(path)
    os.makedirs(output_directory, exist_ok=True)
    
    export_tif(azi_N, n, os.path.join(output_directory, "{}_azi.N.tif".format(frame)))
    export_tif(azi_E, n, os.path.join(output_directory, "{}_azi.E.tif".format(frame)))
    export_tif(azi_U, n, os.path.join(output_directory, "{}_azi.U.tif".format(frame)))

if __name__ == "__main__":
    # List of specific folders to check
    spec_folders = ['014A', '021D','123D','116A']
    
    for folder in spec_folders:
        directory_path = os.path.join(os.getcwd(), folder)
        for root, dirs, files in os.walk(directory_path):
            # Skip 'optical' directories
            dirs[:] = [d for d in dirs if not d.endswith('optical')]
            
            # Process each file
            for file in files:
                if file.endswith('geo.E.tif'):
                    track=file[0:17]
                    file_path = os.path.join(root, track)
                    rng_azi_NEU_from_NEU(file_path)

