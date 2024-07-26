#!/usr/bin/env python3

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import numpy as np
from scipy.ndimage import median_filter
from scipy.constants import speed_of_light
from scipy.signal import convolve2d, medfilt
from scipy.ndimage import sobel
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
import dask.array as da
from scipy.ndimage import generic_filter
import scipy.ndimage as ndfilters
from scipy.interpolate import interp1d
import os

maindir=os.getcwd()
spec_dir=['014A', '116A']

# Initialize an empty DataFrame for df_014A and df_014A_rng
df_014A = pd.DataFrame()
df_014A_rng = pd.DataFrame()
filtered_df_116A = pd.DataFrame()
filtered_df_116A_rng = pd.DataFrame()
for dir in spec_dir:
    for i in os.listdir(dir):
    if 
