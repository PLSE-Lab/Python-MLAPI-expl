# Read in the UCI Human Activity dataset
import numpy as np
import pandas as pd
import xarray as xr
import os

path = '/kaggle/input/UCI HAR Dataset/train/Inertial Signals'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(filename)

channels = {}
for filename in filenames:
    channels[filename] = pd.read_csv(os.path.join(path, filename), delimiter=r'\s+')
    
ntrials, nsamples = channels[filename].shape

xr_channels = {}
for filename in filenames:
   xr_channels[filename] =  xr.DataArray(channels[filename],dims=['n','t'],
        coords={
        'n': range(ntrials),
        't': range(nsamples)})

X = xr.Dataset(xr_channels)
#X.sel[n=1]


