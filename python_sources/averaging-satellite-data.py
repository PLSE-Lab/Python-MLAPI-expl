#!/usr/bin/env python
# coding: utf-8

# # Averaging Satellite Data
# 
# Satellite data doesn't have simple, consistent measures of locations to work with. Due to the variability in the location of the satellite and various other conditions, the exact locations vary. This makes getting a measurement for a specific point more difficult.
# 
# In this notebook we'll explore the Sentinel-5P dataset measurement locations to see if we can make an average.

# In[ ]:


import os

import rasterio as rio
from rasterio.plot import show, show_hist
import matplotlib.pyplot as plt
from matplotlib import animation

from datetime import datetime

import numpy as np
import pandas as pd


# ## Read Data
# 
# We'll work with the Sentinel-5P dataset.
# 
# Prior work: https://www.kaggle.com/jyesawtellrickson/understanding-sentinel-5p-offl-no2

# In[ ]:


def load_sp5():
    # Get the data filenames
    no2_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'
    no2_files = [no2_path + f for f in os.listdir(no2_path)]
    
    data = []
    
    print(f'Reading {len(no2_files)} files')
    for f in no2_files:
        raster = rio.open(f)
        data += [{
            'tif': raster,
            'filename': f.split('/')[-1],
            'measurement': 'no2',
            **raster.meta
        }]
        raster.close()
        
    # Get dates
    for d in data:
        d.update({'datetime': datetime.strptime(d['filename'][:23], 's5p_no2_%Y%m%dT%H%M%S')})

    for d in data:
        d['date'] = d['datetime'].date()
        d['hour'] = datetime.strftime(d['datetime'], '%H')
        d['weekday'] = d['datetime'].weekday()  # Mon = 0

    return data

data = load_sp5()


# In[ ]:


data[0]


# In[ ]:


df = pd.DataFrame(data)


# ## Affine Transformations
# 
# The measurements from satellite are recorded with two important location information. These are:
# 
# - CRS: The coordinate reference system that the measurements are referenced to. This is the world situation.
# - Transform: The affine transformation that used to get from the row/column indexes stored to x/y values in the coordinate reference system.
# 
# With the above two pieces of information we can take the values stored in the TIFF file and transform them to values in an x/y plane in our given coordinate reference system. In this case the CRS is EPSG:4326 for all the readings.
# 
# Further reading:
# - Affine Transformations: https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
# - EPSG:4326: https://epsg.io/4326
# 
# Let's start by investigating the affine transformations being used.

# In[ ]:


# Get the affine transformation values
aff = pd.DataFrame(df['transform'].values.tolist())
aff.head()


# In[ ]:


for col in aff.columns:
    print(aff[col].value_counts(), '\n')


# The letters a through i refer to the values in the affine transformation matrix:
# 
# - a/e are responsible for scaling
# - b/d are responsible for shearing
# - c/f are responsible for translations
# - g/h/i are set to 0, 0 and 1 respectively
# 
# Here we see that the scaling in both directions is the same, 0.004492. This number also represents the width of one of our readings, the pixel width. We have each reading equal to 0.004492 degrees, roughly 500m. The shearing is 0. The translations are varied, but are roughly -67.325 in the x and 17.901 in the y. The variance seen is no more than 0.002, which is less than a half of the pixel width.
# 
# So we see that in most cases are coordinate systems are aligned, but not all. When they do vary, it's quite a small variance, which means as a first attempt, taking a straight average should give a decent result. To be more accurate, we could ignore the readings that show 0.002 variance in the x direction as they may cause some more sizable overlap and their quantity is small. 
# 
# Reading:
# - https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
# - https://en.wikipedia.org/wiki/Decimal_degrees
# - http://www.csgnetwork.com/degreelenllavcalc.html

# In[ ]:


df = pd.merge(df, aff, left_index=True, right_index=True, how='inner')


# ## Creating the Average
# 
# Let's try taking the average and plotting. We can compare using the values that have varying translations or the ones without. We can also compare taking mean vs. median.

# In[ ]:


def plot_average_raster(rasters, band=1, output_file='tmp.tif', avg='mean'):
    all_no2s = []
    print(f'Processing {len(rasters)} files')
    for r in rasters:
        if r.closed:
            r = rio.open(r.name)
        all_no2s += [r.read()[band-1, :, :]]
        r.close()
    temporal_no2 = np.stack(all_no2s)
    
    if avg == 'mean':
        avg_no2 = np.nanmean(temporal_no2, axis=(0))
    else:
        avg_no2 = np.nanmedian(temporal_no2, axis=(0))

    raster = rasters[0]
    
    new_dataset = rio.open(
        output_file,
        'w',
        driver=raster.driver,
        height=raster.height,
        width=raster.width,
        count=1,
        dtype=avg_no2.dtype,
        crs=raster.crs,
        transform=raster.transform,
    )
    
    new_dataset.write(avg_no2, 1)
    new_dataset.close()
    
    tmp = rio.open(output_file)
    
    print('Ranges from {:.2E} to {:.2E}'.format(np.nanmin(tmp.read(1)),
                                                np.nanmax(tmp.read(1))))
    
    # https://rasterio.readthedocs.io/en/latest/topics/plotting.html
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

    show(tmp, transform=tmp.transform, ax=ax1)
    
    show((tmp, 1), cmap='Greys_r', interpolation='none', ax=ax2)
    show((tmp, 1), contour=True, ax=ax2)

    plt.show()
    
    return tmp


# In[ ]:


print('All files, mean', '\n')

tmp = plot_average_raster(df.tif.tolist(), avg='mean')


# In[ ]:


print('No translation files, mean', '\n')

tmp = plot_average_raster(df.query('f == 17.901140352016327 and c == -67.32431391288841').tif.tolist(),
                          avg='mean')


# In[ ]:


print('All files', '\n')

tmp = plot_average_raster(df.tif.tolist(), avg='median')


# There's very little difference made by removing the translated files, so it should be fine to proceed as is. Similarly, mean and median show little differences in the results, though the boundaries in the contour plot seem to be less smooth.
# 
# 
# Also, for some reason our plots are upside down, but we can definitely see strong effects here. Namely, the capital of Puerto Rico, San Juan shows the peak emissions with those emissions also going out to the left strongly (maybe this is wind, or another city). We also see some other isolated areas of high / low concentrations towards the centre-east of the island.
# 
# ## Conclusions
# 
# While averages are great because they allow us to get a quick picture of what is going on, they smooth out all the interesting effects that be required to capture a better idea about the emissions effects of power plants.
# 
# Next up we should look deeper into:
#  - wind strength
#  - day of the week
#  - time of day
#  - special events
