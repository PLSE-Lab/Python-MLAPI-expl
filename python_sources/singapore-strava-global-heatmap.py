#!/usr/bin/env python
# coding: utf-8

# My own notebook for analyzing (transport and human) traffic data in Singapore
# 
# If you're interested to collab please contact me at etheleon@protonmail.com
# https://mothership.sg/2018/01/strava-singapore/

# In[ ]:


import os
#import hdf5

import pandas as pd
import numpy as np
import pickle

from PIL import Image
from io import BytesIO

import urllib
import requests
import matplotlib.pyplot as plt


# We are looking to see if we could combined multiple datasets.
# 
# 1. Uber 
# 2. Real time taxi (https://data.gov.sg/dataset/taxi-availability)
# 3. Strava, which recently came into the news. 
# 
# ![heatmap](https://mothership.sg/wp-content/uploads/2018/01/strava-heatmap-singapore.jpg)
# https://mothership.sg/2018/01/strava-singapore/

# Based on the php code from [HN](
# https://heatmap-external-c.strava.com/tiles/all/hot/15/9651/12318m)
# 
# 
# >HERE IS HOW TO DOWNLOAD IT
# I just spent a few minutes figuring it out.
# First grab some coordinates, I picked a totally random spot in NY: https://labs.strava.com/heatmap/#16.11/-73.96162/40.73006/ho...
# Now feed the GPS coords into the algorithm at https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
# I can use PHP faster than anything else, so I used the PHP example on that page.
# 
# ```php
#   $zoom = 15;
#   $lon = -73.96162;
#   $lat = 40.73006;
# 
#   $xtile = floor((($lon + 180) / 360) * pow(2, $zoom));
#   $ytile = floor((1 - log(tan(deg2rad($lat)) + 1 / cos(deg2rad($lat))) / pi()) /2 * pow(2, $zoom));
# 
#   print "x: $xtile\n";
#   print "y: $ytile\n";
# ```
#   x: 9651
#   y: 12318

# What's this formula that they posted? 
# 
# ```python
# xtile = int(np.floor(((lon + 180) / 360) * 2**zoom))
# ytile = int(np.floor( (1 - np.log(np.tan(np.deg2rad(lat)) + 1 / np.cos(np.deg2rad(lat))) / np.pi) / 2 * 2**zoom))
# ```
# 
# Turns out to be a long/lat to tile formula and is described by Open Street Maps (OSM) in particular [Slippy Maps](https://wiki.openstreetmap.org/wiki/Slippy_Map)
# 
# __Slippy Map__ is, in general, a term referring to modern web maps which let you zoom and pan around (the map slips around when you drag the mouse).

# In[ ]:


def downloadTile( lat=1.31214, lon =  103.97219, zoom = 11, verbose=False):
    xtile = int(np.floor(((lon + 180) / 360) * 2**zoom))
    ytile = int(np.floor( (1 - np.log(np.tan(np.deg2rad(lat)) + 1 / np.cos(np.deg2rad(lat))) / np.pi) / 2 * 2**zoom))
    if verbose: 
        print(f'x: {xtile}')
        print(f'y: {ytile}')
    url = f'https://heatmap-external-c.strava.com/tiles/all/hot/{zoom}/{xtile}/{ytile}.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img



# In[ ]:


file = open("../input/images.pkl", "rb")
img = pickle.load(file)
images = pickle.load(file)
file.close()


# In[ ]:


#img = downloadTile(zoom = 11, lon= 103.97219, lat=1.31214)
img
#images = [downloadTile(lat) for lat in [1.31214+i*0.1 for i in range(-3, 3, 1)]]


# In[ ]:


w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 2
for i in range(1, columns*rows +1):
    img = images[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

