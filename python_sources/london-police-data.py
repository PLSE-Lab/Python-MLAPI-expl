#!/usr/bin/env python
# coding: utf-8

# Options

# In[ ]:


# Number to show out of 1947050
Total = 100
shuffle = True
live_plot = False


# Import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For graph/image plotting
if live_plot:
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.image as mpimg # For importing images
import os # For accessing data


# Show directory structure

# In[ ]:


print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/greater-london"))


# In[ ]:


print(os.listdir("../input/london-police-records"))


# Read CSV

# In[ ]:


data_raw = pd.read_csv("../input/london-police-records/london-outcomes.csv")
indices = np.arange(data_raw.shape[0])
if shuffle:
    np.random.shuffle(indices)
data_raw = data_raw.iloc[indices[:Total]]


# Show the head of the CSV

# In[ ]:


data_raw.head()


# We note that some rows(e.g. row 0) has no info on Longitude/Latitude. So we filter out these rows.

# In[ ]:


data = data_raw[data_raw.Longitude.notnull() & data_raw.Latitude.notnull()]
data.head()


# The map of Greater London below is a screenshot from google map with boundary box: -0.576352, 51.749359, 0.362419, 51.242601

# In[ ]:


london_map = mpimg.imread("../input/greater-london/greater-london-google.png")

# Longitude & Latitude coordinates to Pixel coordinates converter
def convert(lon, lat):
    # Boundary box coordinates
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = -0.576352, 51.749359, 0.362419, 51.242601
    
    # Map image's dimensions
    map_h, map_l = london_map.shape[0], london_map.shape[1]
    
    # Convert to pixel coordinates
    pix_x = (lon - bbox_x1) / (bbox_x2 - bbox_x1) * map_l
    pix_y = (lat - bbox_y1) / (bbox_y2 - bbox_y1) * map_h
    
    return pix_x, pix_y

fig = plt.figure(figsize=(15,15))
plt.imshow(london_map)
plt.axis("off")

for _, row in data.iterrows():
    circle = plt.Circle(convert(row["Longitude"], row["Latitude"]), 3, fc='r')
    plt.gca().add_patch(circle)
    
    if live_plot:
        fig.canvas.draw()

fig.canvas.draw()


# In[ ]:




