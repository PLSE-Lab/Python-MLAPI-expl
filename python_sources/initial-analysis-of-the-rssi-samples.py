#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy as sc
sns.set()
import matplotlib
matplotlib.rcParams['figure.figsize'] = [12, 8]


# In[2]:


path = '../input/rssi.csv'
data = pd.read_csv(path)


# Plot RSSI values at different coordinates.  Naturally, the signal is strongest near the access point and as we go farther it drops.
# 
# 

# In[3]:


fig = plt.figure(figsize=(8, 20))
from itertools import product

axs = fig.subplots(4,2)
for pair, ax in zip(product((1,2), ("A","B","C","D")), axs.flatten()):
    (floor, ap) = pair
    mask = (data.z == floor) & (data.ap == ap)
    signal = data[mask][["signal", "x", "y"]]
    ax.scatter(signal.x, signal.y, c=signal.signal)
    ax.set_title("Floor: %s AP: %s" %(floor, ap))
    


# In[4]:


# Find the Euclidean distance of each sampling location to its respectve AP
ap_coordinates = {"A": (23, 17, 2), "B": (23, 41, 2), "C" : (1, 15, 2), "D": (1, 41, 2)}
g = data.groupby(["x", "y", "z", "ap"])
def dist(df):
    ap_coords = ap_coordinates[df.iloc[0].ap]
    x, y, z = ap_coords
    df["distance"] = np.sqrt((df.x - x) ** 2 + (df.y - y) ** 2 + (df.z - z) ** 2)
    return df
data = g.apply(dist)


# Let's plot the relation between distance to each accees point and RSSI. Remember that smaller values of RSSI mean stronger signals and correspond to smaller distances. Notice the substantial unstability of the RSSI at some locations, specially as we get farther from an access point.

# In[5]:


fig, axes = plt.subplots(4,2, figsize=(18, 16))
for pair, ax in zip(product((1,2), ("A","B","C","D")), axes.flatten()):
    (floor, ap) = pair
    mask = (data.z == floor) & (data.ap == ap)
    signal = data[mask][["signal", "distance"]]
    ax.plot(signal.distance, signal.signal, '.')
    ax.set_ylabel("RSSI")
    ax.set_title("Floor %s, AP: %s" %(floor, ap))


# This wild variabiliy of RSSI in almost all locations can be a hinderance to accurate location tracking. One natural question is, in the face of such unstability, can we find a way to reliably characrerize RSSI at a location? There are few obvious choices: min, max, mean and median. Let's experiment with them, for only one of the access points:

# In[14]:


fig, axes = plt.subplots(2,2, figsize=(18, 16))
estimators = [np.min, np.max, np.mean, np.median]
for ax, estimator in zip(axes.flatten(), estimators):
    mask = (data.z == 2) & (data.ap == "A")
    signal = data[mask][["signal", "distance"]]
    sns.regplot("distance", "signal", data=data, 
                x_estimator=estimator, x_bins=100, ax=ax)
    ax.set_title(estimator.__name__)


# Min and Max seem to be terrible regressors while mean and median, particularly near the access point seem to exhibit more sensible rlation to distance from the access point.

# In[ ]:




