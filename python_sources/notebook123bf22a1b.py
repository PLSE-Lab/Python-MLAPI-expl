#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# Exploring the data before trying anything out lets us examine how much we have and what its features are/might be. 

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df = pd.read_csv("../input/beacon_readings.csv")
df.head()


# In[ ]:


beacon_locs = {'A': np.array([0.10, 1.80]), 'B':  np.array([2.74, 2.66]), 
               'C': np.array([ 1.22, 4.54])}
w, l =  2.74, 4.38


# ### Get the actual distances
# 
# These are withing +/- 1cm

# In[ ]:


for b, loc in beacon_locs.items():
    dists = np.sqrt(np.sum((df[["Position X", "Position Y"]].as_matrix()/100 - loc)**2, axis=1))
    df[f'Actual {b}'] = dists


# In[ ]:


df.head()


# ### Extent of Bluetooth Error
# 
# First we show how the actual distance and the measured distance differ in a joint plot, then in histograms of the difference between measured and 'truth'.

# In[ ]:


for b in 'ABC':
    fig = plt.figure();
    g = sns.jointplot(f'Actual {b}', f'Distance {b}', df);
    g.ax_joint.set_xlim(-0.1, 3); g.ax_joint.set_ylim(-0.1, 3);


# In[ ]:


# Look at the distributions around the 'true' value
for b in 'ABC':
    fig = plt.figure();
    all_data = []
    for truth in df[f'Actual {b}'].unique():
        data = df[df[f'Actual {b}'] == truth][f'Distance {b}'].as_matrix() - truth
        g = sns.distplot(data, norm_hist=True, kde=False, label=f'{truth:.2f}');
        plt.xlabel(f"Distance {b}"); plt.legend();


# ### 2D Location Estimation
# 
# Using a point-cloud style approach, see what the distances from each bluetooth tell us about our nearness to the actual position, and what that distribution looks like

# In[ ]:





# In[ ]:





# In[ ]:




