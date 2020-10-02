#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# This script helps visualize the attributes with 2 libraries: Matplotlib and Seaborn.
# 
# * Matplotlib

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from random import *


# Firstly drop the first column since indices are irrelevant with the following analysis.

# In[ ]:


dt=pd.read_csv('../input/data.csv')
dt.drop(dt.columns[0],axis=1,inplace=True)


# Here I plotted the histogram of 4 attributes: Danceability, Energy, Liveness and Acousticness. It turns out people are looking for danceability and energy, and the less acoustics and live the song is, the more popular it might be.

# In[ ]:


fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
fig.text(0.5, 0.04, 'Score', ha='center',size=20)
fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical',size=20)
axs[0].hist(dt['danceability'])
axs[0].set_title('Danceability')
axs[1].hist(dt['energy'])
axs[1].set_title('Energy')
axs[2].hist(dt['liveness'])
axs[2].set_title('Liveness')
axs[3].hist(dt['acousticness'])
axs[3].set_title('Acousticness')
fig.suptitle('What Makes Good Music Good?',size=20)
plt.show()


# People usually think that high BPM music is more upbeat, but suprisingly there is no correlation between those two objects.

# In[ ]:


fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
fig.text(0.5, 0.04, 'Tempo(BPM)', ha='center',size=20)
axs[0].scatter(dt['tempo'],dt['danceability'])
axs[0].set_title('Danceability')
axs[1].scatter(dt['tempo'],dt['energy'])
axs[1].set_title('Energy')
axs[2].scatter(dt['tempo'],dt['liveness'])
axs[2].set_title('Liveness')
axs[3].scatter(dt['tempo'],dt['acousticness'])
axs[3].set_title('Acousticness')
fig.suptitle('Higher BPM = More Upbeat?',size=20)
plt.show()


# * Seaborn
# 
# The heatmap down below shows the correlation between attributes and there is one strong linear relationship: loudness & valance.

# In[ ]:


# Heatmap
top10=dt.iloc[0:10]
top10=top10[['danceability','energy','liveness',
             'acousticness','loudness','speechiness',
             'valence','tempo','duration_ms']]
corr=top10.corr()
ax = plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
plt.title("Correlation of Song Attributes",size=15)
plt.show("Correlation of Song Attributes")


# This scatterplot say the same thing as the heatmap above.

# In[ ]:


# Subplot of scatterplots
dtall=dt[['danceability','energy','liveness',
             'acousticness','loudness','speechiness',
             'valence','tempo','duration_ms']]
ax1 = plt.figure()
sns.pairplot(dtall)
plt.title("Pairplot of Song Attributes",size=15)
plt.show("Pairplot of Song Attributes")

