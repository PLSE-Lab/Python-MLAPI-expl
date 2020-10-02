#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Import dataset
spotify_dataset = pd.read_csv('../input/top2018.csv')
spotify_dataset.head(5)


# In[ ]:


# print dataset shape
spotify_dataset.shape


# In[ ]:


# Lets check the artists we have here in top 100 tracks and the number of their songs made it to the top 100
plt.figure(figsize=(18,12))
spotify_dataset['artists'].value_counts().plot.bar()

As we can see Post Malone and XXXTENTACTION's 6 tracks eacch have made it to the top 100 hits of this year on Spotify. Whereas Eminem's just 1 song is in the top 100 (may be it will be title track of 'Venom', we will see). 

But, number of tracks wonn't gaurantee you Grammy! May be Childish Gambino, whose not even 1 song could make to the top 100s list, could tell you more about it :P  
# In[ ]:


# Lets see the correlation between features in our dataset
plt.figure(figsize=(15,10))
sns.heatmap(spotify_dataset.corr(), 
            xticklabels=spotify_dataset.corr().columns.values,
            yticklabels=spotify_dataset.corr().columns.values)

The correlation heatmap describes few characterisitcs of this dataset. Energy and loudness of the tracks are directly proportional, means if the loudness of a track increases then chances of it being energetic are quite higher. On contrary, a high pich track reducess the acousticness (A confidence measure of whether the track is acoustic.)
# In[ ]:


# Plot a scatter mattrix
sns.set(style="whitegrid")
sns.pairplot(spotify_dataset)

Scatterplot mattrix is just confirming what we observed in the earlier step.
# ### In the next update we will dive deep into the dataset and see why certain tracks and artists performed well on Spotify in year 2018.

# In[ ]:




