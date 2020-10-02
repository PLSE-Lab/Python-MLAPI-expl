#!/usr/bin/env python
# coding: utf-8

# # Cleaning the Data
# To start working with this dataset, we first flatten the data for each image into a single long continuous linear vector before adding it into a dataframe.

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np
from PIL import Image
import io
import os


# In[2]:


os.listdir("../input/wheres-waldo/Hey-Waldo/")


# Let's do an example with the 64 x 64 RGB images.
# 
# First, read each image as a numpy array.

# In[5]:


wdir = "../input/wheres-waldo/Hey-Waldo/64/"
waldos = np.array([np.array(imread(wdir + "waldo/"+fname)) for fname in os.listdir(wdir + "waldo")])
notwaldos = np.array([np.array(imread(wdir + "notwaldo/"+fname)) for fname in os.listdir(wdir + "notwaldo")])


# Let's take a look at one of our data points.

# In[7]:


plt.imshow(waldos[1])


# There's Waldo!
# Now to clean our data for regression or classification, we first flatten the arrays of images and then put the flattened arrays into a dataframe.

# In[8]:


data = []
for im in waldos:
    data.append(im.flatten('F'))


# Create a column called `waldo` and label the data accordingly.

# In[9]:


df1 = pd.DataFrame(data)
df1['waldo'] = 1


# Now we do the same for the notwaldos.

# In[10]:


data = []
for im in notwaldos:
    data.append(im.flatten('F'))


# In[11]:


df2 = pd.DataFrame(data)
df2['waldo'] = 0


# Now that we have labeled the `waldo` column correctly, we can concatenate the two dataframes into one, and save the result to a csv.

# In[13]:


frames = [df1, df2]
allwaldos = pd.concat(frames)
allwaldos.to_csv('all_waldo64.csv',index=False)


# And, we're done. The resulting csv can then be easily used for learning.

# As a final note, if you want to check whether the original array can be retrieved from the flattened array, we can do the following.

# In[17]:


d = df1.iloc[1].drop('waldo').values.astype('uint8').reshape(3, 64, 64).transpose().reshape(64,64, 3)
plt.imshow(d)


# And checking this data point with our original file, we see that this holds as well.

# In[18]:


(waldos[1] == d).all()

