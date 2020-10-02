#!/usr/bin/env python
# coding: utf-8

#  <img src="https://lh3.googleusercontent.com/-tNe1vwwd_w4/VZ_m9E44C7I/AAAAAAAAABM/5yqhpSyYcCUzwHi-ti13MwovCb_AUD_zgCJkCGAYYCw/w256-h86-n-no/Submarineering.png">

# **Visualize and animate MRI images using open source software.**

# In[23]:


# Import libs
import time
import numpy as np
from skimage import io
import matplotlib.animation as animation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


# Read the data and transpose the matrix
vol = io.imread("../input/attention-mri.tif")
volume = vol.T


# In[25]:


# number of frames
l = np.linspace(0, volume.shape[0], 69)


# In[26]:


#Ploting the animation
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):    
    ax1.clear()
    ax1.imshow(volume[i])
    ax1.axis('off')
ani = animation.FuncAnimation(fig,animate,interval=50, frames=68, repeat=False)


# In[7]:


# save to gif
ani.save('ani.gif', writer='imagemagick')


# **To see the animation go to the output tab**
