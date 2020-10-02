#!/usr/bin/env python
# coding: utf-8

# In[5]:


import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[6]:


d = skimage.io.imread('../input/3d_scenes/img00003.png')
img = skimage.io.imread('../input/3d_scenes/img00003.tiff')


# In[7]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].text(50, 100, 'original image', fontsize=16, bbox={'facecolor': 'white', 'pad': 6})
ax[0].imshow(img)

ax[1].text(50, 100, 'depth map', fontsize=16, bbox={'facecolor': 'white', 'pad': 6})
ax[1].imshow(d)


# In[8]:


d = np.flipud(d)
img = np.flipud(img)


# In[44]:


get_ipython().run_cell_magic('time', '', "fig = plt.figure(figsize=(15,10))\nax = plt.axes(projection='3d')\n\nSTEP = 5\nfor x in range(0, img.shape[0], STEP):\n    for y in range(0, img.shape[1], STEP):\n        ax.scatter(\n            d[x,y], y, x,\n            c=[tuple(img[x, y, :3]/255)], s=3)      \n    ax.view_init(15, 165)")


# In[47]:


ax.view_init(30, 135)
fig


# In[50]:


ax.view_init(45, 220)
fig


# In[54]:


ax.view_init(5, 100)
fig


# In[ ]:




