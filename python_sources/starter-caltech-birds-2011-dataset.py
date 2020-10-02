#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis
# Just giving you the very basic idea about dataset. You are welcome to modify this kernel to more in depth analysis of dataset. 

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.patches as patches

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Directories

# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


get_ipython().system('ls ../input/CUB_200_2011/')


# ## CSV data

# In[ ]:


df = pd.read_csv('../input/image_data.csv')
df.head()


# ## Preview images and Bounding Boxes
# **Bounding Box**
# 
# ![download%20%283%29.jpg](attachment:download%20%283%29.jpg)

# In[ ]:


rows = 5
cols = 10
size = 2
f = plt.figure(figsize=(cols*size,rows*size))
img_paths = df['path'].values
# np.random.shuffle(img_paths)
path = "../input/CUB_200_2011/images/"
for i, img_path in enumerate(img_paths):
    img = cv2.cvtColor(cv2.imread(path+img_path), cv2.COLOR_BGR2RGB)
    
    ax = f.add_subplot(rows, cols, i+1)
    data = df[df['path'] == img_path]
    x, y = data['xmin'].values, data['ymin'].values
    w, h = data['width'].values, data['height'].values
    
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    
    plt.imshow(img)
    plt.axis('off')
    if i == rows*cols -1:
        break
    


# ## Segmentaion Mask
# **Path:**
#     All mask is in segmentaion folder and the directory structure is same as for the images. But the file extention is `PNG`. So you have to change the extention form `JPEG` to `PNG`. 

# In[ ]:


rows = 5
cols = 10
size = 2
f = plt.figure(figsize=(cols*size,rows*size))
img_paths = df['path'].values
path = "../input/segmentations/"
for i, img_path in enumerate(img_paths):
    img = cv2.imread(path+img_path[:-3]+"png")
    
    ax = f.add_subplot(rows, cols, i+1)
    
    plt.imshow(img)
    plt.axis('off')
    if i == rows*cols -1:
        break


# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the "Fork" button at the top of the kernel and delete, modify or add more analysis to provide better understanding of dataset.
