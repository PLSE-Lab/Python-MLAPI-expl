#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Dominant colors in images
# - All images consist of pixels
# - Each pixel has three values: Red,Green and Blue
# - Pixel color : Combination of these RGB values
# - Perform k-means on standardized RGB values to find cluster centers
# - Uses: Indentifying features in satellite images

# ## Importing Iibraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Load Image

# In[ ]:


img = "../input/img_1.jpg"
image = plt.imread(img)
plt.imshow(image)


# In[ ]:


# Check for the image shape
image.shape


# ## Convert image to RGB matrix

# In[ ]:


r = []
g = []
b = []

for row in image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)


# ## Visualization (3d)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


# Plot the figure in 3D graph
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(r,g,b)
plt.show()


# ## Data frame for RGB values

# In[ ]:


pixels = pd.DataFrame({
    'red':r,
    'blue':b,
    'green':g
})

pixels.head()


# In[ ]:


pixels.shape


# ## Importing Libarires for Kmeans and Scipy

# In[ ]:


# for kmeans 
from scipy.cluster.vq import kmeans , vq

# for data normalization
from scipy.cluster.vq import whiten


# In[ ]:


# Normalize the data

# for this we use whiten function from scipy.cluster.vq


pixels['scaled_red'] = whiten(pixels['red'])
pixels['scaled_blue'] = whiten(pixels['blue'])
pixels['scaled_green'] = whiten(pixels['green'])


# In[ ]:


pixels.head()


# ## Create an Elbow Plot

# In[ ]:


distortions = []
num_cluster = range(1,11)

# create a list of distortion from the kmeans method
for i in num_cluster:
    cluster_centers , distortion = kmeans(pixels[['scaled_red',
                                                'scaled_blue',
                                                'scaled_green']], i)
    distortions.append(distortion)
    

# create a data frame with two lists - number of clusters and distortions

elbow_plot = pd.DataFrame({
    'num_clusters':num_cluster,
    'distortions':distortions
})

# create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters',y='distortions',data=elbow_plot)
plt.xticks(num_cluster)
plt.show()


# ## By looking at the graph we can say the no of cluster in around 2 or 3

# In[ ]:


# So can take no_of_cluster = 3
cluster_center , _ = kmeans(pixels[['scaled_red',
                            'scaled_blue','scaled_green']], 3)


# In[ ]:


cluster_center


# In[ ]:


# find Standard Deviations
r_std, b_std, g_std = pixels[['red','blue','green']].std()


# In[ ]:


colors = []

# scaled actual RGB values in range of 0-1
for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    colors.append((
        scaled_r * r_std/255,
        scaled_g * g_std/255,
        scaled_b * b_std/255
  ))


# ## Display Dominant Colors

# In[ ]:


#Dimensions:(N X 3 matrix)
print(colors)


# In[ ]:


#Dimensions: 1 x 2 x 3 (1 X N x 3 matrix)
display(plt.imshow(image))
plt.show()

# we have to pass the list in plt.imshow() function
display(plt.imshow([colors]))
plt.show()


# In[ ]:




