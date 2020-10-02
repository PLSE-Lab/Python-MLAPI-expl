#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# I am using one of the images that i downloaded to extract the dominant colours.
# Well, We have to find dominant color in a given image. I will be using an unsupervised learning algorithm (K-Means Clustering).
# 
# 1. Read Image and convert it into low pixels.
# 2. Using Elbow method to find optimal number of clusters of given image.
# 3. Recreating model with optimal number of Cluster using sklearn KMeans
# 4. Comparing dominance of extracted colour in image.
# 5. In end i will regenerate image using these dominant colours (k centers).
# 
# You can read about Kmeans Clustering [here](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Preprocessing

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.cluster.vq import kmeans,vq
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read image and print dimensions
image = img.imread("/kaggle/input/image3/sunflower.jpg")
image2 = img.imread("/kaggle/input/butterfly/butterfly.jpg")

print('shape', image.shape)
r, c = image.shape[:2]
out_r = 500
new_image = cv2.resize(image, (int(out_r*float(c)/r), out_r))

pixels = new_image.reshape((-1, 3))

print('pixels shape :', pixels.shape)
print('New shape :', new_image.shape)

plt.figure(figsize=(14,10))
plt.axis("off")

plt.subplot(121)
plt.title('Actual Image')
plt.imshow(image)

plt.subplot(122)
plt.title('Image with decreased pixels.')
plt.imshow(new_image)
plt.show()


# **Let us store RGB values of all pixels in lists r, g and b.**

# In[ ]:


r,g,b=[],[],[]
for row in new_image:
    for r_val, g_val, b_val in row:
        r.append(r_val)
        g.append(g_val)
        b.append(b_val)


# ## Scaling
# 
# Let us scale the data using SciPY Library.
# You can read more on SciPy Whiten [Here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html)

# In[ ]:


# using scipy's inbuilt scaler whiten to scale.
scaled_red = whiten(r)
scaled_blue = whiten(b)
scaled_green = whiten(g)


# In[ ]:


import pandas as pd
df = pd.DataFrame({'red':r,'blue':b,'green':g,'scaled_red':scaled_red,'scaled_blue':scaled_blue,
                   'scaled_green':scaled_green})
df.head()


# # Optimizing K(number of clusters)

# 1. **Elbow Method**

# In[ ]:


distortions = []
num_clusters = range(1, 10)

for i in num_clusters:
    cluster_centers, distortion = kmeans(df[['scaled_red','scaled_blue','scaled_green']],i)
    distortions.append(distortion)


# In[ ]:


# Create a line plot of num_clusters and distortions
plt.plot(num_clusters, distortions)
plt.xticks(num_clusters)
plt.title('Elbow Plot', size=18)
plt.xlabel('Number of Clusters')
plt.ylabel("Distortions")
plt.show()


# **We can see from Elbow Plot that optimal value of k is 5.**

# Note - I am not able to implement silhouette method for optimizing n_clusters, so if anyone can help it is appreciated.

# # K-Means Clustering

# In[ ]:


# using sklearn's inbuilt kmean for clustering data and finding cluster centers i.e. means for clusters.
k_means= KMeans(n_clusters=5)
k_means.fit(pixels)
print(k_means.cluster_centers_)


# In[ ]:


colors = np.asarray(k_means.cluster_centers_, dtype='uint8')
print(colors)


# ## Displaying Dominant Colours

# In[ ]:


print("Original Image --->")
plt.axis('off')
plt.imshow(image)
plt.show()

print("Dominant",5,"Colours of Image --->")
plt.axis('off')
plt.imshow([colors])
plt.show()


# # Dominance of Colours Extracted

# In[ ]:


# percentage of each extracted colour in the image
pixels_colourwise = np.unique(k_means.labels_, return_counts=True)[1]
percentage = pixels_colourwise/pixels.shape[0]
percentage


# In[ ]:


colors


# In[ ]:


plt.title('Dominance Of Colours', size=16)
plt.bar(range(1,6), percentage, color=np.array(colors)/255)
plt.ylabel('Percentage')
plt.xlabel('Colours')
plt.show()


# **We can see in the bar plot the blue colour is most dominent in the image and it is actually right.**

# # Regenerating Image.

# In[ ]:


p=pixels.copy()
for px in range(pixels.shape[0]):
    for _ in range(colors.shape[0]):
        p[px]=colors[k_means.labels_[px]]


# In[ ]:


img = p.reshape(out_r, -1, 3)

plt.figure(figsize=(14,10))
plt.subplot(121)
plt.title('Original Image with Decreased Pixels')
plt.imshow(new_image)

plt.subplot(122)
plt.title('Regenerated Image using KMeans')
plt.imshow(img)
plt.show()


# * Thus we have generated pretty decent image from the clustered RGB values.

# #### Thank you for reading this notebook. I hope you like the notebook. ;-)
