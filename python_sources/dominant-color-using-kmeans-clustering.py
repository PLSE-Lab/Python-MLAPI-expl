#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import warnings
warnings.filterwarnings('ignore')


#     Images can have various extensions like JPG, PNG, TIFF. The first step in the process is to read the image. 
#     Image is stored as a list of dots called pixels. 
#     The color of each pixel is a combination of 3 color component(Red,Green,Blue).

#     In Python, the image can be read using the imread method from cv2 library. It can be plotted using imshow.

# In[ ]:


image = cv2.imread('/kaggle/input/kmeans/color.jfif')

plt.imshow(image);


#     This image has 300 rows, 300 columns with 3 RGB values.

# In[ ]:


image.shape


#     Let's just read the RGB values from the image. 
#     Loop over each line of the image, and again loop over each pixel to decode the r,g,b values
# 
#     Here the image pixels are stored in BGR format and hence take care while reading!
# 
#     The R,G,B is then stored into a dataframe and scaled using StandardScaler(with standard deviation set to True)
#     This scaled data is stored in r_scaled,g_scaled and b_scaled columns

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean = 0,with_std=1)


#     Read the pixel data into R,G and B and scale them

# In[ ]:


r = []
g = []
b = []

for line in image:
    for pixel in line:
        temp_b , temp_g,temp_r = pixel
        
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)
    
df = pd.DataFrame({'red': r,'green': g,'blue': b})

df['scaled_red'] = scaler.fit_transform(df[['red']])
df['scaled_green'] = scaler.fit_transform(df[['green']])
df['scaled_blue'] = scaler.fit_transform(df[['blue']])

df.head()


#     Let's just use the scaled RGB values for for KMeans

# In[ ]:


X = df[['scaled_red','scaled_green','scaled_blue']].values
X


#     How to determine the number of clusters to be used. With simple images, its easy to visually identify the 
#     number of clusters. But more the number of color/gradient in the image, the harder it is!
# 
#     Elbow method can used to determine the optimal value of k
# 
#     1: Choose the number of clusters k
#     2: Select k random points from the data as centroids
#     3: Assign all the points to the closest cluster centroid
#     4: Recompute the centroids of newly formed clusters
#     5: Repeat steps 3 and 4
# 
#     Stopping Criteria :
#     Centroids of newly formed clusters do not change
#     Points remain in the same cluster
#     Maximum number of iterations are reached

# In[ ]:


SSE = []

for cluster in range(2,8): 
    kmeans = KMeans(n_clusters=cluster,random_state=42)
    kmeans.fit(X)
    
    pred_clusters = kmeans.predict(X)
    SSE.append(kmeans.inertia_)
    
frame = pd.DataFrame({'Cluster':range(2,8) , 'SSE':SSE})
print(frame)


#     The sum of squared errors can be plotted to see the changes in SSE with respect to k. As k increases, 
#     the cluster will have less values and the average error will reduce. The lesser number of values in a
#     cluster means closer to the centroid. The point where the error almost declines, will be the optimal 
#     value for k and is the elbow point

# In[ ]:


plt.figure(figsize=(5,5))
plt.plot(frame['Cluster'],frame['SSE'],marker='o')
plt.title('Clusters Vs SSE')
plt.xlabel('No of Clusters')
plt.ylabel('Intertia')
plt.show()


#     From the above figure , we can see that k could range from 3 to 7. 
#     However 4 looks more optimal for this image

# In[ ]:


#Fit and predict for k = 4
k=4
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
k_pred = kmeans.predict(X)

#These are the centroids of the clusters
cluster_centers = kmeans.cluster_centers_
cluster_centers


#     First let's see what are the dominant colors in this image.
#     The results from the KMeans cluster_centers are standardized versions of RGB values. 
#     To get the original color values we need to multiply them with their standard deviations. 
#     (In our StandardScaler we used the standardization with standard deviation set to True). 
#     We can plot using imshow.

# In[ ]:


colors = []

r_std, g_std, b_std = df[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    
    colors.append((
    scaled_r * r_std /255,
    scaled_g * g_std / 255,
    scaled_b * b_std/ 255
    ))
    
plt.imshow([colors])
plt.show()


#     We do not see the exact colors as in the image. 
#     That's because the cluster centers are the means all of of the RGB values of all pixels in each cluster. 
#     Hence the ouptput cluster center may not appear in the same color as in the original image
#     It's only the RBG value that is at the center of the cluster of all similar looking pixels from our image.

#     Let's check the predicted image against original image    

# In[ ]:


res = cluster_centers[k_pred.flatten()]
result_image = res.reshape((image.shape))

im_bgr = result_image[:, :, [2, 1, 0]] #restoring the image in bgr form

rescale_image = scaler.inverse_transform(im_bgr).astype(int) #rescaling it back to original

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(image)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(rescale_image)
plt.title('Segmented Image when K = 4'), plt.xticks([]), plt.yticks([])
plt.show()


#     Kindly upvote my work and suggest improvements or corrections. This is my first attempt of KMeans with images
# 
