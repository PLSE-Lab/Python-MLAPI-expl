#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering Vectorized Implementation

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# # Import Data

# In[8]:


img_1 = cv2.imread('../input/k-means-data-1/k-means-1.jpeg')
img_2 = cv2.imread('../input/k-means-data-1/k-means-2.png')


# # Plot Original Images

# In[9]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_1)
plt.title('image 1')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_2)
plt.title('image 2')
plt.axis('off')
plt.show()


# In[10]:


def choose_random(K, vec):
    m = len(vec)
    idx = np.random.randint(0, m, K)
    return vec[idx]

# calculate distortion cost function of clustering
def distortion(mu, c, vec):
    return ((mu[c] - vec) ** 2).sum() / vec.shape[0]

# distance between any two points
def distance(x_1, x_2):
    return np.matmul((x_1-x_2), (x_1-x_2).transpose())

# cluster assigment by using advance numpy broadcasting 
def cluster_assignment(mu, vec):
    return ((vec - mu[:, np.newaxis]) ** 2).sum(axis=2).argmin(axis=0)

# centroid calculations
def move_centroid(mu, c, vec):
    for i in range(len(mu)):
        vec_sub = vec[c==i]
        mu[i] = np.mean(vec_sub, axis=0)
    return mu

# the algorithm
def k_means(img, K, plot=True, verbose=False):
    l, w, ch = img.shape
    vec_img = img.reshape(-1, ch).astype(int)
    mu = choose_random(K, vec_img)
    c = cluster_assignment(mu, vec_img)
    last_dist = distortion(mu, c, vec_img) + 100
    curr_dist = last_dist - 100
    history = [curr_dist]
    # stop the iterations when the change is distortion is less than 1
    while last_dist - curr_dist > 1:
        last_dist = curr_dist
        c = cluster_assignment(mu, vec_img)
        if verbose:
            print(curr_dist)
        mu = move_centroid(mu, c, vec_img)    
        curr_dist = distortion(mu, c, vec_img)
        history.append(curr_dist)
    if plot:
        img_compressed = mu[c].reshape(img.shape)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('original image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_compressed.astype(np.uint8))
        plt.title('compressed image')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.plot(range(len(history)), history)
        plt.title('distortion over iterations')
        plt.xlabel('iterations')
        plt.ylabel('distortion')
    return mu, c, history[-1]


# # Image Compression Using K-Means for Pixel Quantization

# In[11]:


mu, c, dist = k_means(img_1, 3)


# In[12]:


mu, c, dist = k_means(img_2, 5)


# # Optimal number of Clusters using Elbow Method

# In[13]:


def elbow(img):
    K_hist = []
    dist_hist = []
    for K in range(1, 10):
        K_hist.append(K)
        mu, c, dist = k_means(img, K, plot=False)
        dist_hist.append(dist)
    plt.plot(K_hist, dist_hist)
    plt.xlabel("K")
    plt.ylabel("final distortion")


# In[14]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('elbow plot of image 1')
elbow(img_1)
plt.subplot(1, 2, 2)
elbow(img_2)
plt.title('elbow plot of image 2')
plt.show()


# # P.S. For detailed post on k-means clustering please see the [post](https://machinelearningmedium.com/2018/04/19/k-means-clustering/)
# ## Any feedback for improvement of the algorithm is welcome
# ### Multiple initialization for preventing local optima is not done here but can be implemented by just adding a outer loop on the k-means and tracking the corresponding distortions to choose the best one.

# In[ ]:




