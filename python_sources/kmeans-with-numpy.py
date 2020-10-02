#!/usr/bin/env python
# coding: utf-8

# #### So we are importing NumPy and Matplotlib (for illustrative purpose)

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# #### Let set random seed so all of us see the same thing

# In[ ]:


np.random.seed(24)


# #### Let generate some data (Gaussian distribution as always)

# In[ ]:


points = np.vstack(((np.random.randn(100, 2) * 0.5 + np.array([1, 1])),
                  (np.random.randn(100, 2) * 0.5 + np.array([3, 2])),
                  (np.random.randn(100, 2) * 0.5 + np.array([1, 3]))))


# #### And plot them

# In[ ]:


plt.scatter(points[:, 0], points[:, 1]);


# # Step 1. We initialize some random centroids
# 
# I will randomly shuffle the points then take k first points. So we have ours random centroids.

# In[ ]:


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


# #### Let's take a look at our random centroids

# In[ ]:


plt.scatter(points[:, 0], points[:, 1]);
centroids = initialize_centroids(points, 3)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*');


# Ok. So that means we have some work to do.

# # Step 2. Calculate the L2 distance of each point to the centroids. Find out what is its nearest centroid.

# In[ ]:


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


# *Can you notice the np.newaxis thing? Try to explain it ;) If you can't, don't worry. I will explain it to you later ;)*|

# In[ ]:


point_c = closest_centroid(points, centroids)


# In[ ]:


plt.scatter(points[:, 0], points[:, 1], c=point_c);
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*');


# # Step 3. Move the centroids to the middle of points having it as the nearest centroid.

# In[ ]:


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


# *Another 1 liner code. Try to explain it ;) Yoy might want to take a look at advanced indexing with NumPy.*

# #### Let try moving the centroids once

# In[ ]:


print('Old centroids:\n', centroids)
centroids = move_centroids(points, point_c, centroids)
print('New centroids:\n', centroids)


# In[ ]:


point_c = closest_centroid(points, centroids)
plt.scatter(points[:, 0], points[:, 1], c=point_c);
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*');


# # And repeat step 3 till some requirements are met. Profit.
# 
# And that's why we will have a lectures on loss functions and tolerances ;)

# #### Ok. Summary time.
# 
# Now we will plot what we have when we initialize centroids, when we first move them, and when repeat some moves.

# In[ ]:


MOVES = 5
K = 3
centroids = initialize_centroids(points, K)
for m in range(MOVES):
    closest = closest_centroid(points, centroids)
    centroids = move_centroids(points, closest, centroids)
    plt.scatter(points[:, 0], points[:, 1], c=closest)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*')
    plt.show()
    print(centroids)


# In[ ]:




