#!/usr/bin/env python
# coding: utf-8

# # Ex 1
# *You can hand in the assignment in pairs.*
# In this exercise we will implement the k-means algorithm we learned in class.<br>
# We are going to preform the k-menas clustering algorithm on the supplied Mall segementation dataset.
# 
# Don't forget to run the cells.
# ### Just run the following cell no code needed.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib
matplotlib.rcParams['figure.figsize'] = [10, 10]

import matplotlib.pyplot as plt # For plotting the points.
from IPython import display # A convinient way of plotting 
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Loading the data
# We start off by loading the data into a pandas dataframe which we talked about in trecitation 2.<br>
# We use the head command to peek into the data
# 
# ### Just run the following cell no code needed.

# In[ ]:


df = pd.read_csv("../input/Mall_Customers.csv")
df.head()


# Next, we take the 2 relevant features for our clustering : **Annual Income** and __Spending Score__
# 
# ### Just run the following cell no code needed.

# In[ ]:


data = df[['Annual Income (k$)', 'Spending Score (1-100)']]
points = data.values # Now you have a 2d array(list) of (x,y) points.

# here is graph of our points.
data.plot(kind='scatter', x='Annual Income (k$)', y='Spending Score (1-100)')


# # Auxiliry Function
# This function is implemented for you and we will use it to plot the condition
# of the k-means algorithm through out it's iterations.
# 
# ### Just run the following cell no code needed.

# In[ ]:


def plot_points_by_centers(points, centers):
    display.clear_output(wait=True)
    plt.clf()
    colors = ['magenta', 'blue', 'pink', 'black', 'green']
    
    label = "Centeriods"
    for c in centers:
        plt.scatter(c[0], c[1], color='red', label=label, marker = 'X', s=300)
        label = "_nolegend_"

    for p in points:
        c = get_closest_centroids(p, centers)
        i = centers.index(c)
        plt.scatter(p[0], p[1], color=colors[i], alpha=1)

        time.sleep(0.001)
        
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    display.display(plt.gcf())
    
def compute_residual_error(points, centers):
    d_sum = 0
    for p in points:
        c = get_closest_centroids(p, centers)
        d_sum += get_distance(c, p)
        
    return d_sum


# ## Helper functions
# You first assignment is to implement the functions in the cell below, <br>
# This will make implementing the k-menas algorithm much simpler.

# In[ ]:


def get_distance(p1, p2):
    '''
    Get the eucleidan distance between 2 given points p1 and p2.
    '''
    d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1/2)
    return d
    pass
def get_closest_centroids(p, centroids):
    '''
    Given a point and a list of candidates points returns the closet candidate to the given points.
    Distance is calculated using the euclidean distance.
    
    Example : point=[2, 1] candidates = [[3,2], [5,1], [2,2]] return value should be [2,2]
    '''
    minn = centroids[0]
    for i in centroids:
        if  get_distance(p , i) < get_distance(p, minn):
            minn = i
    return minn
    pass

def get_random_points(k):
    '''
    Return a 2d array consisting of n random points.
    Each x and y value of each point should be sampled from numbers between 0 and 100.
    
    Example : if n==2 a possible return value is : [[1,3], [24,50]] or [[62,15], [1, 0]]
    '''
    import random
    list = []
    for i in range(k):
        x = random.randint(0,100)
        y = random.randint(0,100)
        list.append([x,y])
    return list
    pass

def compute_array_mean(arr):
    sum_x = 0
    sum_y = 0
    
    for i in range(len(arr)):
        sum_x += arr[i][0]
        sum_y += arr[i][1]
        
    return [sum_x / (i + 1) , sum_y / (i + 1)]
    '''
    Compute the x and y mean of a 2d array of points
    
    Example : if arr=[[1, 4], [2, 6], [3, 8]] return value is : 2, 6
    
    '''



# ## Test Methods
# Run the following cells to test your functions. <br>
# You don't need to implement anything, but if at any of the cells you run and get an error <br>
# this means your implementation is not good.

# In[ ]:


def test_get_distance():
    if (get_distance([0.,0.], [2.,0.]) != 2):
        print("p1 = [0,0], p2 = [2,0] and distance is not 2")
        return False
    
    if (get_distance([0, 0], [3,4]) != 5):
        print("p1 = [0,0], p2=[3,4] and distance is not 5")
        return False
    
    if (get_distance([0, 0], [6,8]) != 10):
        print("p1 = [0,0], p2=[6,8] and distance is not 10")
        return False
    return True
        
if test_get_distance():
    print ("Test get_distance Passed!")
else:
    print ("Test get_distance Failed!")


# In[ ]:


def test_get_closest_centroids():
    if get_closest_centroids([1,0], [[2,5], [3,0]]) != [3,0]:
        print("p = [1,0], centroids = [[2,5], [3,0]], closest point is not [3,0]")
        return False
    
    if get_closest_centroids([2,1], [[3,2], [5,1], [2,2]]) != [2,2]:
        print("p = [2,1], centroids = [[2,5], [3,0]], closest point is not [2,2]")
        return False
    
    if get_closest_centroids([1,4], [[2,5], [3,0]]) != [2,5]:
        print("p = [1,4], centroids = [[2,5], [3,0]], closest point is not [2,5]")
        return False
    
    if get_closest_centroids([1,4], [[2,5], [3,0], [1,4], [5,5], [6,100]]) != [1,4]:
        print("p = [1,4], centroids = [[2,5], [3,0], [1,4], [5,5], [6,100]], closest point is not [1,4]")
        return False
    
    return True

if test_get_closest_centroids():
    print ("Test get_closest_centroids Passed!")
else:
    print ("Test get_closest_centroids Failed!")


# In[ ]:


def test_compute_array_mean():
    if compute_array_mean([[1,2], [1,2], [1,2]]) != [1, 2]:
        print("arr = [[1,2], [1,2], [1,2]] but mean in not [1,2]")
        return False
    
    if compute_array_mean([[1,2], [2,4], [3,6]]) != [2, 4]:
        print("arr = [[1,2], [2,4], [1,6]] but mean in not [2, 4]")
        return False
    
    if compute_array_mean([[0,2], [0,5], [0,8]]) != [0, 5]:
        print("arr = [[0,2], [0,5], [0,8]] but mean in not [1, 5]")
        return False
    
    if compute_array_mean([[1,2], [2,2.5], [3,3.5], [4,4]]) != [2.5, 3]:
        print("arr = [[1,2], [2,2.5], [3,3.5], [4,4]] but mean in not [2.5, 4]")
        return False
    return True

if test_compute_array_mean():
    print ("Test compute_array_mean Passed!")
else:
    print ("Test compute_array_mean Failed!")


# ## K-means
# Implement the k-means algorithm in the next cell.<br>
# You have the signature of the function.<br>
# In class you talked about 2 stopping conditions:
# 1. No change in cluster centers
# 1. No change to cluster assignment.<br><br>
# For simplicity reasons we are going to use a simpler stopping condition : <br>
# We will run the algorithm for a fixed number of iterations. 

# In[ ]:


import random
def k_means(points, k, t):
    """
    K means algorithm
    Given a set of points, k and n_iterations finds the best k centroids for clustering the points.
    The algorithm runs for t iterations before coming to an halt.
    
    Returns the k best centroids(points) (those who minimize the overall distance.)
    """
    # Start with setting random points as your centers
    # --------------- Your Code Here -----------------
    centroids = get_random_points(k)
    # Iterate for n_iterations and update your centroids.
    for i in range(t):
        # Create empty clusters
        clusters = {i:[] for i in range(k)}
    
        # Allocate each point to the closest centroid
        # --------------- Your Code Here ------------
        for p in points:
            c = get_closest_centroids(p , centroids)
            idx = centroids.index(c)
            clusters[idx].append(p)
        # This function will plot the points and the clusters.
        plot_points_by_centers(points, centroids)

        
        # Compute the new means for each of the centroids
        # --------------- Your Code Here -----------------

    
    # Leave this for plotting purposes.
    display.clear_output(wait=True)
    return centroids
            
    


# ## Running the k-means algorithm and visualizing the results.
# Run the following cells and see if your implementation is good.<br>
# This will give you a better intution of how the algorithm works.<br>
# Notice that on each cell we run the k_means algorithm and then save the residual error in a variable.<br>
# At the end we will use this variables to visualize how the error decreases as k increases.

# ### K-means - with k=2

# In[ ]:


centers_2 = k_means(points, 2, 20)
r2 = compute_residual_error(points, centers_2)


# ### K-means with k=3

# In[ ]:


centers_3 = k_means(points, 3, 20)
r3 = compute_residual_error(points, centers_3)


# ### K-means with k=4

# In[ ]:


centers_4 = k_means(points, 4, 20)
r4 = compute_residual_error(points, centers_4)
    


# ### K-means with k=5

# In[ ]:


centers_5 = k_means(points, 5, 20)
r5 = compute_residual_error(points, centers_5)


# # Plot behaviour
# We will now plot the sum of residual error to see how it decrease as k increases

# In[ ]:


plt.plot(range(2, 6), [r2, r3, r4, r5])
plt.xlabel("K")
plt.ylabel("Residual Sum of Squares")


# # End of Exercise
# To submit the exercise, click *File* -> *Download Notebook*.<br>
# Submit the downloaded file in the moodle under ex1.
