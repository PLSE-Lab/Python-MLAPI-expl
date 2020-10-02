#!/usr/bin/env python
# coding: utf-8

# In[33]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math # Math functions
import matplotlib.pyplot as plt # Plotting for graphs
import seaborn as sns
from time import time
from matplotlib import collections  as mc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[34]:


# retrieve dataset
df1 = pd.read_csv('../input/cities.csv')
df1.head(12)


# In[ ]:


#Visualisation of data:
fig = plt.figure(figsize=(15,15))
plt.scatter(df1['X'],df1['Y'],marker = '.',c=(df1.CityId != 0).astype(int), cmap='Set1', alpha = 0.6, s = 500*(df1.CityId == 0).astype(int)+1)
plt.show()


# In[ ]:


# we only need 10% of the dataset. Therefore:
df2 = pd.read_csv('../input/cities.csv', nrows = math.floor(len(df1)*0.1))
print(len(df2))
df2.head(10)


# In[ ]:


# this is a visualisation of what we have left over:
plt.figure(figsize=(15, 15))
plt.scatter(df2.X, df2.Y, s=1)
plt.scatter(df2.iloc[0: 1, 1], df2.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.show()


# In[ ]:


# The first way i am going to run this dataset is a very basic algorithm which will run them in order of their Id, I'll call this the 'worst case'.
# 'Worst case' will be used to compare to the first algorithm.



def distanceMeasure(x1, x2, y1, y2):
    distance_traveled = math.sqrt(((y2-y1)**2) + ((x2-x1)**2))
    return distance_traveled

def in_index_order(df2):
    n=0
    p=1
    dist = 0
    step_num = 1
    for n in range(len(df2)):
        x1 = df2.X.iloc[n]
        y1 = df2.Y.iloc[n]
        x2 = df2.X.iloc[p]
        y2 = df2.Y.iloc[p]
        dist = dist + distanceMeasure(x1,x2, y1, y2)
        n = p
        p+1
    return dist  
print(in_index_order(df2))



# In[ ]:


# Now we know that the 'worst_case' distance is roughly, we can start to implement better algorithms for distance.
# Firstly, we want to rearrange the dataset so that the algorithm traverses through the how area evenly, without jumping from one side to another. We'll
# generate a list and check if the poisition is a prime number:





# https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths
    


def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)] 
    primes[0] = False 
    primes[1] = False 
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)
prime_cities = sieve_of_eratosthenes(max(df2.CityId))

def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    
    for city_num in path[1:]:
        if stepNumber % 1000 == 0: #We print the progress of the algorithm
            print(f"Time elapsed : {time() - t0} - Number of cities left : {left_cities.size}")
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance



# In[ ]:


# sorting the cities into nearest neighbour by X
   # when sorting by Y, it takes almost twice as long, this is most likely due to the start and end point being closest to the X axis



unsorted = list(df2.iloc[1:,]['CityId'])
unsorted = [0] + unsorted + [0]
print('Total distance with the unsorted city path is '+ "{:,}".format(total_distance(df2,unsorted)))

sorted_cities = list(df2.iloc[1:,].sort_values(['X'])['CityId'])
sorted_cities = [0] + sorted_cities + [0]
print('Total distance with the sorted city by '"X"' path is '+ "{:,}".format(total_distance(df2,sorted_cities)))

sorted_citiesY = list(df2.iloc[1:,].sort_values(['Y'])['CityId'])
sorted_citiesY = [0] + sorted_citiesY + [0]
print('Total distance with the sorted city by '"Y"' path is '+ "{:,}".format(total_distance(df2,sorted_citiesY)))



# In[ ]:


# First 1800 plot points using sorted cities if X

df_path = pd.DataFrame({'CityId':sorted_cities}).merge(df2,how = 'left')
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path.iloc[0:1800,]['X'], df_path.iloc[0:1800,]['Y'],marker = 'o')


# **Nearest Neighbour: **
# algorithm/data structure 2
# 
# https://www.kaggle.com/theoviel/greedy-reindeer-starter-code

# In[ ]:


# Nearest Neighbour
#once again we need an algorithm to get the prime numbers
def sieve_eratosthenes(n):
    primes = [False, False] + [True for i in range(n-1)]
    p = 2
    while (p * p <= n):
        if (primes[p] == True):
            for i in range(p * 2, n + 1, p):
                primes[i] = False
        p += 1
    return primes


# In[ ]:



# get the primes, set as a bool
primes = np.array(sieve_eratosthenes(len(df2)-1)).astype(int)
df2['Prime'] = primes

# calculate the penalization for the reindeer at when they dont stop at major cities
penalization = 0.1 * (1 - primes) + 1

df2.head()


# In[ ]:




# This algorithm calculates the distance from the current

def dist_matrix(coords, i, penalize=False):
    begin = np.array([df2.X[i], df2.Y[i]])[:, np.newaxis]
    mat =  coords - begin
    if penalize:
        return np.linalg.norm(mat, ord=2, axis=0) * penalization
    else:
        return np.linalg.norm(mat, ord=2, axis=0)

    
    
def get_next_city(dist, avail):
    return avail[np.argmin(dist[avail])]


# In[ ]:



# This time we'll use an array to store the data for easier access : 
coordinates = np.array([df2.X, df2.Y])
current_city = 0 
left_cities = np.array(df2.CityId)[1:]
path = [0]
stepNumber = 1

t0 = time()

while left_cities.size > 0:
    if stepNumber % 1000 == 0: #We print the progress of the algorithm
        print(f"Time elapsed : {time() - t0} - Number of cities left : {left_cities.size}")
    # If we are at the ninth iteration (modulo 10), we may want to go to a prime city. Note that there is an approximation here: we penalize the path to the 10th city insted of 11th
    favorize_prime = (stepNumber % 10 == 9)
    distances = dist_matrix(coordinates, current_city, penalize=favorize_prime) # checks the boolean to see if the current_city is a prime number or not
    current_city = get_next_city(distances, left_cities)
    left_cities = np.setdiff1d(left_cities, np.array([current_city]))
    path.append(current_city)
    stepNumber += 1
    
path.append(0)

# getting the total distance traveled compared to the previous attempt
print("Total distance traveled "+ "{:,}".format(sum(distances)))


# In[ ]:


def plot_path(path, coordinates):
    # Plot tour
    lines = [[coordinates[: ,path[i-1]], coordinates[:, path[i]]] for i in range(1, len(path)-1)]
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_aspect('equal')
    plt.grid(False)
    ax.add_collection(lc)
    ax.autoscale()
    
plot_path(path, coordinates)

