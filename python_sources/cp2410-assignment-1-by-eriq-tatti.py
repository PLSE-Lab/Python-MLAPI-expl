#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_cities = pd.read_csv('../input/cities10/cities10.csv')


# For the purpose of this project, to solve the Travelling Santa Problem, I will make use of the insertion sort algorithm found in the "Data Structures and Algorithms in Python" textbook in chapter 5, on page 215 (237 on the pdf). 
# 
# I will also be making use of the ["Nearest Neighbour"](https://www.kaggle.com/thexyzt/xyzt-s-visualizations-and-various-tsp-solvers) sorting algorithm developed by "XYZT" for the travelling santa competition.
# 
# Furthermore I will also be making use of two functions found in the ["Understanding the problem and some sample paths"](https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths) kernel for the Travelling santa competition, to create a list of prime numbers and to calculate the length of a path, so that I can have a measurable metric

# In[ ]:


fig = plt.figure(figsize=(14,10))

plt.scatter(df_cities['X'],df_cities['Y'],marker = '.',c=(df_cities.CityId != 0).astype(int), cmap='Set1', alpha = 0.6, s = 500*(df_cities.CityId == 0).astype(int)+1)
plt.title("All cities (10%)")
plt.xlabel('X', fontsize=10)
plt.ylabel('Y', fontsize=10)
plt.show()


# This ishow the dataset for the Travelling Santa competition appears when visualised as a scatter plot, with the red dot representing the north pole, the cities are arranged in a reindeer pattern. Note, this is only 10% of the original data set.

# In[ ]:


df_path = pd.DataFrame(df_cities)
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path['X'], df_path['Y'])


# This is graph shows the path between cities using the original order of the dataset

# In[ ]:


def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)
prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))

def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

dumbest_path = list(df_cities.CityId[:].append(pd.Series([0])))
print('Total distance with path in original order is '+ "{:,}".format(total_distance(df_cities,dumbest_path)))


# The two functions above have been taken from the ["Understanding the problem and some sample paths"](https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths) kernel for use in this project. 
# 
# The first function the ["Sieve of Eratosthenes"](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) is a simple, ancient algorithm for finding all prime numbers up to any given limit. One of the parameters for the travelling santa competition is that if the chosen path does not originate from a prime city exactly every 10th step, it takes 10% longer. Therefore having a list of prime numbers within the limit is essential for calculating the length of a path.
# 
# The second function calculates the euclidean distance of a path, so that there is a measurable metric. for example the length of the path of the unsorted dataset is 44,304,675.05710135 units.

# **Algorithm 1: Insertion Sort**
# 
# This algorithm is found in the "Data Structures and Algorithms in Python" textbook in chapter 5, on page 215 (237 on the pdf). The example in the text book is used to sort an array to find the highest score for a scoreboard, however for convenience I have decided to convert a modified dataset(ID column has been moved to the third position, and X and Y column moved to first and second position) into a list of lists, rather than an array, as it was more convenient to use a list of lists rather than an array for the Travelling santa dataset.
# 
# The Insertion sort algorithm is what is known as a bubble sort algorithm. The way a bubble sort algorithm works is by comparing the first 2 elements of a sequence and switching the elements if the second element is smaller, so that the smaller of the two elements come before the larger. The algorithm then moves up a step comparing the second and third element of the sequence switching the elements if the second element is smaller, so that the smaller of the two elements come before the larger. The algorithm moves up the sequence until it reaches the end, it then starts at the beginning and repeats the process until the entirety of the list has been arranged in an ascending order. The image bellow is an example of a how a bubble sort algorithm works.
# 
# ![This figure is an example of a bubble sort algorithm](https://cdn-images-1.medium.com/max/800/1*LllBj5cbV91URiuzAB-xzw.gif)

# In[ ]:


dataFile = open('../input/cities10modified/cities10modified.csv', "r")

np = [316.8367391, 2202.340707 , 0]
np_mod = [1, 316.8367391, 2202.340707 , 0]

xy = []
for line in dataFile:
    cordinates = line.strip().split(",")
    xy.append(cordinates)

xy.remove(xy[0])  
xy.remove(xy[0])

for element in xy:
    element[0] = float(element[0])
    element[1] = float(element[1])
    element[2] = int(element[2])
    
xy.append(np_mod)
        
for k in range(1, len(xy)):         # from 1 to n-1
    cur = a=xy[k]                       # current element to be inserted
    j = k                           # find correct index j for current
    while j > 0 and xy[j-1] > cur:    # element A[j-1] must be after current
      xy[j] = xy[j-1]
      j -= 1
    xy[j] = cur                # cur is now in the right place
    
xy[0].remove(1)
xy.append(np)

#for line in xy:
#    print(line)


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random 
cities_xy = pd.DataFrame(xy, columns = ["X", "Y", "ID"])
dumbest_path = list(cities_xy.ID[:].append(pd.Series([0])))
print('Total distance with path sorted by X coordinate value is '+ "{:,}".format(total_distance(df_cities,dumbest_path)))


# In[ ]:


fig, ax = plt.subplots(figsize=(40,40))
ax.plot(cities_xy['X'], cities_xy['Y'])


# What is shown by the path created with the insertion sort algorithm, is that when sorting a list of lists, the algorithm will sort by the first element of each list, in the case of the Travelling Santa dataset the algorithm sorted the data by the X value of the cities coordinates (This is the reason for why I modified the dataset). For the purposes of the competition this is not a very effecient way to travel, as while it reduces the overall distance, the path jumps from extremes of the Y coordinate. The length of the path created with the Insertion sort algorithm is 19,606,395.403784152, which is 24698279.6533 units less than the unsorted path, so while the algorithm has limitations in that it sorts by the first element, it is still effective in that it cut the travelling distance in half.
#          

# **Algorithm 2: Nearest Neighbour **
# 
# I found this algorithm in ["XYZT's Visualizations and Various TSP Solvers"](https://www.kaggle.com/thexyzt/xyzt-s-visualizations-and-various-tsp-solvers). In XYZT's project and in the case of this project, the algorithm sorts a numpy array (the X and Y values), which is a a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The way this algorithm works is by finding the closest neighbour of the previous city, and then finding the closest neighbour of the previous cities closest neighbour. For the purpose of the Travelling Santa problem, regardless of distance the path begins and ends with the coordinates for the north pole. 
# 
# The only forseeable limitation of the nearest neighbour algorithm is that it does not take into account the prime city rule.

# In[ ]:


def nearest_neighbour():
    cities = pd.read_csv("../input/cities10/cities10.csv")
    ids = cities.CityId.values[1:]
    coordinates = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((coordinates - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        coordinates = np.delete(coordinates, nearest_index, axis=0)
    path.append(0)
    return path

nnpath = nearest_neighbour()

print('Total distance with path sorted using Nearest Neighbour algorithm '+  "is {:,}".format(total_distance(df_cities,nnpath)))


# In[ ]:


df_path = pd.DataFrame({'CityId':nnpath}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path['X'], df_path['Y'])


# What is shown by the path created by the Nerest Neighbour algorithm is that the path is very efficient. The path created is 464,127.81601672183 units long, which is 43840547.2411 units less than the unordered path, and 19142267.5878 units less than the path created by the insertion sort algorithm. The nearest Neighbour algorithm is the most efficient algorithm that has been eshibted in this project, however as the algorithm does not take into account the prime cities rule, it is very likely that there is an even shorter path for santa to take on christmas eve. 
#  
