#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


df_cities = pd.read_csv('../input/cities/cities.csv')

For the purpose of this project, to solve the Travelling Santa Problem, I will be making use of the inbuilt python library heapq to sor the data set.

I will also be making use of a merge sort algorithm found at the Problem Solving with Algorithms and Data Structures [The Merge Sort](https://interactivepython.org/runestone/static/pythonds/SortSearch/TheMergeSort.html?fbclid=IwAR1CT0qJtyXGvPcXq4gi7jxwMc2W6sLoOQIBzvbuWKML-3uPBvdJJsG2SJE#the-merge-sort)

Furthermore I will also be making use of two functions found in the ["Understanding the problem and some sample paths"](https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths) kernel for the Travelling santa competition, to create a list of prime numbers and to calculate the length of a path, so that I can have a measurable metric
# In[ ]:


fig = plt.figure(figsize=(14,10))

plt.scatter(df_cities['X'],df_cities['Y'],marker = '.',c=(df_cities.CityId != 0).astype(int), cmap='Set1', alpha = 0.6, s = 500*(df_cities.CityId == 0).astype(int)+1)
plt.title("All cities")
plt.xlabel('X', fontsize=10)
plt.ylabel('Y', fontsize=10)
plt.show()


# This Scatter plot depicts the travelling santa data set. The north pole is depicted as the red dot. The cities are arranged in a reindeer pattern.

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


# The two functions shown above have been taken from the ["Understanding the problem and some sample paths"](https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths) kernel for use in this project. 
# 
# The first function is the ["Sieve of Eratosthenes"](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) it is based off of an ancient algorithm developed by Eratosthenes. The algorithm is used to find all the prime numbers.
# 
# The second function "Total distance" calculates the euclidean distance between points on a path and combines them to get the total distance of the path, so that there is a metric that can be used to measure and compare. The total distance algorithm tells us that the distance of the usorted path is 446884407.5212135 units.

# **Algorithm 1:**
# 
# For the first algorithm I will be making use of the heap data structure which is a binary tree T that stores a collection of items at its positions and that satisfies two additional properties: a relational property defined in terms of the way keys are stored in T and a structural property defined in terms of the shape of T itself (Goodrich, Tamassia and Goldwasser, 2013). For establishing the heap, the inbuilt python library heapq will be utilized. The data will be sorted using the heapq methods heapify and nsmallet. Both methods make use of a heap sort algorithm. A heapsort algorithm is similar to a selection sort algorithm as both algorithms divide their input data into a sorted and unsorted region, and iteratively decreases the size of the unsorted region by finding the largest element in the unsorted region and moving it to the sorted region. The two algorithms differ in that the heap sort algorithm uses a heap data structure rather than a linear-time search to find the maximum element. This makes the heap sort algorithm much more efficient when it comes to sorting large datasets.
# 
# In regards to the effeciency and accuracy of the algorith, the algorithm is farily efficient when it comes to large data sets. As for accuracy, during test runs the algorithm itself accurately aranged the test data in ascending sequence, however with the way the dataset is loaded into the kernel and the way the algorithm works, it only sorts the data by the first element, so in the case of the dataset it will sort the data by the X coordinate value. This should make the path shorter, but will not make the path as short as it could be.

# In[ ]:


import heapq 

dataFile = open('../input/citiesmod/cities_Modified.csv', "r")

xy = []
for line in dataFile:
    cordinates = line.strip().split(",")
    xy.append(cordinates)

#remove element zero twice to get rid of column headings, and the modified northpole coordinates 
xy.remove(xy[0])  
xy.remove(xy[0])

for element in xy:
    element[0] = float(element[0])
    element[1] = float(element[1])
    element[2] = int(element[2])
    
#for element in xy:
#    print(element)

# using heapify() to convert list into heap
heapq.heapify(xy) 
  
# using heappush() to push north pole coordinates to start of xy 
heapq.heappush(xy,[0, 316.8367391, 2202.340707, 0]) 

xy_path = []
xy_path = heapq.nsmallest(1997768, xy)
xy_path[0].remove(0)
xy_path.append([316.8367391, 2202.340707, 0])
print(xy_path[0])
print(xy_path[-1])


# In[ ]:


cities_heappath = pd.DataFrame(xy_path, columns = ["X", "Y", "ID"])
sorted_path = list(cities_heappath.ID[:].append(pd.Series([0])))
print('Total distance with path sorted by X coordinate value is '+ "{:,}".format(total_distance(df_cities,sorted_path)))


# In[ ]:


df_path = pd.DataFrame({'CityId':sorted_path}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(20,20))
plt.title("HeapQ sorted path")
ax.plot(df_path['X'], df_path['Y'])


# What the graph above depicts is the path taken, as stated before the data is sorted by the X coordinate value which is why it appears the way it does. The distance of the path sorted via the heapq data structure and the heapsort algorithm is calculated to be 196,473,894.84053066 which is 250,410,512.6806829 units less than the original path wich was 446,884,407.5212135 units, so the sorted path is less than half of the original unsorted path. As stated before the heap sort algorithm does not take into account the Y coordinate value, so there is a shorter path to be taken. 

# **Algorithm 2:**
# 
# The second algorithm that will be used to sort the data structure is a merge sort algorithm. A merge sort algorithm is a type of divide and conquer algorthm. It works by dividing an unsorted list into sublists until the sublists consist of only a single element, it then begins merging the sublists in ascending order until it has a fully merged and sorted list. 
# 
# In regards to the efficiency and accuracy of the list, merge sort algorithms are fairly efficient when it comes to large datsets. One thing to note is that as previously stated, due to the way the dataset is set out in the kernel, the algorithm will sort the data by the X coordinate value, therefore unless there is an error in the code for the merge sort algorithm, the path and length of the path made with the merge sort algorithm will be the same as the path and length of path made with the heapq data structue and heapsort algorithm.

# In[ ]:


def mergeSort(alist):
    #print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
            
    return alist
    #print("Merging ",alist)
    
dataFile = open('../input/citiesmod/cities_Modified.csv', "r")

cities_xy = []
for line in dataFile:
    cordinates = line.strip().split(",")
    cities_xy.append(cordinates)

#remove element zero twice to get rid of column headings, and the modified northpole coordinates 
cities_xy.remove(cities_xy[0])  
cities_xy.remove(cities_xy[0])

for element in cities_xy:
    element[0] = float(element[0])
    element[1] = float(element[1])
    element[2] = int(element[2])
    
cities_xy.append([0, 316.8367391, 2202.340707, 0])

sorted_xy = []

sorted_xy = mergeSort(cities_xy)

sorted_xy[0].remove(0)
sorted_xy.append([316.8367391, 2202.340707, 0])


# In[ ]:


print(sorted_xy[0])
print(sorted_xy[-1])


# In[ ]:


cities_mergepath = pd.DataFrame(sorted_xy, columns = ["X", "Y", "ID"])
mergesorted_path = list(cities_heappath.ID[:].append(pd.Series([0])))
print('Total distance with path sorted by X coordinate value is '+ "{:,}".format(total_distance(df_cities,mergesorted_path)))


# In[ ]:


merge_path = pd.DataFrame({'CityId':mergesorted_path}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(20,20))
plt.title("Merge sorted path")
ax.plot(df_path['X'], df_path['Y'])


# This scatter plot depcits the path taken by the merge sort algorithm. As stated above it is identical to path created with the heapsort algorithm. The length of the path created is 196,473,894.84053066 units which has cut the distance of the original path down by more than half.

# **Conclusion**
# 
# In conclusion, both the heap sort and merge sort algorithms are effective and efficient sorting algorithms, and were both able to cut the original travel distance down by more than half. However for the path to be even shorter both algorithms would have to consider the Y coordinate value as well as the X coordinate value to calculate a shorter path for santa to take on Christmas eve. 

# **References**
# 
# Goodrich, M., Tamassia, R. and Goldwasser, M. (2013). Data Structures and Algorithms in Python. John Wiley & Sons, p.370.
