#!/usr/bin/env python
# coding: utf-8

# # Travelling Santa 2018 - Prime Paths
# 

# Travelling Santa 2018-Prime Paths is travelling salesman problem which given a list of cities and the distance between each pair of cities to find out the shorest possible route that visits each city and returns to the origin city.
# Overall, it goes to load dataset cities.csv, and take 10% sample data from the row dataset as sp_cities, then detemine the prime number.
# For the path and distance, one way to order with queue by CityIDs from 0 to the end and come back to 0. Another one sort the list cities in XY for coordinate order.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import queue 
import os
import time


# # Loading data - cities.csv

# In[ ]:


# Loading data - cities.csv

cities = pd.read_csv('cities.csv')
cities.head()


# In[ ]:


# Number of cities to visit in row data

print("Number of cities to visit are", len(cities))


# # Import 10% of sample data from row data

# In[ ]:


mini_cities = pd.read_csv ('cities.csv')
mini_cities = cities[0:int(len(cities)*0.1)]

CityId = [mini_cities.values[i][0] for i in range(len(mini_cities))]
X_value = [mini_cities.values[i][1] for i in range(len(mini_cities))]
Y_value = [mini_cities.values[i][2] for i in range(len(mini_cities))]

mini_cities.head()


# In[ ]:


print("Number of cities to visit in the sample data are", len(mini_cities))


# # scatter plot of locations of sample cities 

# In[ ]:


fig = plt.figure(figsize=(7,7))
plt.scatter(mini_cities['X'],
            mini_cities['Y'],
            marker = '.', # CityID=0
            c=(mini_cities.CityId != 0).astype(int), 
            cmap='Set1', alpha = 0.6, 
            s = 500*(mini_cities.CityId == 0).astype(int)+1)
plt.show()


# # Prime Cities
# 
# Determines if integers are prime numbers, use algrorithm to find with Boolean data type to identity the list of cities are prime cities or not.

# In[ ]:


# Determines if integer is prime number
def find_prime(n):
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


# In[ ]:


prime_cities = find_prime(max(mini_cities.CityId))


# # Calculate the distance between two cities

# In[ ]:


def pair_distance(x,y):
    x1 = (mini_cities.X[x] - mini_cities.X[y]) ** 2
    x2 = (mini_cities.Y[x] - mini_cities.Y[y]) ** 2
    return np.sqrt(x1 + x2)


# # Insertion sort cities with X value

# In[ ]:


#Distance between two cities
#function to calculate distance between two cities by using euclidean distance
def pair_distance(x,y):
    x1 = (mini_cities.X[x] - mini_cities.X[y]) ** 2
    x2 = (mini_cities.Y[x] - mini_cities.Y[y]) ** 2
    return np.sqrt(x1 + x2)


# In[ ]:


#Calculate total distance
def total_distance(path):
    distance = [pair_distance(path[x], path[x+1]) + 0.1 * pair_distance(path[x], path[x+1])
    if (x+1)%10 == 0 and mini_cities.is_prime[path[x]] == False else pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]
    return np.sum(distance)


# # Dumbest Path
# 
# Sorting the total distance base on the sequence of City_num.

# In[ ]:


# The Dumbest path of total distance

def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((mini_cities.X[city_num] - mini_cities.X[prev_city]),2) + pow((mini_cities.Y[city_num] - mini_cities.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

dumbest_path = list(mini_cities.CityId[:].append(pd.Series([0])))
print('Total distance with the dumbest path is '+ "{:,}".format(total_distance(mini_cities,dumbest_path)))


# # Dumbest Path plot

# In[ ]:


df_path = pd.merge_ordered(pd.DataFrame({'CityId':dumbest_path}),mini_cities,on=['CityId'])
fig, ax = plt.subplots(figsize=(7,7))
ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')
for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
    ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 15)


# Above is the first 100 steps of the dumbest path looks very bad, it is all over the map without consideration.

# # Sort the cities by insertion sort for  X and Y

# In[ ]:


sorted_cities = list(mini_cities.iloc[1:,].sort_values(['X','Y'])['CityId'])
sorted_cities = [0] + sorted_cities + [0]
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(mini_cities,sorted_cities)))


# # Plot sorted X and Y

# In[ ]:


df_path = pd.DataFrame({'CityId':sorted_cities}).merge(mini_cities,on=['CityId'])
fig, ax = plt.subplots(figsize=(15,15))
ax.set_xlim(0,100)
ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')
for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
    ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 15)
   


# From above can see the path is much more effcient, it goes up and down around the same X values, then moving to next X.         6 x

# # Sort the cities by selection sort with Y 

# In[ ]:


sorty_path=[]
for x in range(1,max(mini_cities.CityId)+1):sorty_path.append(x)
        
City_Y=[]
for x in range(max(mini_cities.CityId)+1):
    City_Y.append(mini_cities['Y'][x])


# In[ ]:


def selectionsort(alist):
    for i in range(len(alist)):minPosition = i
    for j in range(i+1, len(alist)):
        if City_Y[alist[minPosition]] > City_Y[alist[j]]: minPosition = j
            temp = alist[i]
    alist[i] = alist[minPosition]
                


# In[ ]:


selectionsort(sorty_path)


# In[ ]:


#create a path for calculating total distance
sortedy_path=[0]
for each in range(len(sorty_path)-1):
    sortedy_path.append(sorty_path[each])
sortedy_path.append(0)


# In[ ]:


print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(sortedy_path)))


# # Sort the cities by using nearest neighbor algorithm

# In[ ]:


# nearest neighbor algorithm
nnpath_with_primes = nnpath.copy()
for index in range(20,len(nnpath_with_primes)-30):
    city = nnpath_with_primes[index]
    if (prime_cities[city] &  ((index+1) % 10 != 0)):        
        for i in range(-1,3):
            tmp_path = nnpath_with_primes.copy()
            swap_index = (int((index+1)/10) + i)*10 - 1
            tmp_path[swap_index],tmp_path[index] = tmp_path[index],tmp_path[swap_index]
            if total_distance(df_cities,tmp_path[min(swap_index,index) - 1 : max(swap_index,index) + 2]) < total_distance(df_cities,nnpath_with_primes[min(swap_index,index) - 1 : max(swap_index,index) + 2]):
                nnpath_with_primes = tmp_path.copy() 
                break
print('Total distance with the Nearest Neighbor With Prime Swaps '+  "is {:,}".format(total_distance(df_cities,nnpath_with_primes)))


# # Sort the cities by using dictionary

# In[ ]:


def calDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

cities = []
for i in range(len(mini_cities)):
    edge={} # create a dictionary to contain all the edge which contain city as a key and distance as value
    for j in range(len(mini_cities)):
        edge[j] = calDistance(mini_cities.values[i][1],mini_cities.values[i][2],mini_cities.values[j][1],mini_cities.values[j][2])
    cities.append(edge)


# In[ ]:


visited = []# which will be contain the visited city and also for the path
cost = []# contain all the distance every step
position = 0
visited.append(position)
while len(visited) < len(cities):
    tempt = position # assign current position into a tempt variavle which will use for find the distance
    position = findMin(cities[position], visited)# find the city which is near to current city
    cost.append(cities[tempt][position])# add the distance into cost list
    visited.append(position)# add the visited city into list
# at the end add the zero city into the visited list to complete the path 
# and also calculate the distance from the last city to zero city and add it into cost list
visited.append(0)
cost.append(calDistance(sample.values[0][1],sample.values[0][2],sample.values[position][1],sample.values[position][2]))


# In[ ]:


distance = 0 
step = 0 
flag = False # which will let you know whether we met the end because there are 2 zero element in the visited list
for city in visited: # go through all the city in the list
    if city == 0 and flag == False: #start with the 0 city
        distance = distance + cost[step]*1.1
        step += 1
        flag = True # just for separate between 0 at beginning and 0 at the end
    elif city == 0 and flag == True: #end at the city 0
        break;
    elif step % 10 == 0 and isPrime(city) == False:#if there is a 10th step and not a prime city 
        distance = distance + cost[step]*1.1
        step += 1
    else:
        distance = distance + cost[step]
        step += 1


# In[ ]:


print("using list of dictionary structure to calculate the distance with penalized distance , distance = " + str(distance))


# In[ ]:


def submission():
    dict = {'Path': nnpath}
    df = pd.DataFram(dict)
    df.to_csv('Final.csv', index = False)


# In[ ]:


submission()


# To sum up, the most effciency path with distance of 462,595.62466666184.
