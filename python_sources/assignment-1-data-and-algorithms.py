#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.path import Path
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dfcities = pd.read_csv('../input/cities10.csv')
dfcities.head()


# In[ ]:


cities.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))
north_pole = dfcities[dfcities.CityId==0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.axis('off')
plt.show()


# To get the prime number we use a simple algorithm to check.

# In[ ]:


def is_prime(number):
    prime = False
    if number == 2:
        prime = True
    elif number > 1:
    # check for factors
        for i in range(2,np.ceil(np.sqrt(number)).astype(int) + 1):
            if (number % i) == 0:
                break
        else:
            prime = True
    return prime


# This gives us the prime cities since each 10 cities would be 10% more effective.

# In[ ]:


dfcities['prime'] = dfcities['CityId'].apply(is_prime)
prime_cities_count = np.sum(dfcities['prime'])


# In[ ]:


#Detailing cities
print("The North Pole is located at: {:,.2f}, {:,.2f}".format(dfcities.X[0],dfcities.Y[0]))
print("There are {:d} cities".format(len(dfcities.index) - 1)) 
print("There are {:d} prime cities".format(prime_cities_count))
print("There are {:d} that are not prime".format(len(dfcities.index) - 1 - prime_cities_count))


# Now we check the dumbest path going from city to city so we know that our code has to be better than this one.

# In[155]:


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

dumbest_path = list(dfcities.CityId[:].append(pd.Series([0])))
print('Total distance with the dumbest path is '+ "{:,}".format(total_distance(dfcities,dumbest_path)))


# now we can sort the cities.

# In[157]:


sorted_cities = list(dfcities.iloc[1:,].sort_values(['X','Y'])['CityId'])
sorted_cities = [0] + sorted_cities + [0]
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(dfcities,sorted_cities)))


# this markdown the trip by approximately 25 million.

# Wikipedia however suggest that the quickest way to approach this problem is the nearest neighbour solution.

# In[160]:


# Function from XYZT's Kernel on the same topic. 
def nearest_neighbour():
    cities = pd.read_csv("../input/cities10.csv")
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    return path

nnpath = nearest_neighbour()
print('Total distance with the Nearest Neighbor path '+  "is {:,}".format(total_distance(dfcities,nnpath)))


# and as we can see this is an improvement of the first and second results its approximatly 1% of doing the dumbest path therefore is an improvement of 99%
