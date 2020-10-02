#!/usr/bin/env python
# coding: utf-8

# *Assignment I Travelling Reindeer*
# ==================
# 
# 

# Setting up python and csv reader
# 

# In[6]:


import numpy
import pandas as pandas
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import os
from time import time
import math

df_cities = pandas.read_csv('../input/cities/cities.csv')
df_cities.head()


# Number of total cities

# In[ ]:


num_cities = max(df_cities.CityId)
print(num_cities, " cities to visit")


# In[ ]:


rows = len(df_cities.index) -1
print("The total number of cities Santa has to visit is:", rows)


# Let's plot all of the cities on a scatterplot
# 

# In[ ]:


fig = plt.figure(figsize=(15,10))
#cmap, norm = from_levels_and_colors([0.0, 0.5, 1.5], ['red', 'black'])
plt.scatter(df_cities['X'],df_cities['Y'],marker = '.',c=(df_cities.CityId != 0).astype(int), 
            cmap='Set1', alpha = 0.6, s = 500*(df_cities.CityId == 0).astype(int)+1)
plt.show()


# Interesting figure, however there are too many cities, let's only use 10% of this dataset

# In[ ]:


df_cities_10 = pandas.read_csv('cities.csv', nrows = math.floor(len(df_cities)*0.1))
print("There are ", len(df_cities_10), " in this new dataset")
df_cities_10.head()


# Plotting new dataset

# In[ ]:


fig = plt.figure(figsize=(15,10))
#cmap, norm = from_levels_and_colors([0.0, 0.5, 1.5], ['red', 'black'])
plt.scatter(df_cities_10['X'],df_cities_10['Y'],marker = '.',c=(df_cities_10.CityId != 0).astype(int), 
            cmap='Set1', alpha = 0.6, s = 500*(df_cities_10.CityId == 0).astype(int)+1)
plt.show()


# Prime function to determine prime cities

# In[ ]:


def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)] # all prime numbers
    primes[0] = False
    primes[1] = False
    for i in range(2,int(numpy.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)

start = time()
prime_cities = sieve_of_eratosthenes(max(df_cities_10.CityId))
end = time()
elapsed = end - start
print("Time elapsed: ", elapsed, "\n")
print(sieve_of_eratosthenes(max(df_cities_10.CityId)))


# how many prime cities are there

# In[ ]:


Prime_counter = prime_cities.count(True) # Count the number of True in the list
print("There are",Prime_counter,"prime cities out of",len(df_cities_10),"cities.")


# Dumbest path distance

# In[ ]:


def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             numpy.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + 
                       pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) * \
            (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

dumbest_path = list(df_cities_10.CityId[:].append(pandas.Series([0])))
print('Total distance with the dumbest path is '+ "{:,}".format(total_distance(df_cities_10,dumbest_path)))


# Dumbest path plot
# 

# In[ ]:


df_path = pandas.merge_ordered(pandas.DataFrame({'CityId':dumbest_path}),df_cities_10,on=['CityId'])
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')
for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
    ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 15)
    


# Sorting X and Y and gettint new distance

# In[ ]:


sorted_cities = list(df_cities_10.iloc[1:,].sort_values(['X','Y'])['CityId'])
sorted_cities = [0] + sorted_cities + [0]
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(df_cities_10,sorted_cities)))


# Plot sorted X and Y

# In[ ]:


df_path = pandas.DataFrame({'CityId':sorted_cities}).merge(df_cities_10,on=['CityId'])
fig, ax = plt.subplots(figsize=(15,10))
ax.set_xlim(0,10)
ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')
for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
    ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 15)
    
#     df_path = pandas.merge_ordered(pandas.DataFrame({'CityId':dumbest_path}),df_cities_10,on=['CityId'])
# fig, ax = plt.subplots(figsize=(15,10))
# ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')
# for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
#     ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 15)


# In[ ]:




