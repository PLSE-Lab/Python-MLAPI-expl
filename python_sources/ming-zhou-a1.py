#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random 
import os
from collections import Counter


# In[ ]:


#read in csv file
df_cities = pd.read_csv("/kaggle/input/cities10/cities10.csv")


# In[ ]:


#To show all the cities when visualised as a scatter plot.
plt.figure(figsize=(15, 10))
plt.scatter(df_cities.X, df_cities.Y, s=1)
plt.scatter(df_cities.iloc[0: 1, 1], df_cities.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.title("Cities of 10% of dataset")
plt.show()


# **This shows the dataset for the Travelling Santa competition appears when visualised as a scatter plot.**
# * The red dot indicates the North Pole (CityId = 0). 
# * This is 10% of the full data set.

# In[ ]:


number_of_cities = max(df_cities.CityId)
print("Number of cities to visit : ", number_of_cities)


# **The "Sieve of Eratosthenes" function below is for finding all prime numbers up to any given limit.**
# * One of the parameters for the travelling santa competition is that if the chosen path does not originate from a prime city exactly every 10th step, it takes 10% longer. 
# * Therefore having a list of prime numbers within the limit is essential for calculating the length of a path.

# In[ ]:


# Getting Number of Prime Cities
def sieve_eratosthenes(n):
    primes = [False, False] + [True for i in range(n - 1)]
    p = 2
    while (p * p <= n):
        if (primes[p] == True):
            for i in range(p * 2, n + 1, p):
                primes[i] = False
        p += 1
    return primes


# **The data structure used in this algorithm is a list.**
# * This function does not have a linear run time as it contains a for loop nested in a while loop
# * At worst, it will run through n elements n times O(n^2)
# * List comprehension runs n times with 2 operations
# * While loop runs n times
# * If statement runs 1 time with 1 primitive operations inside it
# * A for loop runs n times with 2 operations

# In[ ]:


primes = np.array(sieve_eratosthenes(number_of_cities)).astype(int)
df_cities['Prime'] = primes


# In[ ]:


penalization = 0.1 * (1 - primes) + 1


# In[ ]:


plt.figure(figsize=(15, 10))
sns.countplot(df_cities.Prime)
plt.title("Prime repartition : " + str(Counter(df_cities.Prime)))
plt.show()


# In[ ]:


#To show the prime cities when visualised as a scatter plot.
plt.figure(figsize=(15, 10))
plt.scatter(df_cities[df_cities['Prime'] == 0].X, df_cities[df_cities['Prime'] == 0].Y, s=1, alpha=0.4)
plt.scatter(df_cities[df_cities['Prime'] == 1].X, df_cities[df_cities['Prime'] == 1].Y, s=1, alpha=0.6, c='blue')
plt.scatter(df_cities.iloc[0: 1, 1], df_cities.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.title('Prime Cities')
plt.show()


# There are prime cities approximately all around the map.

# In[ ]:


def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(primes[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance


# * This function is linear however the worst operation
# * There are 4 primitive operations before the for loop (assignment and sequence access)
# * The for loops runs for n amount of time
# * The numpy package is then accessed, adding to the run time

# In[ ]:


start = time.time()
original_path = list(df_cities.CityId[:].append(pd.Series([0])))
print('Total distance with path in original order is '+ "{:,}".format(total_distance(df_cities,original_path)))
end = time.time()

#Algorithm run time
print("Run time:", end - start)


# In[ ]:


#show the first 100 steps with path in original order
df_path = pd.merge_ordered(pd.DataFrame({'CityId':original_path}),df_cities,on=['CityId'])
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')

for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
    ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 15)


# **In order to solve the Travelling Santa Problem for the assignment 1, I will make use of the insertion sort algorithm:**

# In[ ]:


start = time.time()
sorted_cities = list(df_cities.iloc[1:,].sort_values(['X','Y'])['CityId'])
sorted_cities = [0] + sorted_cities + [0]
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(df_cities,sorted_cities)))
end = time.time()

#Algorithm run time
print("Run time:", end - start)


# In[ ]:


#show the first 100 steps with sorted in X,Y coordinate_order
df_path = pd.DataFrame({'CityId':sorted_cities}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(15,10))
ax.set_xlim(0,50)
ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')

for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
    ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 15)

