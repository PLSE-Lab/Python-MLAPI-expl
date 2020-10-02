#!/usr/bin/env python
# coding: utf-8

# # Random Insertion
# ![Imgur](https://i.imgur.com/Y4XV0R9.jpg)

# # Import some libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/cities.csv")

print(df.head())
all_path = df.values
print(all_path)


# # Function to calculate distance simply 

# In[ ]:


def path_length(path=None):
    length = 0
    now_city = path[0]
    for city in path[1:]:
        between = np.sqrt((city[1]-now_city[1])**2 + (city[2]-now_city[2])**2)
        now_city = city
        length += between
    return length


# # Function for searching best insertion position
# 1. Calculate the distance between the selected city and each already set town
# 2. Adjacent ones of the distances calculated in 1 and Subtract the distance between already set towns
# 3. In 2, the distance which is increased when the selected town is inserted between already set towns is calculated
# 4. Return the insertion point with the least increase

# In[ ]:


def short_path_search(now_path=None, new_city=None, dist_list=None):
    
    dist_to_city = (now_path[:,1] - new_city[1])**2 + (now_path[:,2] - new_city[2])**2
    increase_dist = dist_to_city[:-1] + dist_to_city[1:] - dist_list
    best_point = np.argmin(increase_dist)
    return best_point, dist_to_city[best_point], dist_to_city[best_point+1] 


# # Set initial path(start city and goal city)
# 1. Set start 0 city and goal 0 city
# 2. Remove 0 city from rest_city

# In[ ]:


# set initial path(start and goal)
now_path = np.stack([all_path[0], all_path[0]])
rest_city = np.delete(all_path, 0, 0)
dist_list = np.array([0])
new_dist = np.ndarray([2])


# # Choose a city at random and Insert it in the best position
# 1. Choose a city from rest_city at random
# 2. Calculate insertion point with the least increase
# 3. Insert it to now_path and Remove it from rest_city

# In[ ]:


i = 0
while len(rest_city)>0:
    # choice next city randomly
    rand = np.random.randint(len(rest_city))

    #calc best point
    best_point, new_dist[0], new_dist[1] = short_path_search(now_path, rest_city[rand,:], dist_list)
    # insert best point
    now_path= np.insert(now_path, best_point+1, rest_city[rand], 0)
    # delete city from rest_city
    rest_city = np.delete(rest_city, rand, 0)
    # update dist_list
    dist_list[best_point] = new_dist[0]
    dist_list = np.insert(dist_list, best_point+1, new_dist[1], 0)
    #print(now_path)
    #print('')
    if i%10000==0:
        print(i)
    i += 1


# # Calculate distanse simply and make submission file

# In[ ]:


print(path_length(now_path))
pd.DataFrame({'Path': now_path[:,0].astype(np.int32)}).to_csv('submission.csv', index=False)

