#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sqrt, pow

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from tqdm import tqdm

# Any results you write to the current directory are saved as output.


# In[ ]:


# read csv
cities = pd.read_csv("../input/cities.csv")
smpsb_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


# red point is start and goal
cities.plot.scatter(x="X", y="Y", s=1, alpha=.1)
plt.scatter(cities.iloc[0: 1, 1], cities.iloc[0: 1, 2], s=10, c="red")


# In[ ]:


X = cities.X.values.astype(np.float64)
Y = cities.Y.values.astype(np.float64)
print("X: [{:.2}, {:.5}]".format(cities.X.min(), cities.X.max()))
print("Y: [{:.2}, {:.5}]".format(cities.Y.min(), cities.Y.max()))
cities = cities.iloc[1:, :]


# In[ ]:


# initialize path
path = [0]
path.extend(cities.query("X < 315 & Y >= 2200").sort_values("Y")["CityId"].tolist())

forward = True
band = 17
zigzag_cities = cities.query("X >= 315")
zigzag_Y = zigzag_cities.Y.values
for i in range(-(-3400 // band)):
    mini_cities = zigzag_cities.iloc[(3400 - band*(i+1) <= zigzag_Y) &
                                     (zigzag_Y < 3400 - band*i), :]
    mini_cities = mini_cities.sort_values("X", ascending=forward)
    path.extend(mini_cities["CityId"].tolist())
    forward = not forward

path.extend(cities.query("X < 315 & Y < 2200").sort_values("Y")["CityId"].tolist())
path.append(0)
assert len(path) == 197770


# In[ ]:


# calculate pathlength
def distance_from(a):
    global path
    return sqrt(pow(X[path[a+1]] - X[path[a]], 2) + pow(Y[path[a+1]] - Y[path[a]], 2))

def neighborh(a):
    global path
    return path[a-1], path[a+1]

def distance_between(a, b):
    global path
    return sqrt(pow(X[path[a]] - X[path[b]], 2) + pow(Y[path[a]] - Y[path[b]], 2))

before_length = 0
for i in tqdm(range(197770 - 1)):
    before_length += distance_from(i)
before_length


# In[ ]:



for i in range()


# In[ ]:


from datetime import datetime as dt
tic = dt.now()
# greedy swapping

np.random.seed(2434)
swap = 0
for _ in range(int(1e8)):
    i1 = np.random.randint(10, 197740)
    j1 = i1 + np.random.randint(2, 7)
    i0, i2 = i1 - 1, i1 + 1
    j0, j2 = j1 - 1, j1 + 1
    before = (distance_between(i0, i1) +
              distance_between(i1, i2) + 
              distance_between(j0, j1) +
              distance_between(j1, j2))
    after = (distance_between(i0, j1) +
             distance_between(j1, i2) + 
             distance_between(j0, i1) +
             distance_between(i1, j2))
    if before > after:
        swap += 1
        path[i1], path[j1] = path[j1], path[i1]
toc = dt.now()
print("{} sec".format((toc-tic).seconds))
print("swapping rate is:", swap / 1000000)


# In[ ]:


after_length = 0
for i in tqdm(range(197770 - 1)):
    after_length += distance_from(i)
after_length


# In[ ]:


# submit file
smpsb_df["Path"] = path
smpsb_df.to_csv("zig_zag_greedy_1e8.csv", index=None)
assert smpsb_df.loc[0 , "Path"] == 0
assert smpsb_df.loc[197769 , "Path"] == 0
assert smpsb_df["Path"].unique().shape[0] == 197769

