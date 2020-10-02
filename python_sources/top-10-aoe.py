#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from csv import reader # using list of list to print top 10 AOE
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **To start, create a list of lists and remove the header row. **

# In[ ]:


open_file = open('/kaggle/input/college-basketball-dataset/cbb.csv')
read_file = reader(open_file)
ncaa = list(read_file)

ncaa = ncaa[1:]


# **Next, create a function to sort the data in the lists by creating tuples then using the sort function. **

# In[ ]:


def sort(dictionary, show=False):
    table_display = []
    for key in dictionary:
      if float(key) > 100:
        key_val_as_tuple = (dictionary[key], key)
        table_display.append(key_val_as_tuple)

    table_sorted = sorted(table_display, key=lambda x: x[1], reverse=True)
    if show == True:
        for entry in table_sorted:
            print(entry[1], ':', entry[0])
    return(table_sorted)


# **Now we must create a dictionary to hold our values of the team and the aoe. I created a new team name since we are working with the same teams, but in different years. **

# In[ ]:


aoe_dict = {}

for team in ncaa:
  name = team[0] + '_' + team[23]
  aoe = team[4]
  if name not in aoe_dict:
    aoe_dict[aoe] = name


# In[ ]:


print(aoe_dict)


# **We have the data, but we can't really tell what is the top 10 AOE from it. In ordered to do this, we can use the sort function we created earlier. Then we can print out the top 10 from that list.**

# In[ ]:


aoe = sort(aoe_dict)
for entry in aoe[:9]:
  print(entry[1], ':', entry[0])


# **I am going to see if we can plot the data to make it more visual.**

# In[ ]:


x = []
y = []
for entry in aoe[:10]:
  x.append(entry[0])
  y.append(float(entry[1]))

plt.scatter(y,x)
plt.show()

