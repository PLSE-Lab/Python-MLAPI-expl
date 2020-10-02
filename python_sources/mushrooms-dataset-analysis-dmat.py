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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


mushroom_data = pd.read_csv("../input/mushrooms.csv")
mushroom_data.head()


# In[ ]:


mushroom_data['class'].nunique()


# In[ ]:


mushroom_data.shape


# In[ ]:


mushroom_data.count()


# In[ ]:


mushroom_data['cap-color'].value_counts()


# In[ ]:


# I changed the size on the basis of how John Nana did it in his coursework
#-------------------------------------------------------
plt.figure(figsize=(15,5))
#-------------------------------------------------------
mushroom_data['cap-color'].value_counts().plot.bar()


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
mushroom_data['cap-color'].value_counts().plot.area()


# In[ ]:


mushroom_data.describe()


# In[ ]:


mushroom_data.info(verbose = None)
#verbose - whether to print the full summary, set to None (but max_info_columns by default)


# In[ ]:


# Here is the code suggested by Shahin Rostami to convert categorical data to numerical data
#---------------------------------------------------------------------------------------------------------
mushroom_dataNum = pd.read_csv("../input/mushrooms.csv").astype('category')
cat_columns = mushroom_dataNum.select_dtypes(['category']).columns
mushroom_dataNum[cat_columns] = mushroom_dataNum[cat_columns].apply(lambda x: x.cat.codes)
#---------------------------------------------------------------------------------------------------------

# Data is converted to numerical and colours are adjusted for better contrast between two classes
plt.figure(figsize=(30,10))
pd.plotting.parallel_coordinates(mushroom_dataNum, 'class', color=('#2c8913', '#ba1a4f'))

plt.show()


# In[ ]:


# Data is converted to numerical and colours are adjusted for better contrast between two classes
# Additionally, data is filtered to smaller selection of columns.

plt.figure(figsize=(30,10))
pd.plotting.parallel_coordinates(mushroom_dataNum.filter(['class', 'cap-surface', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'gill-size', 'ring-number']), 'class', color=('#faff05', '#0564ff'))

plt.show()

