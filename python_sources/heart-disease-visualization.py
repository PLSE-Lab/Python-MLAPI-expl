#!/usr/bin/env python
# coding: utf-8

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


data = pd.read_csv('../input/heart.csv')

type(data)
data.head()


# In[ ]:


print(data.shape)
print(data.columns)


# In[ ]:


data.corr()


# Below is a parallel coordinates analysis from heart disease dataset. This visualization shows the cholesterol attribute is the unique that presents any distinction to categorize the two classes.

# In[ ]:


from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
parallel_coordinates(data, 'target',color=('#556270', '#4ECDC4'))

plt.show()


# In[ ]:


import seaborn as sns
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot = True)


# In[ ]:



data.plot.scatter(x='chol', y='chol',c='target',  colormap='viridis', figsize=(20,10))

