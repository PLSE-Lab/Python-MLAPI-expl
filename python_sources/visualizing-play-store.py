#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # data visualization
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Read  CSV

# In[ ]:


dataframe=pd.read_csv('../input/googleplaystore.csv')
dataframe.head()
nRows,nCols=dataframe.shape


# # Remove duplicate rows

# In[ ]:


# Clean your data
dataframe=dataframe.drop_duplicates(subset=['App','Rating'],keep=False)
nRows=nRows-dataframe.shape[0]
print('{} duplicate rows deleted'.format(nRows))
dataframe=dataframe.drop(dataframe.index[dataframe['Rating']>5])
dataframe.shape


# # Know your data

# In[ ]:


dataframe.head()


# In[ ]:


dataframe.info()


# In[ ]:


rating=dataframe.loc[:,["App","Category","Rating","Size","Installs","Type","Price","Content Rating"]]
rating=rating.sort_values(by=["Rating"],ascending=False)


# # Visualize the data

# In[ ]:


plt.figure(figsize=(10,8))
sb.countplot(y=rating['Category'])


# In[ ]:


plt.figure(figsize=(15,7))
sb.countplot(rating["Rating"])


# In[ ]:




