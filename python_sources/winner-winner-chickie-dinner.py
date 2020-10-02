#!/usr/bin/env python
# coding: utf-8

# # Let's get this party started!

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


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Load data and split the features and y_label

# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


features = data[lambda data: data.columns[1:-1]]


# In[ ]:


features.head()


# In[ ]:


y_label = data.iloc[:,-1]


# In[ ]:


y_label.head()


# ## Show me those visualizations! Enough of numbers

# In[ ]:


#sns.pairplot(features[:100])


# In[ ]:


#sns.pairplot(data[:100])


# ### Something less complex maybe ?

# In[ ]:


data.hist()


# ### Hmm, this doesn't look so good. Let's add size to the plot for better placement of the histograms.

# In[ ]:


fig = plt.figure(figsize = (20,20))
ax = fig.gca()
data.hist(ax = ax)


# In[ ]:


fig = plt.figure(figsize = (20,20))
ax = fig.gca()
data.plot(kind='density', subplots=True, layout=(8,4), sharex=False, ax=ax)
#data.plot(kind='density', subplots=True, layout=(8,4), sharex=False)


# In[ ]:




