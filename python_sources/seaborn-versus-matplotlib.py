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


# # Seaborn Versus Matplotlib
# 

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# Let's Create Some Data

# In[ ]:


range = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(range.randn(100, 5))


# In[ ]:


# Plot the data with Matplotlib defaults
plt.plot(x,y)
plt.legend('A', ncol=2, loc='upper right')


# In[ ]:


import seaborn as sns
sns.set()


# In[ ]:





# In[ ]:


data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)


# In[ ]:




for col in 'xy':
    sns.kdeplot(data[col], shade=True)


# In[ ]:


#SEABORN distplot
sns.distplot(data['x'])
sns.distplot(data['y']);


# #If we pass the full two-dimensional dataset to kdeplot, we will get a two-dimensional visualization of the data:

# In[ ]:


sns.kdeplot(data);


# **jointplot in Seaborn***

# In[ ]:


with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');


# In[ ]:


#histogram type form of above jointplot


# In[ ]:


with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')


# 

# In[ ]:




