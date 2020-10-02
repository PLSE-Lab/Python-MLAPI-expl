#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataframe = pd.read_csv('/kaggle/input/weight-height/weight-height.csv')


# In[ ]:


dataframe.head()


# In[ ]:


heights = np.array(dataframe['Height'])
weights = np.array(dataframe['Weight']).reshape(-1, 1)


# In[ ]:


heights.shape


# In[ ]:


weights.shape


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(weights, heights, 'bs')


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


my_linear_model = LinearRegression()


# In[ ]:


my_linear_model.fit(weights, heights)


# In[ ]:


heights_prediction = my_linear_model.predict(weights)


# In[ ]:


plt.plot(weights, heights, 'bs')
plt.plot(weights, heights_prediction, 'ro')


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mse = mean_squared_error(heights, heights_prediction)


# In[ ]:


mse


# In[ ]:


mse**0.5

