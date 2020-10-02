#!/usr/bin/env python
# coding: utf-8

# <h3> This notebook is the homework of <a href="https://www.kaggle.com/kanncaa1">Kaan Can's</a> Udemy course. </h3>
# 

# <h1> Reading and Viewing Data </h1>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h3> Reading our data </h3>

# In[ ]:


data = pd.read_csv("../input/Iris.csv")


# <h3> Reviewing data </h3>

# In[ ]:


data.info()


# <h3> Correlation </h3>
# <p> Checking correlations beetween features. </p>

# In[ ]:


data.corr()


# <p> We can see the correlations between features.
#         If correlation closer to<b> 1</b>, there is a positive correlation between features and if correlation closer to <b>-1</b>, there is a negative correlation between features.
# </p>
# 
# <div></div>
# 
# <p> What is the correlation? http://www.statisticshowto.com/what-is-correlation/ </p> 

# <h3> Visualizing our correlations </h3>
# <p> You can play with parameters to understand what are they. Dont be afraid to changing code. </h3>

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(10, 10)) 
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) 
plt.show()


# <h3> Take a first look our data </h3>
# <p> You can change the number and see result. </p>

# In[ ]:


data.head(10)


# <h3> Viewing features of data </h3>
# <p> This is important point. We need know features of data. </p>

# In[ ]:


data.columns


# <h1> Visualizing Data </h1>
# <p> We are going to use matplotlib for visualization. </p>

# <h3> Line Plot </h3>
# <p> Line plot is more useful for time series datasets. <p>

# In[ ]:


data.PetalWidthCm.plot(kind = 'line', color = 'g',label = 'PetalWidthCm',linewidth=1,grid = True)
data.SepalLengthCm.plot(kind = 'line', color = 'r',label = 'SepalLengthCm',linewidth=1,grid = True)
plt.legend()     # legend = puts label into plot
plt.xlabel('X Axis')              # label = name of label
plt.ylabel('Y Axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# <h3> Scatter Plot </h3>
# <p> Scatter is better if there is correlation between two values.<p>

# In[ ]:


data.plot(kind='scatter', x='PetalWidthCm', y='SepalLengthCm',alpha = 0.4,color = 'red')
plt.xlabel("PetalWidthCm")              # label = name of label
plt.ylabel("SepalLengthCm")
plt.title("Scatter Plot")            # title = title of plot


# <h3> Histogram </h3>
# <p> Histogram makes us to see distribution of numerical data.</p>

# In[ ]:


data.SepalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# <h1> Filtering Data </h1>
# <p> Filtering ise very useful when we want to specify features.<p>

# 

# In[ ]:


filtered_data = data['SepalWidthCm'] > 3
filtered_data


# <h3> Filtering with "and" <h3>

# In[ ]:


data.head()
and_filtered_data = data[(data['SepalWidthCm']>3) & (data['SepalLengthCm']>4)]
and_filtered_data


# <h3>This notebook will updating regularly</h3>

# 
