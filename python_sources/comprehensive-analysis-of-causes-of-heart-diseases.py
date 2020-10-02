#!/usr/bin/env python
# coding: utf-8

# In this analysis study, possible causes of heart diseases were tried to learn by using features obtained from data set.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


data.info()


# In[ ]:


#Corelation Matrix
ax = plt.subplots(figsize=(18,18))
ax = sns.heatmap(data.corr(), vmin=-1, vmax=1, center=0, annot=True, linewidths=1, cmap=plt.get_cmap("PiYG", 10))


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


#Matplotlib
data.age.plot(kind='line', color='g', label='Age', linewidth=1, grid=True, linestyle=':')
data.thalach.plot(color='r', label='Thalach', linewidth=1, grid=True, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


#Scatter Plot
#thalach --> The person's maximum heart rate achieved
#chol --> The person's cholesterol measurement in mg/dl
data.plot(kind='scatter', x='thalach', y='chol', alpha=0.5, color='red')
plt.xlabel('trestbps')
plt.ylabel('fbs')
plt.title('The persons maximum heart rate & cholesterol measurement')


# In[ ]:


#Histogram
data.chol.plot(kind='hist', bins=50, figsize=(18,9))
plt.title('The person\'s cholesterol measurement in mg/dl')
plt.show()


# In[ ]:


data.shape


# In[ ]:


data.tail(10)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column='chol', by='target')
plt.show()


# In[ ]:


data_new = data.head()
melted = pd.melt(frame=data_new, id_vars='age', value_vars=['trestbps', 'oldpeak'])
melted


# In[ ]:


melted.pivot(index='age', columns='variable', values='value')


# In[ ]:


data1 = data.head()
data2 = data.tail()
concat_data = pd.concat([data1, data2], axis=0, ignore_index=True)
concat_data


# In[ ]:


data.info()


# The data set does not contain any empty or missing data.

# In[ ]:


data1 = data.loc[:,['chol', 'trestbps']]
data1.plot()


# In[ ]:




