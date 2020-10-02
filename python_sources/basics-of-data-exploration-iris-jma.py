#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

data.head()


# In[ ]:


from pandas.api.types import is_numeric_dtype

for col in data.columns:
    if is_numeric_dtype(data[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data[col].mean())
        print('\t Standard deviation = %.2f' % data[col].std())
        print('\t Minimum = %.2f' % data[col].min())
        print('\t Maximum = %.2f' % data[col].max())


# In[ ]:


data['class'].value_counts()


# In[ ]:


data.describe(include='all')


# In[ ]:


print('Covariance:')
data.cov()


# In[ ]:


print('Correlation:')
data.corr()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

data['sepal length'].hist(bins=8)


# In[ ]:


data.boxplot()


# In[ ]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(12,12))
index = 0
for i in range(3):
    for j in range(i+1,4):
        ax1 = int(index/2)
        ax2 = index % 2
        axes[ax1][ax2].scatter(data[data.columns[i]], data[data.columns[j]], color='red')
        axes[ax1][ax2].set_xlabel(data.columns[i])
        axes[ax1][ax2].set_ylabel(data.columns[j])
        index = index + 1


# In[ ]:


from pandas.plotting import parallel_coordinates
get_ipython().run_line_magic('matplotlib', 'inline')

parallel_coordinates(data, 'class')

