#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from numpy.random import randn, randint, uniform, sample


# In[ ]:


df = pd.DataFrame(randn(1000), index = pd.date_range('2019-06-07', periods = 1000), columns=['value'])
ts = pd.Series(randn(1000), index = pd.date_range('2019-06-07', periods = 1000))
df.head()


# In[ ]:


df['value'] = df['value'].cumsum()
df.head()


# In[ ]:


ts = ts.cumsum()
ts.head()


# In[ ]:


type(df), type(ts)


# In[ ]:


ts.plot(figsize=(5,5))


# In[ ]:


plt.plot(ts)


# In[ ]:


df.plot()


# In[ ]:


iris = sns.load_dataset('iris')
iris.head()


# In[ ]:


ax = iris.plot(figsize=(15,8), title='Iris Dataset')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')


# In[ ]:


ts.plot(kind = 'bar')
plt.show()


# In[ ]:


df = iris.drop(['species'], axis = 1)


# In[ ]:


df.iloc[0]


# In[ ]:


df.iloc[0].plot(kind='bar')


# In[ ]:


df.iloc[0].plot.bar()


# In[ ]:


titanic = sns.load_dataset('titanic')


# In[ ]:


titanic.head()


# In[ ]:


titanic['pclass'].plot(kind = 'bar')


# In[ ]:


df = pd.DataFrame(randn(10, 4), columns=['a', 'b', 'c', 'd'])
df.head(10)


# In[ ]:


df.plot.bar()


# In[ ]:


df.plot(kind = 'bar')


# In[ ]:


df.plot.barh()


# In[ ]:


iris.plot.hist()


# In[ ]:


iris.plot(kind = 'hist')


# In[ ]:


iris.plot(kind = 'hist', stacked = False, bins = 100)


# In[ ]:


iris.plot(kind = 'hist', stacked = True, bins = 50, orientation = 'horizontal')


# In[ ]:


iris['sepal_width'].diff()


# In[ ]:


iris['sepal_width'].diff().plot(kind = 'hist', stacked = True, bins = 50)


# In[ ]:





# In[ ]:


df = iris.drop(['species'], axis = 1)
df.diff().head()


# In[ ]:


df.diff().hist(color = 'b', alpha = 0.1, figsize=(10,10))


# In[ ]:


color = {'boxes': 'DarkGreen', 'whiskers': 'b'}
color


# In[ ]:


df.plot.scatter(x = 'sepal_length', y = 'petal_length')


# In[ ]:


df.plot.scatter(x = 'sepal_length', y = 'petal_length', c = 'sepal_width')


# In[ ]:


df.head()


# In[ ]:


df.plot.scatter(x = 'sepal_length', y = 'petal_length', label = 'Length');
#df.plot.scatter(x = 'sepal_width', y = 'petal_width', label = 'Width', ax = ax, color = 'r')
#df.plot.scatter(x = 'sepal_width', y = 'petal_length', label = 'Width', ax = ax, color = 'g')


# In[ ]:


df.plot.scatter(x = 'sepal_length', y = 'petal_length', c = 'sepal_width', s = 190)


# In[ ]:


df.plot.hexbin(x = 'sepal_length', y = 'petal_length', gridsize = 5, C = 'sepal_width')


# In[ ]:


d = df.iloc[0]
d


# In[ ]:


d.plot.pie(figsize = (10,10))


# In[ ]:


d = df.head(3).T


# In[ ]:


d.plot.pie(subplots = True, figsize = (20, 20))


# In[ ]:


d.plot.pie(subplots = True, figsize = (35, 25), fontsize = 26, autopct = '%.2f')
plt.show()


# In[ ]:


[0.1]*4


# In[ ]:


series = pd.Series([0.2]*5, index = ['a','b','c', 'd','e'], name = 'Pie Plot')
series.plot.pie()
plt.show()


# In[ ]:


from pandas.plotting import scatter_matrix


# In[ ]:


scatter_matrix(df, figsize= (8,8), diagonal='kde', color = 'b')
plt.show()


# In[ ]:


ts.plot.kde()


# In[ ]:


from pandas.plotting import andrews_curves


# In[ ]:


andrews_curves(df, 'sepal_width')


# In[ ]:


ts.plot(style = 'r--', label = 'Series', legend = True)
plt.show()


# In[ ]:


df.plot(legend = True, figsize = (10, 5), logy = True)
plt.show()


# In[ ]:


x = df.drop(['sepal_width', 'petal_width'], axis = 1)
x.head()


# In[ ]:


y = df.drop(['sepal_length', 'petal_length'], axis = 1)
y.head()


# In[ ]:


ax = x.plot()
y.plot(figsize = (16,10), secondary_y=True, ax = ax)
plt.show()


# In[ ]:


x.plot(figsize=(10,5), x_compat = True)
plt.show()


# In[ ]:


df.plot(subplots = True)
plt.show()


# In[ ]:


df.plot(subplots = True, sharex = False, layout = (2,3), figsize = (16,8))
plt.tight_layout()
plt.show()

