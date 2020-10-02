#!/usr/bin/env python
# coding: utf-8

# ## Univariate,Bivariate and MultiVariate Analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# ### Univariate Analysis

# In[ ]:


df_setosa=df.loc[df['species']=='setosa']


# In[ ]:


df_virginica=df.loc[df['species']=='virginica']
df_versicolor=df.loc[df['species']=='versicolor']


# In[ ]:


plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.xlabel('Petal length')
plt.show()


# ### Bivariate Analysis

# In[ ]:


sns.FacetGrid(df,hue="species",size=5).map(plt.scatter,"petal_length","sepal_width").add_legend();
plt.show()


# ### Multivariate Analysis

# In[ ]:


sns.pairplot(df,hue="species",size=3)

