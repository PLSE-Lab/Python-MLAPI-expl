#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('../input/Iris.csv')


# Let's learn little about how to explore the data basically, and get a tip about how to visualizing with seaborn.
# The first step is to sample from dataset randomly, so we are trying to have a brief glance of the data. Here 10 samples are picked randomly, and feature Id help nothing to learn the iris data, so we drop that.

# In[ ]:


df.sample(10).sort_index()


# In[ ]:


df.drop('Id',axis=1, inplace=True)


# In[ ]:


df.shape


# Of course, we can have a simple correlation plot between each pair features by sns.pairplot; And a reachable method for customizing plot is to use Sns.Pairplot

# In[ ]:


sns.pairplot(df, hue = 'Species')


# In[ ]:


g = sns.PairGrid(df, hue = 'Species',palette='Set2')
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.regplot)
g.map_upper(sns.kdeplot)


# And you can name features to learn there relations here using x_vars and y_vars

# In[ ]:


sns.pairplot(df,x_vars=['SepalLengthCm','SepalWidthCm'], 
             y_vars=['PetalLengthCm','PetalWidthCm'],hue='Species',palette='Set2')


# Because dataset has been sorted by species, here we use sns.heatmap to see species are how to related to Sepal and petal.

# In[ ]:


sns.heatmap(df.select_dtypes(include='float'))


# Another usage of sns.heatmap is to emphasis correlation, based on pd.DataFrame.corr, So we have noticed there might be a strong relations between some features.

# In[ ]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm',center=0)


# I didn't use any model to learn how predicting a species with its features here, but you can use sklearn to make prediction. Really thanks for your passion going through my virgin work on kaggle :), hoping it is helpful for you.

# In[ ]:




