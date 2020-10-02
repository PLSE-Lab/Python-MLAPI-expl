#!/usr/bin/env python
# coding: utf-8

# ## This notebook explores Python data visualizations on the World Food fact Dataset
# https://www.kaggle.com/openfoodfacts/world-food-facts/downloads/world-food-facts-release-2016-01-13-03-19-37.zip
# This Python 3 environment comes with many helpful analytics libraries installed. It is defined by the [kaggle/python docker image](https://github.com/kaggle/docker-python)
# 
# We'll use pandas and seaborn for  this tutorial: [pandas](http://pandas.pydata.org/) [seaborn](http://seaborn.pydata.org/)

# In[ ]:


import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sbn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3

# Input data files are available in the "../input/" directory. 
foodfacts=pd.read_csv("../input/FoodFacts.csv")
foodfacts.head()#print the data


# In[ ]:


ax=sbn.boxplot(x="energy_100g",y="sugars_100g",data=foodfacts)
ax=sbn.stripplot(x="energy_100g",y="sugars_100g",data=foodfacts,jitter=True)


# In[ ]:


foodfacts.plot(kind="scatter",x="energy_100g", y="sugars_100g",title="A plot of energy to sugar", legend=True)


# In[ ]:


sbn.jointplot(x="energy_100g", y="sugars_100g", data=foodfacts, size=5)


# In[ ]:



ax = sbn.stripplot(x="energy_100g", y="sugars_100g", data=foodfacts, jitter=True, edgecolor="gray")
ax=sbn.boxplot(x="energy_100g", y="sugars_100g", data=foodfacts)


# In[ ]:


sbn.violinplot(x="energy_100g", y="sugars_100g", data=foodfacts, size=5)


# In[ ]:





# In[ ]:





# In[ ]:




