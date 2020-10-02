#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/2017.csv") # Read dataset


# In[ ]:


data.info()


# In[ ]:


data.head(10) # Top 10 countries


# In[ ]:


data.tail(10) # Worst 10 countries


# In[ ]:


df = data.drop(columns = ['Happiness.Rank']) # I think it's unnecessary
df.corr()


# **HAPPINESS DEPEND ON ECONOMY AND HEALTHY LIFE**
# 
# As you see in this heat map, economy and health are main determinant in happiness.

# In[ ]:


# The Correlation Map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot = True, linewidths = .5, fmt= '.2f', ax=ax)
plt.show()


# In[ ]:


data.columns # Our features


# In[ ]:


# Let's look closer to Economy and Health life exp.

data.plot(kind = 'line', x = 'Happiness.Score', y = 'Economy..GDP.per.Capita.', color = 'b', label = 'Eco', linewidth = 1, grid = True, linestyle = '-', figsize = (10,8))
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Happiness Score')              # label = name of label
plt.ylabel('GDP per capita')
plt.title('Eco.-Happ.')            # title = title of plot
plt.show()


# In[ ]:


data.plot(x = 'Happiness.Score', y = 'Health..Life.Expectancy.', color = 'r', label = 'Health', linewidth = 1, alpha = 1, grid = True, figsize=(10,8))
plt.legend(loc='upper right')     
plt.xlabel('Happiness Score')              
plt.ylabel('Health Life Expectancy')
plt.title('Life Exp.-Happ.')           
plt.show()


# In[ ]:


# Scatter Plot 
data.plot(kind='scatter', x='Family', y='Freedom',alpha = 0.7,color = 'g', figsize = (10,8))
plt.xlabel('Family')    
plt.ylabel('Freedom')
plt.title('Family-Freedom Scatter Plot')
plt.show()


# **Is generosity good indicator to determinant happiness rank ?**
# 
# Actually, I expected "Yes" but according to plot its not seen like that.

# In[ ]:



data.Generosity.plot(kind = 'line',color = 'b', figsize = (10,8), grid = True)
plt.title("Generosity-Happ.Rank")
plt.xlabel("Happiness Rank")
plt.ylabel("Generosity")
plt.show()

