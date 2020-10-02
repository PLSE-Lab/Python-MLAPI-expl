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


data = pd.read_csv('../input/cwurData.csv')  #reading the data


# After reading our data let's get informations about our data.

# In[ ]:


data.info()


# As we can see here our data contains 14 columns which are 2 float types, 10 int types and 2 string types.

# To look the data widely, we get data's first 5 rows and last 5 rows.

# In[ ]:


data.head(5)  #first 5


# In[ ]:


data.tail(5) #last 5


# Now we know overview of data.  

# To see data's columns clearly we may use another method.

# In[ ]:


data.columns


# Now let's check if there is correlation between columns or not.

# In[ ]:


data.corr()


# This is the correlations between columns.For instance There is a positive correlation beetween quality_of_education and  world_rank.

# To see these correlations easily we can plot these correlations.

# In[ ]:


f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.2f', ax=ax)
plt.show()


# ----------------------------------------------

# To investigate the data in 2015 we're filtering the data as it contains only 2015.

# In[ ]:


dF2015 = data[data['year'] == 2015]


# 
# 

# Now to understand the data, we use different types of  plotting techniques which are line, scatter and histogram plotting.

# First, we're going to do line plotting.

# In[ ]:


# Line plotting
dF2015.broad_impact.plot(kind = 'line', color = 'g', label = 'broad_impact', linewidth = 1, alpha = 0.6, grid = True, linestyle = ':')
dF2015.world_rank.plot(kind = 'line', color = 'blue', label = 'world_rank', linewidth = 1, alpha = 0.6, grid = True, linestyle = '-.')
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


# Scatter Plot
dF2015.plot(kind = 'scatter', x = 'broad_impact', y = 'world_rank',color = 'b', alpha = 0.5)
plt.xlabel('broad_impact')
plt.ylabel('world_rank')
plt.title('broad_impact  world_rank scatter plot ')
plt.show()


# As you can see in the scatter plot, universities broad impact increas cause to increase of their world record's.

# In[ ]:


#Histogram    To see the frequency of one column's variables.
data.publications.plot(kind = 'hist', bins = 100, figsize = (12,12),color = 'b',alpha = 0.8)
plt.show()


# We can easily see the positive correlation  beetween two contents by using scatter plot.

# Let's compare first 5 unique country's scores.

# First filter data as a unique countries.

# In[ ]:


dFUSA = dF2015[dF2015.country == 'USA']
dFUK = dF2015[dF2015.country == 'United Kingdom']
dFJAPAN = dF2015[dF2015.country == 'Japan']
dFSWITZERLAND = dF2015[dF2015.country == 'Switzerland']
dFISRAEL = dF2015[dF2015.country == 'Israel']


# Secondly, take the means of country's scores and put them in a list.

# In[ ]:


score_means =[dFUSA[0:5].score.mean(),dFUK[0:5].score.mean(),dFJAPAN[0:5].score.mean(),dFSWITZERLAND[0:5].score.mean(),dFISRAEL[0:5].score.mean()]


# Lastly, use bar plotting to see clearly.

# In[ ]:


plt.bar(dF2015.country.unique()[0:5] , score_means[0:5], width=0.5, color='r',alpha = 0.7)
plt.show()


# We can see here score means of country's top 5 schools. Which is just look like our expectations
