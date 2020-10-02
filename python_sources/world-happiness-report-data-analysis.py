#!/usr/bin/env python
# coding: utf-8

# > **World Happiness Report ** 
# > data analysis

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#First is first, Let's import the dataset
data = pd.read_csv('../input/2017.csv')


# In[ ]:


#Let's see what we have in this data set and tops
data.head(10)


# In[ ]:


#Columns has to be renamed. Having 'dot' in the names of the features might create problems later. 
data = data.rename(columns={"Happiness.Score":"HappinessScore", "Happiness.Rank":"HappinessRank",
                           "Whisker.high":"Whisker_high","Whisker.low":"Whisker_low",
                           "Economy..GDP.per.Capita.":"Economy_GDPperCapita", 
                            "Health..Life.Expectancy.": "HealthLifeExpectancy",
                           "Trust..Government.Corruption.":"TrustGovernmentCorruption",
                           "Dystopia.Residual":"DystopiaResidual"})


# In[ ]:


#Last 10
data.tail(10)


# In[ ]:


#Let's see what features we have only in titles
data.columns


# In[ ]:


#General information abaout data
data.info()


# In[ ]:


#Let's see some  basic statistics about the data
data.describe()


# 
# 
# > Now, we know what sort of data we have.
# > To have an insight about the data itself let's see how it changes and the relation between features. 
# 
# 

# In[ ]:


# Let's see the correlation betwen features.
data.corr()


# In[ ]:


#To see this correlations in a better visulised form we will use sns.heatmap can take all parameters:

#seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, 
#fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, 
#cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)

f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, fmt ='.2f', linewidths=1, ax=ax)

plt.show()


# In[ ]:


#Lets' see the relation between HappinessScore and Economy_GDPperCapita on line plot
data.HappinessScore.plot( kind='line', color= 'r', label = 'HappinessScore', linewidth=1, alpha = 1,
                         grid = True, linestyle = ':')
data.Economy_GDPperCapita.plot( kind='line', color= 'b', label = 'Economy_GDPperCapita', linewidth=1, alpha = 1,
                         grid = True, linestyle = ':')
plt.legend(loc="upper right")

plt.show()


# In[ ]:


#scatter plot
data.plot(kind= 'scatter', x='Economy_GDPperCapita', y='HealthLifeExpectancy', color= 'b', 
          grid=True, alpha=0.5 )
plt.show()


# In[ ]:


#histogram
data.HappinessScore.plot(kind='hist', color='y',bins=25, alpha=1, grid=True, figsize=(7,7))
plt.xlabel('HappinessScore')
plt.show()


# 
# > **PANDAS**
# > 
# > Filtering 
# 

# In[ ]:


#Lets list countries that Happiness Score is higher than the 7
x=data['HappinessScore']>7
data[x]


# In[ ]:


#countries that Happiness Score is higher than the 7 & life expectancy is lower than 0.7
data[(data['HappinessScore']>7)&(data['HealthLifeExpectancy']<0.8)]


# In[ ]:


#Let's find the data of Turkey only
data[data['Country'] == 'Turkey']


# In[ ]:





# In[ ]:




