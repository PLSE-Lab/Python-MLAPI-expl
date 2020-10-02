#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization tool
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/2017.csv") # reading csv data through pandas


# In[ ]:


data.info() # columns,count,data types


# In[ ]:


data.columns # columns's names


# In[ ]:


data.rename(columns={"Country": "Country", "Happiness.Rank":"HappinessRank","Happiness.Score":"HappinessScore",
                        "Health..Life.Expectancy.":"HealthLifeExpectancy","Whisker.high":"WhiskerHigh","Whisker.low":"WhiskerLow",
                        "Economy..GDP.per.Capita.":"EconomyPerCapital","Freedom":"Freedom",
                        "Generosity":"Generosity","Trust..Government.Corruption.":"TrustGovernmentCorruption",
                        "Dystopia.Residual":"DystopiaResidual"},inplace = True)
# We are using inplace=True to change column names in place.
data.columns # final form of columns's names


# In[ ]:


data.head(10) # shows first 10 rows 


# In[ ]:


data.describe() # shows values such as max,min,mean (for numeric feature)


# In[ ]:


data.corr() # relevance between columns


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


data.EconomyPerCapital.plot(kind="line",color="green",label="EconomyPerCapital",linewidth=1,alpha=0.8,grid=True,linestyle=':',figsize = (8,8))
data.HealthLifeExpectancy.plot(kind="line",color="blue",label="HealthLifeExpectancy",linewidth=1,alpha=0.8,grid=True,linestyle='-.',figsize = (8,8))
plt.legend(loc='upper left')
plt.title('Line Plot') 
plt.xlabel("Country")
plt.ylabel("Variables")
plt.show()


# ****The relationship between Economy per Capita - Health Life Expectancy****

# In[ ]:


data.plot(kind="scatter",x="EconomyPerCapital",y="HealthLifeExpectancy",alpha=0.5,color="r",figsize = (6,6))
plt.title('Scatter Plot') 
plt.xlabel("Economy Per Capital")
plt.ylabel("Health Life Expectancy")
plt.show()


# In[ ]:


data.Freedom.plot(kind='hist',bins=60,grid=True,figsize=(10,10)) 
plt.title('Histogram Plot') 
plt.xlabel("Freedom")
plt.show()


# ****Freedom in countries****

# In[ ]:


NewData = data[(data.Freedom > 0.5) & (data.TrustGovernmentCorruption > 0.4)] # used for filtering
NewData


# ****You can filter as you like with this way.****
