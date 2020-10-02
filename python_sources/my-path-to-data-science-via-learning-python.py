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

print("Welcome")
# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/data.csv")
print("Lets start")


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.plot(kind='scatter', x='Agility', y='Finishing',alpha = 0.5, color = 'green')
plt.xlabel('Agility')        
plt.ylabel('Finishing')
plt.title('Agility - GOAL Scatter Plot')  


# In[ ]:


x = data[(data["Finishing"] >85) & (data["Penalties"] > 85)]
print(x.Name)


# Let's visualize our data 
# 
# I am gonna show top 10 teams via bar plot in seaborn

# In[ ]:


data.info()


# In[ ]:


club = list(data['Club'].unique())


# In[ ]:


team_power = []
for i in club:
    a = data[data["Club"] == i]
    power = sum(a.Overall)/len(x)
    club.append(power)

clublist = pd.DataFrame({"club": club, "power": team_power})
new_index = (data["power"].sort_values(ascending = False)).index.values
sorted_data = clublist.reindex(new_index)


plt.figure(figsize = (15,10))
sns.barplot(x=sorted_data["club"], y = sorted_data["power"])
plt.xticks(rotation = 45)

