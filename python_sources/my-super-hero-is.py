#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


info = pd.read_csv('../input/heroes_information.csv')
info.head()


# In[ ]:


powers = pd.read_csv('../input/super_hero_powers.csv')
powers.head()


# In[ ]:


info.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


info.isnull().sum()


# In[ ]:


info.Publisher.fillna('other',inplace=True)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls


# In[ ]:


temp_series = info['Publisher'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Comic-wise Superheroes distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Comic")


# In[ ]:


powers = powers*1
powers.head() #this is coverting boolean values to binary values like 1 and 0


# In[ ]:


powers.loc[:, 'no_of_powers'] = powers.iloc[:, 1:].sum(axis=1)
#Here we just sum all the powers and added new column here at the end


# In[ ]:


powers.head()#check the last column


# In[ ]:


super_powers = powers[['hero_names','no_of_powers']]
super_powers.head()


# In[ ]:


super_powers['no_of_powers'].max()


# In[ ]:


super_powers = super_powers.sort_values('no_of_powers',ascending=False)
super_powers.head()


# In[ ]:


powers.shape


# In[ ]:


info.Gender.value_counts()


# In[ ]:


sns.countplot(info.Gender.values) #this is much easier


# In[ ]:


plt.figure(figsize=(25,5))
sns.countplot(info.Height.values)
plt.show()


# the above one is not good for this height data, we will use boxplot then, and if you observe pink line indicates null values.

# In[ ]:


plt.figure(figsize=(15,5))
sns.boxplot(x="Gender", y="Height", data=info)
plt.ylabel('Height Distribution (cm.)', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.title("Height Distribution by Gender", fontsize=14)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="Gender", y="Weight", data=info)
plt.ylabel('Weight Distribution (kg.)', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.title("Weight Distribution by Gender", fontsize=14)
plt.xticks(rotation='vertical')
plt.show()


# I hope this is super easy notebook for beginners, if you like it please upvote for me. 
# 
# Thank you : )
