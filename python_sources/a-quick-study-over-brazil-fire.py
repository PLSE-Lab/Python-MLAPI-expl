#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# We will explore the fire rate in Brazil in the last 20 years and which states are the responsible for most of the fires.
# 

# In[ ]:


df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',  encoding='ISO-8859-1')


# In[ ]:


''' How much empty columns do we have'''
df.isna().sum()


# In[ ]:


''' Unique states that has forest fire'''
states = df.state.unique()
print(f"States: {states}")
print(f"Numb: {len(states)}")


# Since brazil has 26 states and 1 federal district, we haven't data for all the states splited.
# <p> In this data set, we have Mato Grosso and Mato Grosso do Sul together. 
# <p> And Rio de Janeiro, Rio Grande do Sul e Rio Grande do Norte together.

# In[ ]:


''' Get the states with more fire '''
top_10_states = df.groupby(['state'])['number'].sum().sort_values(ascending=False).head(10).keys().tolist()
top_10_values = df.groupby(['state'])['number'].sum().sort_values(ascending=False).head(10)


# In[ ]:


print(f"TOP 10 STATE \t VALUE")
print(top_10_values)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


all_years_data = df.groupby('year').sum()
fig, ax = plt.subplots(figsize=(8,8))
plot = sns.lineplot(data=all_years_data, markers=True)
plot.set_title("Fire per year in Brazil")
plot.set(xlabel='Year', ylabel='No. of Fires')


# Saddly we can see an increase in the number in the last 20 years.
# <p> We can search what states has the most mire and explore it further

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.barplot(x=top_10_states, y= top_10_values, palette='Accent')
plot.set(xlabel= 'State', ylabel='Number of fire occurence', title='Top 10 States with most fire occurence ')


# <h4> Exploring Mato Grosso(South + North) Data</h4>

# In[ ]:


''' Lets explore when the fire occour more on Mato Grosso '''
mato_grosso= df[df.state == 'Mato Grosso']
fire_sum_per_month = mato_grosso.groupby(['month'])['number'].sum().sort_values(ascending=False)
month = mato_grosso.groupby(['month'])['number'].sum().sort_values(ascending=False).keys().tolist()


# In[ ]:


''' Exploring mato grosso data by month'''
explode = len(month) * [0]
explode[0] = 0.1  ##Get the biggest falue and explode it 
fig, ax = plt.subplots( figsize= (8,8))
ax.pie(fire_sum_per_month, labels=month, autopct='%1.1f%%', startangle=145, explode=explode)
plt.tight_layout()
plt.show()


# In[ ]:


all_years_data = mato_grosso.groupby('year').sum()
fig, ax = plt.subplots(figsize=(8,8))
plot = sns.lineplot(data=all_years_data, markers=True)
plot.set_title("Fire per year in Mato Grosso")
plot.set(xlabel='Year', ylabel='No. of Fires')


# Let's explore the evolution of fire over the years on Setembro.

# In[ ]:


setember = mato_grosso[mato_grosso.month == 'Setembro']
fig, ax = plt.subplots(figsize=(8,8))
plot = sns.lineplot(x='year', y='number', data=setember, markers=True)
plot.set_title("Setember fires over the Years in Mato Grosso")
plot.set(xlabel='YEAR', ylabel='No. of Fires')
print(f"Max: {mato_grosso.number.max()}")
print(f"Min: {mato_grosso.number.min()}")
print(f"Mean: {mato_grosso.number.mean()}")
plt.show()


# **Why Mato Grosso?**
# <p> As we can see, Brazil fire has been increasing over the last 20 Years. Mostly due to Mato Grosso. 
# <p> We have dat for Mato Grosso do Sul (south Mato Grosso) and Mato Grosso do Norte (Noth mato Grosso), they're side by side on Brazil territorie and had an increasing in population in 20th century, due to timber and aggricultural culture. 
# <p> 40% of Mato Grosso Economy is based on Aggricultural, mostly, due to large areas of florest part of Pantanal.
# <p> Mato Grosso fire can be due a lot of factores, Aggricultural, High Fishing and camping activite and Livestock.
# <p> Mato Grosso is also an region where we have fight between locals and aggriculture.
#     
#     
# 

# In[ ]:




