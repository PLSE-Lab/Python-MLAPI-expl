#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# My plan for this study 
# * Analysing Data
# * Making Visualisations
# * Making a simple machine learning prediction application

# In[ ]:


data = pd.read_csv('../input/Pokemon.csv',index_col=0)


# In[ ]:


data.info()


# In[ ]:


data.corr()
# We can easily see that attack and speed attack has a good correlation with total


# In[ ]:


data.head(10)


# **Seaborn**
# 
# We will start visualisation with Seaborn

# In[ ]:


#Bar Plot for Attack
plt.figure(figsize=(15,10))
sns.barplot(x=data['Type 1'],y=data['Attack'])
plt.xticks(rotation= 45)
plt.xlabel('Type 1')
plt.ylabel('Attack')
plt.title('Attack by Types')


# In[ ]:


# Barplot for HP

plt.figure(figsize=(15,10))
sns.barplot(x=data['Type 1'],y=data['Speed'])
plt.xlabel('Type 1')
plt.ylabel('Speed')
plt.xticks(rotation=45)
plt.title('HP by Types')


# In[ ]:


#Point Plot

f,ax1=plt.subplots(figsize=(20,10))
sns.pointplot(x=data['Type 1'],y=data['Attack'],data=data,color='green',alpha=0.3)
sns.pointplot(x=data['Type 1'],y=data['HP'],data=data,color='red',alpha=0.3)
plt.text(1,120,"Attack by types",color='green',fontsize=17,style='italic')
plt.text(1,117,"HP by types",color='red',fontsize=17,style='italic')
plt.title("Attack and HP value by Their Types",fontsize=17,color='blue')
plt.grid()


# In[ ]:


#Joint plot
# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables
# Show the joint distribution using kernel density estimation 

g = sns.jointplot(data.Attack,data.Total,kind='kde',size=7)
plt.savefig('Pokemongrapg.png')
plt.show()


# In[ ]:


sns.jointplot(data.Speed,data.Total,kind='kde',size=7)
plt.savefig('Pokemongrapg3.png')
plt.show()


# In[ ]:


data.head()


# In[ ]:


#Scatter plot 
#x=Types, y=Total

data.plot(kind='Scatter',x='Speed',y='Attack',alpha=0.3,color='green')
plt.xlabel('Speed')
plt.ylabel('Attack')
plt.title('Attack by Speeds')


# In[ ]:


#Simple Seaborn visualisation with only one line of code

sns.countplot(data['Attack'])


# In[ ]:


sns.countplot(data['Type 1'])


# In[ ]:


sns.kdeplot(data.query('Attack > 160').Attack)
#Nice tool for deal with outliers

