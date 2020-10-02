#!/usr/bin/env python
# coding: utf-8

# **SOME BASIC PLOTS USING PANDAS AND SEABORN PACKAGE**
# 
# - Example using Titanic data
# - Good reference for data visualization:
# https://www.kaggle.com/residentmario/welcome-to-data-visualization

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns #(pip install seaborn)
import matplotlib.pyplot as plt
#for the IPython notebook is used
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Using the Titanic training data as an example
df1 = pd.read_csv('../input/titanic/train.csv')
print(df1.shape)
print('All columns:', df1.columns)


# **1. DATA VISUALIZATION WITH PANDAS**
# 
# **1.1. Explore single feature (column) (unvariate plots) **
# 
# Line plot, area plot, bar plot, histogram

# In[ ]:


df1['Fare'].value_counts().sort_index()


# In[ ]:


# line plot
fig = df1['Fare'].value_counts().sort_index().plot.line()
fig.set_title('Distribution of Fare', fontsize = 15)
fig.set_xlabel('Fare values')
fig.set_ylabel('Count')


# In[ ]:


# area plot
fig = df1['Fare'].value_counts().sort_index().plot.area() #default bin=10
fig.set_title('Distribution of Fare', fontsize = 15)


# In[ ]:


# bar plot
fig = df1['Fare'].value_counts().sort_index().plot.bar(stacked=True) #default bin=10
fig.set_title('Distribution of Fare', fontsize = 15)


# In[ ]:


# histogram plot
fig = df1['Fare'].plot.hist() #default bin=10
fig.set_title('Distribution of Fare', fontsize = 15)


# In[ ]:


# histogram plot with cutoff (remove outliners)
fig = df1[df1['Fare']<300]['Fare'].plot.hist() #default bin=10
fig.set_title('Distribution of Fare', fontsize = 15)


# **1.2. Explore the relationship among features (columns) (bivariate plot)**

# In[ ]:


# scatter plot
fig = df1.plot.scatter(x='Fare',y='Survived')
fig.set_title('Distribution of Fare', fontsize = 15)


# In[ ]:


# hex plot
fig = df1[df1['Fare']<50].plot.hexbin(x='Fare',y='Age')
fig.set_title('Distribution of Fare', fontsize = 15)
fig.set_xlabel('Fare values')
fig.set_ylabel('Age')


# **2. DATA VISUALIZATION WITH SEABORN**
# 
# **2.1. Explore single feature (column)**

# In[ ]:


sns.countplot(df1['Fare'])


# In[ ]:


#kde plot
#"kernel density estimate"
#a statistical technique for smoothing out data noise
sns.kdeplot(df1.Fare)


# In[ ]:


#kde plot with cutoff
sns.kdeplot(df1.query('Fare<50').Fare)


# **2.2. Explore the relationship among features (columns) (bivariate plot)**

# In[ ]:


#bar plot
sns.barplot(x='Embarked', y='Survived', hue='Sex', data=df1)


# In[ ]:


# # pointplot - simple
sns.pointplot(x='Embarked', y='Survived', hue='Sex', data=df1)


# In[ ]:


# pointplot - customized
sns.pointplot(x='Embarked', y='Survived', hue='Sex', data=df1,
    palette = {'male':'blue', 'female': 'pink'},
    markers=['*','o'], linestyles=['-','--'])


# In[ ]:


#joinplot
sns.jointplot(x='Fare', y='Age', data=df1[df1['Fare']<50])


# In[ ]:


#joinplot
sns.jointplot(x='Fare', y='Age', data=df1[df1['Fare']<50], kind='hex',
             gridsize=20)

