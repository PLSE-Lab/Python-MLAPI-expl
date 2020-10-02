#!/usr/bin/env python
# coding: utf-8

# # This tutorial deals with Titanic data exploration and Visualization. 
# 
# ## I have simply tried to explain some unique and cool features provided by some of the most powerful libraries of python such as pandas,seaborn etc. 
# 
# ## There is no trick or tips to learn data visualization, It simply depends on your mindset and imagination. The more you practice the more you become master in using right function at right place. 
# 
# ## And yeah even I am not a master, just a beginner like most of you are but again everything comes from pratice. So just keep going and upvote if you like this tutorial. 

# In[ ]:


# libraries required for handling and visulaising data

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read_csv() -> to read csv files (provided by pandas library)

train_data=pd.read_csv('../input/titanic/train.csv')
train_data.head() # head function prints top 5 rows of dataset if parameter is not given 


# In[ ]:


# describe function -> calculates basic mathematical operations such as mean, min, max etc  
train_data.describe()


# 
# Seaborn is the most powerful and efficient library used for data visualization .
# 
# For example if we want to visulize the null columns present in our data set we can simply plot heatmap as given below: 
# 

# In[ ]:


# matplotlib deals with plotting figures and graphs 
plt.figure(figsize=(12,6)) # figsize -> defines size of the figure 

sns.heatmap(train_data.isnull(),yticklabels=False,cmap='viridis') #to check null values


# Countplot is another cool feature provided by seaborn which counts values for a particular column based on class.
# For example if we simply want to check number of survived passenger we can plot it as: 

# In[ ]:


sns.countplot(x='Survived',data=train_data)  


# In[ ]:


'''there can be a 'hue' factor associated with countplot, for example if we want to check number of survived passanger 
based on sex we can use hue=sex '''

sns.countplot(x='Survived',data=train_data, hue='Sex')  


# Distribution plot shows the overall distibution of certain parameters. For example if we want to check the age group of dataset we can do it as:  

# In[ ]:


sns.distplot(train_data['Age'].dropna()) # most of them were between 20-40

# distribution plot also shows a kde(kernel density estimation) line (slightly complex concept).


# In[ ]:


# histogram plot 
train_data['Fare'].hist(bins=30) # most of them have bought cheaper tickets


# # This was a very very basic data visualization tutorial. 
# ## I just wanted to show that how can you visualize data based on your requirement. Every dataset may require different kind of plots and functions for exploration. So just keep practicing.

# In[ ]:




