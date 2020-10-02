#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as p # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math as mt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Analysis of the Adult data set from Pyton**
# 
# This data set is a dual class classification to estimate that one's income does not exceed 50,000 per year based on some census data.
# Our goal here is to keep and improve the knowledge I have learned in Data scientist courses.

# **Attribute Information:**
# 
# Listing of attributes:
# 
# >50K, <=50K.
# 
#     age: continuous.
#     workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#     fnlwgt: continuous.
#     education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#     education-num: continuous.
#     marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#     occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#     relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#     race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#     sex: Female, Male.
#     capital-gain: continuous.
#     capital-loss: continuous.
#     hours-per-week: continuous.
#     native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# First we check the library versions we have used

# In[ ]:


p.__version__ #Which Version Pandas we use


# In[ ]:


np.__version__


# In[ ]:


sns.__version__


# Then we load our data set

# In[ ]:


df=p.read_csv("../input/adult.csv")#Import Adult Data


# We then check our data set (record, info, column based)

# In[ ]:


df.info()       # memory footprint and datatypes


# In[ ]:


df.all()


# In[ ]:


df.head()       # first five rows


# In[ ]:


df.tail()       # last five rows


# In[ ]:


df.describe()   # calculates measures of central tendency


# In[ ]:


df.sample(5)    # random sample of rows


# In[ ]:


df.shape        # number of rows/columns in a tuple


# We separate our  data gender-based  and  checks data again

# In[ ]:


Male=(df[df.sex==" Male"]) # Filter By sex columns for Male


# In[ ]:


Male.sex.unique()


# In[ ]:


Female=(df[df.sex==" Female"]) # Filter By sex columns for Female


# In[ ]:


Female.sex.unique()


# In[ ]:


df.sex.unique()


# In[ ]:


df.columns


# In[ ]:


df.groupby('race').count() # group by race all data 


# We can begin visual analysis of our Data Set

# We first cycle through the data coming from the columns of our data set and look at the different data.

# In[ ]:


fig = plt.figure(figsize=(20,15))
cols = 5
rows = mt.ceil(float(df.shape[1]) / cols)
for i, column in enumerate(df.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if df.dtypes[column] == np.object:
        df[column].value_counts().plot(kind="bar", axes=ax)
    else:
        df[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)


# At first we are evaluating weekly earnings visually.

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df.capital_gain.plot(kind = 'line', color = 'g',label = 'capital_gain',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.capital_loss.plot(color = 'r',label = 'hours_per_week',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# We assess weekly earnings visually differently.

# In[ ]:


# Scatter Plot 
# x = capital_gain, y = hours_per_week
df.plot(kind='scatter', x='capital_gain', y='hours_per_week',alpha = 0.5,color = 'red')
plt.xlabel('capital_gain')              # label = name of label
plt.ylabel('hours_per_week')
plt.title('capital_gain-hours_per_week Scatter Plot')            # title = title of plot
plt.show()


# We are looking at the links of the fields of female data 

# In[ ]:



#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(Female.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# When we want to examine the data in our dataset by age, a chart like the one below is born.

# In[ ]:


# Histogram
# bins = number of bar in figure
df.age.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:




