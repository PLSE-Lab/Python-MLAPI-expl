#!/usr/bin/env python
# coding: utf-8

# > **Aim- Exploratory Data Analysis of Chocolate Bar Rating.**
# 

# In[ ]:


#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Reading file
choco=pd.read_csv('../input/flavors_of_cacao.csv');


# In[ ]:


#getting the head of data set
choco.head()


# In[ ]:


#Information about data set
choco.info()


# In[ ]:


#Slightly Modifying the columns name
choco.columns = choco.columns.str.replace("\\n","-").str.replace(" ","-").str.strip(" ")
choco.columns


# In[ ]:


#datatypes of features
choco.dtypes


# In[ ]:


#changing cocoa-percent data
choco['Cocoa-Percent'] = choco['Cocoa-Percent'].str.replace('%','').astype(float)/100
choco.head()


# In[ ]:


choco.columns


# In[ ]:


## Look at most frequent species
choco['Specific-Bean-Origin-or-Bar-Name'].value_counts().head(10)


# In[ ]:


choco['Company-Location'].value_counts().head(10)


# In[ ]:


#Data Visualization


# In[ ]:


## Distrubution of Rating
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choco['Rating'],ax=ax)
ax.set_title('Rating Distrubution')


# **Observation**
# Most of the chocolates have rating between 3-4, Very less number have rating above 4.

# In[ ]:


## Look at distribution of Cocoa %
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choco['Cocoa-Percent'], ax=ax)
ax.set_title('Cocoa %, Distribution')
plt.show()


# **Observation**
# Most of the chocolate has 70% of cocoa in them.

# **Bivariate Analysis**

# In[ ]:


choco.plot(kind='scatter', x='Rating', y='Cocoa-Percent') ;
plt.show()


# **Observation-**
# The chocolates with Cocoa-Percentage from 60-80% have good rating between 3 to 4.

# In[ ]:


choco.plot(kind='scatter',x='Rating',y='Review-Date')


# **Observation**
# The rating has  imporved with time.

# **Box Plots**

# In[ ]:


## Look at boxplot over the company location,
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choco,
    y='Company-Location',
    x='Rating'
)
ax.set_title('Boxplot, Rating for countries')


# **Observation**
# Countries like USA, CANADA, EQUADOR,AUSTRALIA,SCOTLAND,ICELAND seems to have good rating.
# 

# In[ ]:


## Look at rating by cocao-percent
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choco,
    y='Cocoa-Percent',
    x='Rating'
)
ax.set_title('Boxplot, Rating by Cocao-Percent')


# **Observation**
# Chocolate with 100% cocoa has a worst rating.
# All the choclates have 70% above cocoa percent.
# Choclates with above 3 rating has cocoa percent from 70-75%.

# In[ ]:


## Let's define blend feature
choco['is_blend'] = np.where(
    np.logical_or(
        np.logical_or(choco['Bean-Type'].str.lower().str.contains(',|(blend)|;'),
                      choco['Company-Location'].str.len() == 1),
        choco['Company-Location'].str.lower().str.contains(',')
    )
    , 1
    , 0
)
## How many blends/pure cocoa?
choco['is_blend'].value_counts()


# **Observation**
# No of choclates which are pure is 102.
# and blend is 1693

# In[ ]:


## What better? Pure or blend?
fig, ax = plt.subplots(figsize=[6, 6])
sns.boxplot(
    data=choco,
    x='is_blend',
    y='Rating',
)
ax.set_title('Boxplot, Rating by Blend/Pure')


# **Observation**
# Pure choclate have worst rating.
# Blends have better rating.

# **Mean And Variance **

# In[ ]:


choco_best_beans = choco.groupby('Broad-Bean-Origin')['Rating']                         .aggregate(['mean', 'var', 'count'])                         .replace(np.NaN, 0)                         .sort_values(['mean', 'var'], ascending=[False, False])
choco_best_beans.head()


# **Observation**
# As we can see, the origins ranking first are only providing one kind of cocoa beans.

# In[ ]:


choco_best_beans = choco_best_beans.sort_values('count', ascending=False)[:20]                             .sort_values('mean', ascending=False)
choco_best_beans.head()


# **Observation**
# Gautemala have the best beans.

# In[ ]:


#Country producing best choclate-bar
choco_highest = choco.groupby('Company-Location')['Rating']                         .aggregate(['mean', 'var', 'count'])                         .replace(np.NaN, 0)                         .sort_values(['mean', 'var'], ascending=[False, False])
choco_highest.head()


# **Observation**
# We can't decide only with 4 counts.

# In[ ]:


choco_highest = choco_highest.sort_values('count', ascending=False)[:20]             .sort_values('mean', ascending=False)
    
choco_highest.head()


# **Observation**
# 1. Brazil seems to be best chocolate producer.

# **CONCLUSION**
# 
# * Most of the choclates have rating between 3-4, Very less numbers have rating above 4.
# * Brazil is the best choclate producer.
# * Gautemala have the best beans.
# * Pure choclate have worst rating.
# * Blends have better rating.
# * All the choclates have 70% above cocoa present in them.
# * Choclates with above 3 rating has cocoa percent from 70-75%.
# * Countries like USA, CANADA, EQUADOR,AUSTRALIA,SCOTLAND,ICELAND seems to have good rating.
# * The rating has  imporved with time, from 2006 the rating are only bad to 2016 the ratings are mixed.
# * The choclates with Cocoa-Percentage from 60-80% have good rating between 3 to 4.

# In[ ]:




