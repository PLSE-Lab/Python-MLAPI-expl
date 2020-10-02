#!/usr/bin/env python
# coding: utf-8

# This is an initial analysis focusing on:
# 
# * Year
# * Victim and Perpetrator Ages
# * Explorations around the Weapons
# * Visualisations
# 
# Please upvote If you liked this as I've just started with Data Analysis using Python and i'm excited to learn more and improve

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/database.csv')


# In[ ]:





# Time to drop all the irrelevant columns

# In[ ]:


df.drop(['Record ID','Agency Code','Agency Name','Agency Type','Incident','Victim Ethnicity','Perpetrator Ethnicity', 'Perpetrator Count','Record Source'], axis=1, inplace=True)


# In[ ]:


df.columns


# Let's make the crime type be only murder or manslaughter

# In[ ]:


df['Crime Type'].value_counts()


# In[ ]:


df=df[df['Crime Type']=='Murder or Manslaughter']


# In[ ]:


df['Crime Type'].value_counts()


# In[ ]:





# Let's plot the number of crimes per year, but before that, lets do some analysis

# The year where the most murders were recorded

# In[ ]:


print('The Year with the most murders was', df['Year'].value_counts().index[0], ' with a number:',df['Year'].value_counts().max())


# In[ ]:


print('The year with the least number was',df['Year'].value_counts().index[34], 'with the number:', df['Year'].value_counts().min())


# In[ ]:


sns.distplot(df['Year'], kde=False)


# Now lets examine the murders along state lines:

# In[ ]:


df['State'].unique()


# In[ ]:


print('The state with the most murders was', df['State'].value_counts().index[0],'with a number of:',df['State'].value_counts().max())


# In[ ]:


print('The state with the least murders was', df['State'].value_counts().index[50],'with a number of:',df['State'].value_counts().min())


# In[ ]:


#plot descending bar graph to show this
#sns.countplot(x=df['State'],data=df)


# Time to analyse the ages of the victim and perps. As we can see the below the maximum age of the victim is 998 which is imposible, so now we have to clean the data.

# In[ ]:


#clean victim max age and say max mean and average (then dictplot)
df['Victim Age'].max()


# In[ ]:


df=df[df['Victim Age']<90]


# In[ ]:


print('Males were',df['Victim Sex'].value_counts(normalize=True).apply(lambda x: x*100)['Male'],'percent of the victims\n'
      'Females were', df['Victim Sex'].value_counts(normalize=True).apply(lambda x: x*100)['Female'], 'percent of the victims\n'
      'The rest are unknown')


# In[ ]:


print('The average age of the victims was', df['Victim Age'].median())


# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df['Victim Age'], kde=False)


# Time to find out the average age of the perps

# In[ ]:


df['Perpetrator Age'] = pd.to_numeric(df['Perpetrator Age'], errors='coerce')


# In[ ]:


df=df[(df['Perpetrator Age']<90) & (df['Perpetrator Age']>0)]


# In[ ]:


df['Perpetrator Age'].max()


# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df['Perpetrator Age'])


# In[ ]:


#To find the most occuring perp age
df['Perpetrator Age'].mode()


# In[ ]:


#scatter plot of vic and perp ages
#sns.jointplot(x='Victim Age',y='Perpetrator Age',data=df, kind='reg')


# In[ ]:


df['Weapon'].value_counts().max()


# Weapons

# In[ ]:


#The list of popular weapons used
df['Weapon'].value_counts()


# In[ ]:


print('The most popular weapon used was the:',df['Weapon'].value_counts().index[0],'\naccounting for about:',df['Weapon'].value_counts().max(),'occurences')


# Weapons and Gender

# In[ ]:


#weapon and perp gender
plt.figure(figsize=(12, 5))
g=sns.countplot(x='Weapon', data=df, order=df['Weapon'].value_counts().index,hue='Perpetrator Sex')
for item in g.get_xticklabels():
    item.set_rotation(90)


# Weapons and Race

# In[ ]:


#weapon and perp gender
plt.figure(figsize=(12,5))
r=sns.countplot(x='Weapon', data=df, order=df['Weapon'].value_counts().index,hue='Perpetrator Race')
for item in r.get_xticklabels():
    item.set_rotation(90)


# Weapon by Relationships

# In[ ]:


plt.figure(figsize=(14,5))
w=sns.countplot(x='Relationship',data=df, order=df['Relationship'].value_counts().index[:11], hue='Weapon')
for item in w.get_xticklabels():
    item.set_rotation(90)
    

