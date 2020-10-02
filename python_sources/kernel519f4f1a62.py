#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


t=pd.read_csv('/kaggle/input/titanicdataset/Titanic1.csv')


# reading the dataset

# In[ ]:


t.head()


# > printing the first five entries in the data set to understand the data

# In[ ]:


t=t.drop(columns=['Cabin'])


# > cabin has no value in analysis ,so its better to drop the column 

# In[ ]:


t.info()


# > info gives the basic information about the dataset
# 
# > it provides information like data type ,number of non-null variables in the dataset
# 

# In[ ]:


t.shape


# > .shape gives the total rows and columns in the dataframe

# In[ ]:


t['SibSp'].value_counts()


# > first I would like to know how the values of sibsp(siblings/spouse) data is distributed

# In[ ]:


t['Parch'].value_counts()


# > Now I would like to know how the parch(parent/children) value is distributed

# In[ ]:


t['Survived'].mean()


# > As the analysis is on the survival pattern of the passengers travelled in titanic , i would like to see the mean survival

# In[ ]:


t.groupby('Pclass')['Survived'].value_counts()


# > this will give us the number of people survived and lost their life according to their class
# 
# > this won't be a good estimator as the number of people travelled in each class are different

# In[ ]:


t.groupby('Pclass')['Survived'].mean()


# > groupby and .mean() helps us to view the mean suvival rate according to the class in which they travelled
# 
# > here we can see that 43 % of the passengers travelled in 1st class have survived
# 
# > And only 28% of the passengers from 3rd have survived

# In[ ]:


t.groupby('Sex')['Survived'].mean()


# > did most of the survived people are male??
# 
# > to answer this question when we group gender and survival ,more or less male and female are equally survived 

# In[ ]:


t.groupby(['Pclass','Sex'])['Survived'].mean()


# > it will be intresting to see the class in which the survived male and female are travelling
# 
# > it surprisingly only 24% and 29% of female and male are survived in 3rd class, whereas in 1st class was 42% and 42% respectively

# In[ ]:


t.groupby('Parch')['Survived'].mean()


# > this is find whelther the people travelled with their parents of chlidren survived the most
# 
# > 30% of people who travelled without any parent or children survived 
# 
# > 50% of the people who travelled with 6 or 9 of their parent/children had survived,this may be highly miss leading as may be only few passengers must have travelled in such a condition 

# In[ ]:


t['Age'].describe()


# > describe gives us the basic statistical measures for the respected column
# 
# > from this we can find the minimum, maximum , mean of the age

# In[ ]:


t[t.Fare==0].describe()


# > i found few passengers fare was extremely zero
# 
# > syntax displays the passangers details who's fare is zero
# 
# > by external source I found that those passengers are workers and they are related to titanic ship 

# In[ ]:


t['relatives']=t['Parch']+t['SibSp']


# >  Now im creating a new column so that all the parents/children and siblings/spouse will be taken into count as relatives, which wwill be usefull for analysis

# In[ ]:


t['relatives'].min()


# > we found that there are passengers travelling without any relatives

# In[ ]:


t['relatives'].max()


# > there are also passengers travelling with 10 of their relatives

# In[ ]:


t[t.relatives==10]


# > when we loook of the passenger travelled with his 10 other relatives
# 
# > it was only one passenger along with 10 other relatives.also their seat no are continous
# 

# In[ ]:


t['family']=[1 if x>0 else 0for x in t['relatives']]


# > here we are creating another column named family, if the passenger has relatives then he is marked as 1 else marked as 0

# In[ ]:


t['family']


# In[ ]:


t.isnull().sum()


# >isnull().sum() gives us the sum of null variables in the dataframe

# In[ ]:


t['age_group']=['<13' if x <=13 else '13-18' if x <18 else '18-40' if x  <40 else '40-60' if x  <60 else '60-80' if x  <=80 else '80+'if x >80 else "unknown" for x in t['Age']]


# > since the age is continous we could not make the maximum out of it , so I will bin the age
# 
# > <13,18-40,40-60,60-80 are the limit age are recorded

# In[ ]:


t['age_group'].value_counts()


# In[ ]:


pd.crosstab(t.Pclass,t.age_group)


# > crosstab provides the tabular respresentation of agegroup vs class they travelled

# In[ ]:


f=t[['Pclass','age_group']][t.Survived==0]


# > this saves class and age details of the passengers who didnot surrvive 

# In[ ]:


pd.crosstab(f.Pclass,f.age_group)


# In[ ]:


e=t[['Pclass','age_group']][t.Survived==1]


# > saves class and age of the passengers who survived the crisis

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.xlabel('Age')
plt.hist(t['Age'])
plt.show()


# > this shows the distribution of the age of the passengers travelled

# In[ ]:


plt.xlabel('Fare')
plt.hist(t['Fare'])
plt.show()


# > graph shows the fare price given by the passengers

# In[ ]:


plt.xlabel('Pclass')
plt.hist(t['Pclass'],color='pink')
plt.show()


# >graphical representation of number of passangers travelled in three classes

# In[ ]:


fig =plt.figure(figsize=(9,9))
ax1 =fig.add_subplot(221)
plt.xlabel('Age')
plt.hist(t['Age'])
ax2=fig.add_subplot(222)
plt.hist(t['Fare'],color='r')
plt.suptitle('two subplots',size=20)
plt.tight_layout(pad=4)
plt.show()


# > this displays two subplots one histogram of age and the other one is histogram of fare  

# In[ ]:


t.boxplot(column=['Age'],grid=False)


# > box plot is use to find whelther is any outliers in the data
# 

# In[ ]:


sns.countplot(x='Survived',data=t)


# In[ ]:


sns.catplot(x='Pclass',data=t,hue='Sex',kind='count')


# In[ ]:


sns.catplot(x='Pclass',data=t,hue='Sex',col='Survived',kind='count')


# In[ ]:


sns.catplot(x='Pclass',data=t,hue='age_group',kind='count')


# In[ ]:


sns.catplot(x='Pclass',data=t,y='Age',kind='box')


# > box plot according to class in which they travelled

# In[ ]:


sns.catplot(x='Pclass',data=t,y='Fare',kind='box')
sns.catplot(x='Pclass',data=t,y='Fare',hue='Sex',kind='box')


# In[ ]:


sns.catplot(x='Pclass',data=t,y='Fare',hue='Sex',kind='box')

