#!/usr/bin/env python
# coding: utf-8

# I hope you find this kernel helpful... and some **UPVOTES** would be very much appreciated!

# **Questions to answer :**
# 
# 1. How to imporve the students performance in each test ?
# 2. What are the major factors influencing the test scores ?
# 3. Effectiveness of test preparation course?
# 
# What patterns and interactions in the data can you find? Let me know in the comments section below.
# 

# **Assertion #1**
# 
# Those who receive reduced/free lunch are economically disadvantages and can be detected by the parents education level
# 
# **Outcome**
# 
# This assertion does not hold up. Data shows that students of college and above educated parents receive free/reduced lunches at roughly the same rate as standard lunches; we should have seen that these students nearly all had standard lunchs for this assertion to hold true.
# 
# 
# **Assertion #2**
# 
# Males/females should score in general the same across all scores
# 
# **Outcome**
# 
# Female dominate in reading and writing. Male have a slight advantage in math.
# 
# 
# **Assertion #3**
# 
# There is no difference between the meals on whether one pays full price or reduced/free.
# 
# **Assertion #4**
# 
# Students who receive free/reduced lunches do so because they do not have the funds to afford full rate.
# 

# **Answers**
# 
# Q1: Below distribution curves for **test_prep** show a significant improvement in scores in reading and writing and a slight improvement in math; it's not strongly correlated with any other feature. Given all the other features are immutable attributes to at least the student,  **test_prep** is the best way to improve a students scores. 
# 
# Based on the data below one might argue that stoping the free/lunches to all standard would make the scores go up. This is obviously not true (with assertion #3), without this program they would not be able to afford lunch and their scores would definitely go down being hungry.
# 
# **key finding:  although there was no relationship between lunch and education levels of parents. There was a finding that those who received free/reduced lunches were less likely to get test prep courses. Given test prep did improve scores significantly, a pilot test program should be enacted to offer test_prep to these students also at a free/reduced rate.**
# 
# Q2: Built a basic random forest regression model and exported the feature importance: 'gender': 0.11086688, 'race/ethnicity': 0.25042748, 'parental_level_of_education': 0.29552437, 'lunch': 0.18123146, 'test_preparation_course': 0.16194981]
# 
# Q3: **test_prep** is positively effective and really the only thing within this dataset that a student can do to improve their scores.
# 
# 
# 

# 
# 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


# I don't like spaces in my column names, lets fix that
df.columns = df.columns.map(lambda x: x.replace(' ', '_'))


# In[ ]:


#lets see the distinct values for each of the categorical columns
print(df['race/ethnicity'].unique().tolist())
print(df['parental_level_of_education'].unique().tolist())
print(df['lunch'].unique().tolist())
print(df['test_preparation_course'].unique().tolist())


# In[ ]:


# i don't like the education field, let's create some order to it. converting these over will make them play nicer with sns
mapper = {'some high school':0, 'high school':1, 'some college':2, "associate's degree":3, "bachelor's degree":4, "master's degree":5}
df['parental_level_of_education'] = df['parental_level_of_education'].map(mapper)

mapper = {'none':0, 'completed':1}
df['test_preparation_course'] = df['test_preparation_course'].map(mapper)

mapper = {'standard':1, 'free/reduced':0}
df['lunch'] = df['lunch'].map(mapper)

mapper = {'group B':1, 'group C':2, 'group A':0, 'group D':3, 'group E':4}
df['race/ethnicity'] = df['race/ethnicity'].map(mapper)

mapper = {'male':1, 'female':0}
df['gender'] = df['gender'].map(mapper)

df['avg_score'] = (df['math_score'] + df['reading_score'] + df['writing_score'])/3
df['avg_score_decile'] = (df['avg_score']/10).astype('int')


# In[ ]:


# example of some items
df.head()


# **EDA**

# In[ ]:


# check for missing values - there are no missing values
df[(df.isnull().any(axis=1)) | (df.isna().any(axis=1))]


# In[ ]:


# lets see the size of the dataset
df.shape


# In[ ]:


# get the general math stats
df.describe()


# In[ ]:


# lets see a pairplot for a high level view
sns.pairplot(df)


# In[ ]:


# notice the high correction between the three scores
sns.heatmap(df.corr(), cmap="Blues",annot=True,annot_kws={"size": 7.5},linewidths=.5)


# In[ ]:





# Observations...
# Histograms of scores are all three are slightly left skewed normally distributed. All plot diagrams of scores show a positive correlation with eachother.
# 
# From a high level point of view, **lunch** and **test_prep** appear the most influencial. Followed by **parents**, **gender**, **race** all being about equal.
# 
# Lets get a better insight into the histograms. High level view can be deceptive.

# In[ ]:


# scores
fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.countplot(x="math_score", data = df, palette="muted", ax=ax[0])
sns.countplot(x="reading_score", data = df, palette="muted", ax=ax[1])
sns.countplot(x="writing_score", data = df, palette="muted", ax=ax[2])
plt.show()


# Notice the 100% spike which is prominently on reading and writing? That is oddly characteristic I wouldn't have expected.

# In[ ]:


print('math', str(Counter(df[df.math_score > 90].math_score)))
print('reading', str(Counter(df[df.reading_score > 90].reading_score)))
print('writing', str(Counter(df[df.writing_score > 90].writing_score)))


# In[ ]:





# **Assertion #2**
# 
# Males/females should score in general the same across all scores
# 
# **Outcome**
# 
# women are blue, men are red. Women dominate in reading and writing. 

# In[ ]:


fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.distplot(df[df.gender == 0]['math_score'], ax=ax[0], color='blue')
sns.distplot(df[df.gender == 1]['math_score'], ax=ax[0], color='red')
sns.distplot(df[df.gender == 0]['reading_score'], ax=ax[1], color='blue')
sns.distplot(df[df.gender == 1]['reading_score'], ax=ax[1], color='red')
sns.distplot(df[df.gender == 0]['writing_score'], ax=ax[2], color='blue')
sns.distplot(df[df.gender == 1]['writing_score'], ax=ax[2], color='red')
plt.show()


# In[ ]:


fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.distplot(df[df.gender == 0]['math_score'],    ax=ax[0], color='blue')
sns.distplot(df[df.gender == 1]['math_score'],    ax=ax[0], color='red')
sns.distplot(df[df.gender == 0]['reading_score'], ax=ax[1], color='blue')
sns.distplot(df[df.gender == 1]['reading_score'], ax=ax[1], color='red')
sns.distplot(df[df.gender == 0]['writing_score'], ax=ax[2], color='blue')
sns.distplot(df[df.gender == 1]['writing_score'], ax=ax[2], color='red')
plt.show()


# # doing test prep dominates significantly in both reading and writing, not so much so in math.

# In[ ]:


fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.distplot(df[df['test_preparation_course'] == 0]['math_score'], ax=ax[0], color='black')
sns.distplot(df[df['test_preparation_course'] == 1]['math_score'], ax=ax[0], color='green')
sns.distplot(df[df['test_preparation_course'] == 0]['reading_score'], ax=ax[1], color='black')
sns.distplot(df[df['test_preparation_course'] == 1]['reading_score'], ax=ax[1], color='green')
sns.distplot(df[df['test_preparation_course'] == 0]['writing_score'], ax=ax[2], color='black')
sns.distplot(df[df['test_preparation_course'] == 1]['writing_score'], ax=ax[2], color='green')
plt.show()


# In[ ]:


fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.violinplot(y="math_score",    x='parental_level_of_education', data = df, palette="muted", ax=ax[0], )
sns.violinplot(y="reading_score", x='parental_level_of_education', data = df, palette="muted", ax=ax[1])
sns.violinplot(y="writing_score", x='parental_level_of_education', data = df, palette="muted", ax=ax[2])
plt.show()


# In[ ]:


fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 5)
sns.violinplot(y="math_score",    x='race/ethnicity', data = df, palette="muted", ax=ax[0])
sns.violinplot(y="reading_score", x='race/ethnicity', data = df, palette="muted", ax=ax[1])
sns.violinplot(y="writing_score", x='race/ethnicity', data = df, palette="muted", ax=ax[2])
plt.show()


# ** feature importance **

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *


# In[ ]:


X = df[['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
y = df['avg_score']


# In[ ]:


regressor = RandomForestRegressor(oob_score=True)
regressor.fit(X,y)
print(X.columns.tolist())
print(regressor.feature_importances_)


# **Assertion #1**
# 
# Test out the assertion that those who get reduced/free lunch are economically disadvantages. Assume that their parents who have more advanced education make more on average, and hence are not eligible for free/reduced lunches.
# 
# **Outcome**
# 
# this assertion does not hold up. If it had been true, college and above educated parents receive free/reduced lunches at the same rate as standard lunches. This is quite counterintutitive, the opposite should have been seen.

# In[ ]:


sns.countplot(x='parental_level_of_education', data=df, hue='lunch')


# In[ ]:


sns.countplot(x='avg_score_decile', data=df, hue='lunch')


# In[ ]:


sns.countplot(x='avg_score_decile', data=df, hue='test_preparation_course')


# In[ ]:


sns.countplot(x='parental_level_of_education', data=df, hue='lunch')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




