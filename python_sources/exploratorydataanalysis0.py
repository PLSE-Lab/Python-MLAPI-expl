#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')


# In[ ]:


df.head()


# # Descriptive Statistics

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


sns.heatmap(df.isnull(), cbar=False)
plt.show()


# No missing value in data

# In[ ]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# Not surprisingly, scores of all the subject are highly correlated with each other. 

# calculating average score

# In[ ]:


df['avg score']=(df['math score']+df['reading score']+df['writing score'])/3


# In[ ]:


df.head()


# In[ ]:


scores=['math score','reading score','writing score','avg score']


# In[ ]:


print('uniqe values:''\n''\n') 
print('test prep course:' )
print(df['test preparation course'].value_counts())
print('\n')
print('lunch:')
print(df['lunch'].value_counts())
print('\n')
print('parent education:')
print(df['parental level of education'].value_counts())


# # Checking skewness of data
# 

# In[ ]:


plt.rcParams['figure.figsize'] = (18, 6)
plt.subplot(1, 3, 1)
sns.distplot(df['math score'])

plt.subplot(1, 3, 2)
sns.distplot(df['reading score'])

plt.subplot(1, 3, 3)
sns.distplot(df['writing score'])

plt.suptitle('Checking for Skewness', fontsize = 18)
plt.show()


# * No skewness in data

# # EDA

# In[ ]:


sns.countplot('gender',data=df)
plt.show()


# Here we have more number of females than males.

# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot('parental level of education',data=df)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby('parental level of education')[scores].mean().plot.bar(ax=ax)
plt.ylabel('scores')
plt.show()


# 1. Despite of lower number of parents who have master's degree, their child outperfromed in each and every test in comparison to others.
# 
# 2. Children's whose parent have only done high school, performed lower as compared to others.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby('gender')[scores].mean().plot.bar(ax=ax)
plt.ylabel('score')
plt.show()


# From the above plot we see that performance of male in math exam is better than that of female in math exam.
# Except in maths female students out-performed male students in other tests. 

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby('race/ethnicity')[scores].mean().plot.bar(ax=ax)
plt.ylabel('scores')
plt.show()


# * Students from group E have done better in tests when compared to other groups.
# * Surprisingly Group E have performed better in Maths than other tests.*
# * Group A performed worst in comparison to others Groups.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby('test preparation course')[scores].mean().plot.bar(ax=ax)
plt.ylabel('scores')
plt.show()


# Students who have completed the Test Preparation Course have performed better in each test when compared to one who haven't done any course. 

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
df.groupby('lunch')[scores].mean().plot.bar(ax=ax)
plt.ylabel('scores')
plt.show()


# Students with standard lunch performed better than the ones with free or reduced lunch

# # Summary 

# **Following important Insights are drawn from the given data:-**
# * Children who's parents have Master's degree performed well in each test. While children who's parents have performed worst when compared to others.
# * Children from Group E performed best and Group A performed worst.
# * Female students performed better in writing and reading test, but in maths exam male students performed better.
# 
