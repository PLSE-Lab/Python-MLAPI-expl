#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')


# In[ ]:


names = ['gender', 'race_or_ethnicity', 'parent_education', 'lunch',
       'test_prep_course', 'math_score', 'reading_score',
       'writing_score']

df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv", names=names, header=0)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 3, 1)
sns.distplot(df.math_score,ax=ax, color = 'r')
plt.title('Math Score', fontsize=12)
ax = fig.add_subplot(1, 3, 2)
sns.distplot(df.reading_score,ax=ax, color = 'r')
plt.title('Reading Score', fontsize=12)
ax = fig.add_subplot(1, 3, 3)
sns.distplot(df.writing_score,ax=ax, color = 'r')
plt.title('Writing Score', fontsize=12)
plt.show()


# In[ ]:


sns.pairplot(df)


# There is a linear correlation between scores in different subjects.

# In[ ]:


df.describe()


# In[ ]:


df.columns


# ### EDA
# 
# ### Gender

# In[ ]:


sns.countplot('gender', data = df)


# In[ ]:


df.groupby('gender').mean()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 3, 1)
sns.barplot(x = 'gender', y = 'math_score', data = df)
ax.set_title('Math Score', fontsize = 12)
ax = fig.add_subplot(1, 3, 2)
sns.barplot(x = 'gender', y = 'reading_score', data = df)
ax.set_title('Reading Score', fontsize = 12)
ax = fig.add_subplot(1, 3, 3)
sns.barplot(x = 'gender', y = 'writing_score', data = df)
ax.set_title('Writing Score', fontsize = 12)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Gender vs Score', fontsize = 12)
plt.show()


# Male students appear to be good at maths. Female students are good at reading and writing.
# 
# ### Race / Ethnicity

# In[ ]:


sns.countplot('race_or_ethnicity', data = df)
plt.title('Number of students from each Race/Ethnicity', fontsize=12)
plt.show()


# In[ ]:


df.groupby('race_or_ethnicity').mean()


# Students from 'Group E' are good in all three areas.
# 
# ### Parent Education

# In[ ]:


plt.figure(figsize=(11,4))
sns.countplot('parent_education', data = df)
plt.title('Parents education', fontsize=12)
plt.show()


# In[ ]:


df.groupby('parent_education', sort = True).mean()


# In[ ]:


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(2, 2, 1)
sns.barplot(y = 'parent_education', x = 'math_score', data = df)
ax.set_title('Math Score', fontsize = 12)
ax = fig.add_subplot(2, 2, 2)
sns.barplot(y = 'parent_education', x = 'reading_score', data = df)
ax.set_title('Reading Score', fontsize = 12)
ax = fig.add_subplot(2, 2, 3)
sns.barplot(y = 'parent_education', x = 'writing_score', data = df)
ax.set_title('Writing Score', fontsize = 12)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Parent Education course vs Score', fontsize = 12)
plt.show()


# Students, whose parents are degree holders, seems to perform better than students with parents who studied only school or college.
# 
# ### Lunch

# In[ ]:


plt.figure(figsize=(5,4))
sns.countplot('lunch', data = df)
plt.title('Type of Lunch', fontsize=12)
plt.show()


# In[ ]:


df.groupby('lunch').mean()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 3, 1)
sns.barplot(x = 'lunch', y = 'math_score', data = df)
ax.set_title('Math Score', fontsize = 12)
ax = fig.add_subplot(1, 3, 2)
sns.barplot(x = 'lunch', y = 'reading_score', data = df)
ax.set_title('Reading Score', fontsize = 12)
ax = fig.add_subplot(1, 3, 3)
sns.barplot(x = 'lunch', y = 'writing_score', data = df)
ax.set_title('Writing Score', fontsize = 12)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Test preparation course vs Score', fontsize = 12)
plt.show()


# Students who eat standard lunch have higher averages in all subjects compared to students getting free/reduced lunch.
# 
# ### Test Preparation course

# In[ ]:


plt.figure(figsize=(5,4))
sns.countplot('test_prep_course', data = df)
plt.title('Number of students who finished Test preparation course', fontsize=12)
plt.show()


# Many students did not take the preparation course.

# In[ ]:


df.groupby('test_prep_course').mean()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 3, 1)
sns.barplot(x = 'test_prep_course', y = 'math_score', data = df)
ax.set_title('Math Score', fontsize = 12)
ax = fig.add_subplot(1, 3, 2)
sns.barplot(x = 'test_prep_course', y = 'reading_score', data = df)
ax.set_title('Reading Score', fontsize = 12)
ax = fig.add_subplot(1, 3, 3)
sns.barplot(x = 'test_prep_course', y = 'writing_score', data = df)
ax.set_title('Writing Score', fontsize = 12)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Test preparation course vs Score', fontsize = 12)
plt.show()


# Performance of the students who to a test preparation course seem to perform well.
# 

# In[ ]:




