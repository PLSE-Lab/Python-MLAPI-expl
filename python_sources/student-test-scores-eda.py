#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_full = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
train_full = train_full.replace('free/reduced', 'free')
train_full.head()


# In[ ]:


row, col = train_full.shape
print('Rows: %d, Columns: %d' %(row, col))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# Helper function to make a table of averages:

# In[ ]:


import collections

# sorted_dict = collections.OrderedDict(sorted_x)

def get_average_math(df, feature):
    unique_vals = pd.Series(df[feature].unique())
    scores = {}
    for e in unique_vals:
        rows = df[df[feature]==e]
        scores[e] = rows['math score'].mean()
    
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    scores = collections.OrderedDict(sorted_scores)
    table = pd.DataFrame(scores.items(), columns=[feature, 'Average math score'])
    return table

def get_average_reading(df, feature):
    unique_vals = pd.Series(df[feature].unique())
    scores = {}
    for e in unique_vals:
        rows = df[df[feature]==e]
        scores[e] = rows['reading score'].mean()
    
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    scores = collections.OrderedDict(sorted_scores)
    table = pd.DataFrame(scores.items(), columns=[feature, 'Average reading score'])
    return table


# # Math scores
# 
# This section will only look at math scores for every combination of the features.

# ## Race/Ethnicity Analysis
# 
# Looking at the population of high school students by race/ethnicity and their average math scores.

# In[ ]:


sns.catplot(data=train_full, x='race/ethnicity', kind='count')
plt.title('Population by race/ethnicity')


# In[ ]:


sns.catplot(data=train_full, y='race/ethnicity', x='math score', kind='box')
plt.ylabel('Race/Ethnicity')
plt.xlabel('Math score')
plt.title('Math scores by race/ethnicity')


# ### Table of averages by race

# In[ ]:


get_average_math(train_full, 'race/ethnicity')


# ## Gender analysis
# 
# Looking at the population of high school students by gender and their average math scores.

# In[ ]:


sns.catplot(data=train_full, x='gender', kind='count')
plt.title('Population by gender')


# In[ ]:


sns.catplot(data=train_full, y='gender', x='math score', kind='box')
plt.ylabel('parental level of education')
plt.xlabel('Math score')
plt.title('Math scores by gender')


# ### Table of averages by gender

# In[ ]:


get_average_math(train_full, 'gender')


# ## Level of education of analysis

# In[ ]:


sns.catplot(data=train_full, x='parental level of education', kind='count', aspect=3)
plt.title('Population by parents\' level of education')


# In[ ]:


sns.catplot(data=train_full, y='parental level of education', x='math score', kind='box', height=5, aspect=2)
plt.ylabel('parental level of education')
plt.xlabel('Math score')
plt.title('Math scores by parental level of education')


# ### Table of averages by parental level of education

# In[ ]:


get_average_math(train_full, 'parental level of education')


# ## Lunch analysis

# In[ ]:


sns.catplot(data=train_full, x='lunch', kind='count')
plt.title('Population by lunch')


# In[ ]:


sns.catplot(data=train_full, y='lunch', x='math score', kind='box')
plt.ylabel('Lunch')
plt.xlabel('Math score')
plt.title('Math scores by lunch')


# ### Table for average math scores based on lunch type

# In[ ]:


get_average_math(train_full, 'lunch')


# ## Preparation analysis

# In[ ]:


sns.catplot(data=train_full, x='test preparation course', kind='count')
plt.title('Population by preparation')


# In[ ]:


sns.catplot(data=train_full, y='test preparation course', x='math score', kind='box')
plt.ylabel('Test Preparation')
plt.xlabel('Math score')
plt.title('Math scores by preparation')


# ### Table for average math scores based on preparation

# In[ ]:


get_average_math(train_full, 'test preparation course')


# ## Race and gender analysis

# In[ ]:


gr_data = train_full.copy()
gr_data['gender_race'] = gr_data['gender'] + '_' + gr_data['race/ethnicity']
sns.catplot(data=gr_data.sort_values('gender_race'), y='gender_race', x='math score', kind='box', height=8)
plt.ylabel('Race and gender')
plt.xlabel('Math score')
plt.title('Math scores by gender and race/ethnicity')


# In[ ]:


get_average_math(gr_data, 'gender_race')


# ## Race and parents' level of education

# In[ ]:


er_data = train_full.copy()
er_data['education_race'] = gr_data['parental level of education'] + '_' + gr_data['race/ethnicity']
sns.catplot(data=er_data.sort_values('education_race'), y='education_race', x='math score', kind='box', height=8)
plt.ylabel('Race and parental level of education')
plt.xlabel('Math score')
plt.title('Math scores by parental level of education and race/ethnicity')


# In[ ]:


get_average_math(er_data, 'education_race')


# ## Gender and level of education
# 
# Above, we see that students whose parents graduated with some high school performed better on the math test than students whose parents graudated high school.

# In[ ]:


ge_data = train_full.copy()
ge_data['gender_education'] = ge_data['gender'] + '_' + ge_data['parental level of education']
sns.catplot(data=ge_data.sort_values('gender_education'), y='gender_education', x='math score', kind='box', height=8)
plt.ylabel('Race and parental level of education')
plt.xlabel('Math score')
plt.title('Math scores by gender and parent level of education')


# ## Lunch section

# In[ ]:


lr_data = train_full.copy()
lr_data['lunch_race'] = lr_data['lunch'] + '_' + lr_data['race/ethnicity']
sns.catplot(data=lr_data.sort_values('lunch_race'), x='lunch_race', kind='count', aspect=6)
plt.title('Population by lunch/race')


# There doesn't seem to be a difference in proportion of the different ethnic groups receiving more or less standard lunches.

# In[ ]:


sns.catplot(data=lr_data.sort_values('lunch_race'), y='lunch_race', x='math score', kind='box', height=6)
plt.ylabel('Race and lunch')
plt.xlabel('Math score')
plt.title('Math scores by race and lunch')


# In[ ]:


sns.catplot(y="lunch_race", x="math score", hue="gender", kind="box", data=lr_data.sort_values('lunch_race'), height=8)
plt.title('Math scores based on lunch/race and gender')


# In[ ]:


sns.catplot(y="lunch_race", x="reading score", hue="gender", kind="box", data=lr_data.sort_values('lunch_race'), height=8)
plt.title('Reading scores based on lunch/race and gender')


# In[ ]:


get_average_math(lr_data, 'lunch_race')


# In[ ]:


lg_data = train_full.copy()
lg_data['lunch_gender'] = lg_data['lunch'] + '_' + lr_data['gender']
sns.catplot(data=lg_data.sort_values('lunch_gender'), x='lunch_gender', kind='count', aspect=2)
plt.title('Population by lunch/gender')


# In[ ]:


sns.catplot(data=lg_data.sort_values('lunch_gender'), y='lunch_gender', x='math score', kind='box')
plt.ylabel('Gender and lunch')
plt.xlabel('Math score')
plt.title('Math scores by gender and lunch')


# In[ ]:


get_average_math(lg_data, 'lunch_gender')


# In[ ]:


le_data = train_full.copy()
le_data['lunch_education'] = le_data['lunch'] + '_' + le_data['parental level of education']
sns.catplot(data=le_data.sort_values('lunch_education'), x='lunch_education', kind='count', aspect=6)
plt.title('Population by lunch/education')


# In[ ]:


sns.catplot(data=le_data.sort_values('lunch_education'), y='lunch_education', x='math score', kind='box', height=8)
plt.ylabel('Education and lunch')
plt.xlabel('Math score')
plt.title('Math scores by education and lunch')


# In[ ]:


get_average_math(le_data, 'lunch_education')


# ## Preparation section

# In[ ]:


pr_data = train_full.copy()
pr_data['prep_race'] = pr_data['test preparation course'] + '_' + pr_data['race/ethnicity']
sns.catplot(data=pr_data.sort_values('prep_race'), x='prep_race', kind='count', aspect=6)
plt.title('Population by preparation and race')


# In[ ]:


sns.catplot(data=pr_data.sort_values('prep_race'), y='prep_race', x='math score', kind='box', height=8)
plt.ylabel('Race and preparation')
plt.xlabel('Math score')
plt.title('Math scores by race and preparation')


# In[ ]:


get_average_math(pr_data, 'prep_race')


# In[ ]:


pg_data = train_full.copy()
pg_data['prep_gender'] = pg_data['test preparation course'] + '_' + pg_data['gender']
sns.catplot(data=pg_data.sort_values('prep_gender'), x='prep_gender', kind='count', aspect=2)
plt.title('Population by preparation and gender')


# In[ ]:


sns.catplot(data=pg_data.sort_values('prep_gender'), y='prep_gender', x='math score', kind='box')
plt.ylabel('Gender and preparation')
plt.xlabel('Math score')
plt.title('Math scores by gender and preparation')


# In[ ]:


get_average_math(pg_data, 'prep_gender')


# In[ ]:


pe_data = train_full.copy()
pe_data['prep_education'] = pe_data['parental level of education'] + '_' + pe_data['test preparation course']
sns.catplot(data=pe_data.sort_values('prep_education'), x='prep_education', kind='count', aspect=6)
plt.title('Population by preparation and education')


# In[ ]:


sns.catplot(data=pe_data.sort_values('prep_education'), y='prep_education', x='math score', kind='box', height=8)
plt.ylabel('Education and preparation')
plt.xlabel('Math score')
plt.title('Math scores by preparation and education')


# In[ ]:


get_average_math(pe_data, 'prep_education')


# # Reading score (comparison)
# 
# Females perform better overall on test scores, and better on both reading and writing scores. The analysis done above for Math scores can also be done for reading and writing test scores.

# In[ ]:


gr_data = train_full.copy()
gr_data['gender_education'] = gr_data['gender'] + '_' + gr_data['parental level of education']
sns.catplot(data=gr_data, y='gender_education', x='reading score', kind='box', height=8)
plt.ylabel('Race and parental level of education')
plt.xlabel('Reading score')
plt.title('Reading scores by gender and parent level of education')


# In[ ]:


get_average_reading(gr_data, 'gender_education')

