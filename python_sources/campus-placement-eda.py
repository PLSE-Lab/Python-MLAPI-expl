#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# * sl_no - Serial Number
# * gender - Gender- Male='M',Female='F'
# * ssc_p - Secondary Education percentage- 10th Grade
# * ssc_b - Board of Education- Central/ Others
# * hsc_p - Higher Secondary Education percentage- 12th Grade
# * hsc_b - Board of Education- Central/ Others
# * hsc_s - Specialization in Higher Secondary Education
# * degree_p - Degree Percentage
# * degree_t - Under Graduation(Degree type)- Field of degree education
# * workex - Work Experience
# * etest_p - Employability test percentage ( conducted by college)
# * specialisation - Post Graduation(MBA)- Specialization
# * mba_p - MBA percentage
# * status - Status of placement- Placed/Not placed
# * salary - Salary offered by corporate to candidates

# In[ ]:


df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# #### Null values are due to the fact that students who are not placed have an input of NaN in the salary column.

# In[ ]:


df.info()


# ## Visualising categorical variables

# In[ ]:


## object type features are categorical variables here.
col_list = ['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s',
       'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p',
       'status', 'salary']

cat = []
for col in col_list:
    if df[col].dtypes == 'object':
        cat.append(col)
        
cat


# In[ ]:


## Visualisation using countplot
for col in cat:
    print(df[col].value_counts())
    sns.countplot(df[col])
    plt.show()


# ## Comparing gender, work experience and specialisation with status.

# #### 1. Gender
# 100 out of 139 males were placed and 48 out of 76 females were placed.

# In[ ]:


male_pct = round((100/139) * 100, 2)
female_pct = round((48/76) * 100, 2)
print(f"Male placement percentage: {male_pct}%")
print(f"Female placement percentage: {female_pct}%")


# #### 2. Work experience
# 64 out of 74 who had workex were placed and 84 out of 141 who didn't have workex were placed.

# In[ ]:


y = round((64/74) * 100, 2)
n = round((84/141) * 100, 2)
print(f"Percentage of students with workex who got placed: {y}%")
print(f"Percentage of students without workex who got placed: {n}%")


# #### 3. Specialisation
# 53 out of 95 in Mkt&HR got placed and 95 out of 120 in Mkt&Fin got placed

# In[ ]:


hr = round((53/95) * 100, 2)
fin = round((95/120) * 100, 2)
print(f"Percentage of students in Mkt&HR who got placed: {hr}%")
print(f"Percentage of students in Mkt&Fin who got placed: {fin}%")


# ## Finding correlations between ssc_p, hsc_p, degree_p and mba_p

# In[ ]:


pct = df[['ssc_p', 'hsc_p', 'degree_p', 'mba_p']]
sns.heatmap(pct.corr(), annot=True)


# Here degree_p has highest correlation with respect to mba_p followed by hsc_p.

# ## Visualising mba percentage vs. employability test percentage with respect to their status

# In[ ]:


sns.scatterplot(x='mba_p', y='etest_p', data=df, hue='status')


# ### The interesting thing here is that even after getting higher percentage in MBA and the employability test, students were not placed.

# In[ ]:




