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


# # 1-Import the data

# In[ ]:


df=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.describe()
df_new = df.rename(columns={'race/ethnicity': 'race_ethnicity','parental level of education': 'parental','test preparation course': 'test','math score': 'math','reading score': 'reading','writing score': 'writing'})
df_new


# # 2- Correlation between gender and reading score
# 
# As we can see in the box plot, their seems to be a higher number of females with a higher reading score, which we will confirm with a statistical test.
# 
# Since the gender variable has 2 categories, I decided to perform a t-test to know if there is a significance difference in the reading score of males and females. Since the p-value is very small (under alpha=0.05), then we can say that there is a significant difference in the reading scores of males and females. 

# In[ ]:


import matplotlib.pyplot as plt 
import seaborn as sns

from scipy import stats

pd.value_counts(df_new.gender).to_frame().reset_index()

boxprops = dict(linestyle='-', linewidth=3, color='v')
medianprops = dict(linestyle='-', linewidth=3, color='v')
df_new.boxplot( by='gender',column=['reading'],grid= False, showfliers=False, showmeans=True, boxprops=boxprops,medianprops=medianprops )

sns.boxplot(x='diagnosis', y='area_mean', data=df[["gender","reading"]])

df_newf=df_new[df_new.gender != 'male']
df_newm=df_new[df_new.gender != 'female']

stats.ttest_ind(df_newf.reading, df_newm.reading,equal_var=True)


# # 3- Correlation between race and math score
# 
# As we can see in the box plot, the math score between each ethnicity does not seem equal. Group E seems to have the higher average, while group seems to have the lowest one.
# 
# Since the race variable has 5 categories, I decided to perform anova test to know if there is a significance difference in the reading score of each group. Since the p-value is very small (under alpha=0.05), then we can say that there is a significant difference in the math scores of each ethnicity.

# In[ ]:


import scipy.stats as stats

pd.value_counts(df_new.race_ethnicity).to_frame().reset_index()

df_new.boxplot( by='race_ethnicity',column=['math'],grid= False )

stats.f_oneway(df_new['math'][df_new['race_ethnicity'] == 'group A'],
               df_new['math'][df_new['race_ethnicity'] == 'group B'],
               df_new['math'][df_new['race_ethnicity'] == 'group C'],
               df_new['math'][df_new['race_ethnicity'] == 'group D'],
               df_new['math'][df_new['race_ethnicity'] == 'group E'],
               )


# # 4-Correlation between lunch and writing score
# 
# As we can see in the box plot, the average writing score in the standard group seems higher, which we will confirm with a statistical test.
# 
# Since the lunch variable has 2 categories, I decided to perform a t-test to know if there is a significance difference in the writing score of males and females. Since the p-value is very small (under alpha=0.05), then we can say that there is a significant difference in the writing scores between the free/reduced and standard group.

# In[ ]:


df_new.boxplot( by='lunch',column=['writing'],grid= False )

pd.value_counts(df_new.lunch).to_frame().reset_index()

df_newf=df_new[df_new.lunch != 'standard']
df_news=df_new[df_new.lunch != 'free/reduced']

stats.ttest_ind(df_newf.reading, df_news.reading,equal_var=False)


# **DO NOT FORGET TO UPVOTE MY NOTEBOOK IF YOU LIKE IT :)**
