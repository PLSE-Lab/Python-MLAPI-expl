#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


DATA = pd.read_csv('../input/HR_comma_sep.csv')
DATA.head()


# In[ ]:


#Info on column data typs
DATA.info()


# In[ ]:


#Summary statistics for the numerical fields (8/10)
DATA.describe()


# In[ ]:


#Correlation among these 8 of 10 variables
cor = DATA.corr()
sns.heatmap(cor)


# In[ ]:


#Check the effect of the categorical fields on people who have left
#Build crosstabs
pd.crosstab(DATA.sales,DATA.left).apply(lambda r: r/r.sum(), axis = 0)


# Highest number of people left from sales followed by technical and support staff.

# In[ ]:


#Similar analysis for salary and people leaving
pd.crosstab(DATA.left, DATA.salary)


# In[ ]:


#Check department wise salary break-up only for people who left
DATA1 = DATA[DATA.left == 1]
pd.crosstab(DATA1.sales, DATA1.salary)


# In[ ]:


DATA2 = pd.get_dummies(DATA)
cor2 = DATA2.corr()
sns.heatmap(cor2)


# In[ ]:


DATA2['satisfaction_level_cube'] = DATA2['satisfaction_level']**3
DATA2['satisfaction_level_sqrt'] = np.sqrt(DATA2['satisfaction_level'])
DATA2['satisfaction_level_log'] = np.log(DATA2['satisfaction_level'])
DATA2.hist('number_project')
#cor3 = DATA2.corr()
#sns.heatmap(cor3)


# In[ ]:


DATA1.boxplot('satisfaction_level' , by = 'salary')


# In[ ]:


#Finding the ratio of classes
DATA2['left'][DATA2.left == 1].count()/DATA2['left'].count()


# Only 23% of data falls into class 1. Rest 77% are the people who have not left. Example of a class imbalance problem.

# In[ ]:


#Baseline model

