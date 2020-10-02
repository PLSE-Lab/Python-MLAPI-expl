#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df= pd.read_csv("../input/HR-Employee-Attrition.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# *No null values found in data sets*

# In[ ]:


print(df.shape)


# In[ ]:


print(list(df.columns))


# In[ ]:


#Attrition is our dependent column.


# In[ ]:


df.drop(columns='Attrition').dtypes


# In[ ]:


print(df['Attrition'].dtype)


# With full of Yes and No

# 

# In[ ]:


df.duplicated().sum()


# No duplicated data. So no need to worry about duplications.
# 

# In[ ]:


###Data analysis


# **Data Analysis**
df.replace?
# In[ ]:


get_ipython().run_line_magic('pinfo', 'df.replace')


# In[ ]:


df.replace(to_replace={'No': '0', 'Yes': '1'},inplace=True)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'select_dtypes')


# In[ ]:


num_cols = df.select_dtypes(include = np.number)


# In[ ]:


a = num_cols[num_cols.columns].hist(bins=15, figsize=(15,35), layout=(9,3),color = 'blue',alpha=0.7)


# In[ ]:


cat_col = df.select_dtypes(exclude = np.number)


# In[ ]:


cat_col


# In[ ]:


cat_col.columns


# In[ ]:


cat_col.columns.shape


# In[ ]:


fig, ax = plt.subplots(4, 2, figsize=(15, 15))
for variable, subplot in zip(cat_col, ax.flatten()):
    sns.countplot(df[variable], ax=subplot,palette = 'plasma')
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.tight_layout()


# In[ ]:


df[['StandardHours','EmployeeCount']].describe()


# In[ ]:


df[['StandardHours','EmployeeCount']].corr()


# In[ ]:


corr = df.drop(columns=['StandardHours','EmployeeCount']).corr()
corr.style.background_gradient(cmap='YlGnBu')


# In[ ]:


cols = ['Age', 'BusinessTravel', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
        'EnvironmentSatisfaction', 'Gender', 
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']
for col in cols:
    pd.crosstab(df[col],df.Attrition).plot(kind='bar',color = ('blue','red'),figsize=(10,5))


# In[ ]:





# # Age Vs Attrition - From data, it appears that attrition is more at age group 18-23
# # % of attrition is more among people who travel frequently
# # % of attrition is more in sales department
# # %of attrition is more during 0-1 years of working in company
# # People in job role of Sales Representative tend to have more attrition %
# # From given data, overtime population has more attrition

# In[ ]:


### One hot encoding


# In[ ]:


df.columns.shape


# In[ ]:


cat_cols.columns.shape


# In[ ]:


num_cols.columns.shape


# In[ ]:


cat_col_encoded = pd.get_dummies(cat_col)


# In[ ]:


cat_col_encoded.head()


# In[ ]:


dx = pd.concat([num_cols,cat_col_encoded],sort=False,axis=1)


# In[ ]:


dx.head()


# In[ ]:


cat_cols.columns


# In[ ]:


num_cols.columns


# In[ ]:


print("Attrition")


# In[ ]:


X = df.drop(columns='Attrition')


# In[ ]:


y = df['Attrition']


# Couldnot make than this

# In[ ]:




