#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# * The San Francisco Controller's Office maintains a database of the salary and benefits paid to City employees between fiscal years 2013 - 2017.
# * The dataset hosted by the city of San Francisco. The organization has an open data platform and they update their information according the amount of data that is brought in.
# * This dataset is updated annually. New data is added on a bi-annual basis when available for each fiscal and calendar year.
# * Our target is to predict the salary.
# * The csv file includes 213K observations and 22 features. After cleaning the Nan's and defining the interested population as employees with total annual salary of at least 35,000$, we have almost 150K observations.

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
## print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder, KBinsDiscretizer, MaxAbsScaler, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split as split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from time import time


# ## Load Data

# In[ ]:


## nRowsRead = 1000
df = pd.read_csv('/kaggle/input/sf-employee-compensation/employee-compensation.csv')
## df.dataframename = 'employee-compensation.csv'
## nRow, nCol = df.shape
## print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## Check Missing Value(NaN)

# In[ ]:


print(df.isnull().sum())


# In[ ]:


df[df.Union.isnull() == True].head()


# In[ ]:


df[df.Union.isnull() == True]['Organization Group'].value_counts()


# In[ ]:


df[df.Union.isnull() == True][df['Organization Group'] == 'Communit Health'].Job.value_counts()


# In[ ]:


df[df.Job == 'Technology Expert II'].shape


# ## EDA

# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


salaries_sm = scatter_matrix(df[['Salaries', 'Total Salary', 'Total Compensation']])


# In[ ]:


benefits_sm = scatter_matrix(df[['Retirement', 'Health and Dental', 'Other Benefits', 'Salaries']])


# ## Target

# In[ ]:


ax = sns.kdeplot(df['Salaries'])


# In[ ]:


df['Salaries'].describe()


# In[ ]:


salary = ['Salaries', 'Total Salary', 'Total Compensation']
for col in salary:
    ax_salary = sns.kdeplot(df[col])
    ax_salary


# In[ ]:


benefits = ['Retirement', 'Health and Dental', 'Other Benefits', 'Total Benefits']
for col in benefits:
    ax_benefits = sns.kdeplot(df[col])
    ax_benefits


# ## Sample the Data

# In[ ]:


## Remove salaries lower than 35,000
df[df['Salaries']<35000].count()


# In[ ]:


df1 = df[df['Salaries'] > 35000]


# In[ ]:


## How many organizations we are losing by reducing the data to salaries <35,000
org_x = df[df.Salaries<35000]['Organization Group'].value_counts()
org_y = df['Organization Group'].value_counts()
org_z = pd.concat([org_x, (org_x/org_y)], axis=1, join='inner', sort=False)
org_z.columns = ['Organization Count', 'Organization %']
org_z


# In[ ]:


org_ax = org_z['Organization Count'].plot('bar')
for p in org_ax.patches:
    org_ax.annotate(int(p.get_height()), (p.get_x(), p.get_height()*1.01))


# In[ ]:


## How many departments we are losing by reducing the data to salaries <35,000
dep_x = df[df.Salaries<35000].Department.value_counts()
dep_y = df.Department.value_counts()
dep_z = pd.concat([org_x, (org_x/org_y)], axis=1, join='inner', sort=False)
dep_z.columns = ['Department Count', 'Department %']
dep_z


# In[ ]:


## How many jobs we are losing by reducing the data to salaries <35,000
job_x = df[df.Salaries<35000].Job.value_counts()
job_y = df.Job.value_counts()
job_z = pd.concat([org_x, (org_x/org_y)], axis=1, join='inner', sort=False)
job_z.columns = ['Job Count', 'Job %']
job_z


# ### Distribution of salaries by organizations

# In[ ]:


plt.figure(figsize=(15,8))
for col in ist(df1['Organization Group'].unique()):
    ax=sns.kdeplot(df1['Organization Group' = col], label = col)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




