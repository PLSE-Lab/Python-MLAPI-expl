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


# **Lets do exploratory data analysis for this dataset**

# 1. Check few records from both dataset 
# 1. Check how many categorical variables are present in dataset
# 1. Handle Missing values in dataset
# 1. Visualize distribution of data along with each group 
# 

# In[ ]:


#importing libraries which we required further
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#lets read the salary data and store it in dataframe using pandas
data = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv')
data.head()


# In[ ]:


#lets see overview of dataset using describe()
data.describe()


# In[ ]:


data['Benefits'].value_counts()


# In[ ]:


data['BasePay'].value_counts()


# In[ ]:


data['Status'].value_counts()


# In[ ]:


#since there are lot of missing values in status and notes variable in datsaet so lets drop it 
data.drop(columns=["Status","Notes"],inplace=True,axis=1)


# In[ ]:


data.head()


# In[ ]:


#lets identify categorical columns in dataset
# Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


for col in ['BasePay','OvertimePay','OtherPay','Benefits']:
    data[col]=pd.to_numeric(data[col],errors='coerce')


# Now we have to deal with missing values in column BasePay and Benifits
# Lets fill them by there mean value for simplicity further we will modify for model improvement.
# 

# In[ ]:



data['BasePay'].fillna(value=data['BasePay'].mean(),inplace=True,axis=0)
data['Benefits'].fillna(value=data['Benefits'].mean(),inplace=True,axis=0)
data['OvertimePay'].fillna(value=data['OvertimePay'].mean(),inplace=True,axis=0)
data['OtherPay'].fillna(value=data['OtherPay'].mean(),inplace=True,axis=0)


# Now final check for null or missing values 

# In[ ]:


data.isnull().sum()


# In[ ]:


data['JobTitle'].value_counts()


# In[ ]:


print(data.JobTitle.unique())


# In[ ]:


data['EmployeeName'] = data['EmployeeName'].apply(str.upper)
data.head()


# In[ ]:


data['JobTitle'] = data['JobTitle'].apply(str.upper)
data['JobTitle'].value_counts()


# In[ ]:


d_hsp={"1":"I","2":"II","3":"III","4":"IV","5":"V","6":"VI","7":"VII","8":"VIII",
       "9":"IX","10":"X","11":"XI","12":"XII","13":"XIII","14":"XIV","15":"XV",
       "16":"XVI","17":"XVII","18":"XVIII","19":"XIX","20":"XX","21":"XXI",
       "22":"XXII","23":"XXIII","24":"XXIV","25":"XXV"}
data['JobTitle'] = data['JobTitle'].replace(d_hsp, regex=True)


# In[ ]:


data['JobTitle'].value_counts()


# 

# In[ ]:


data.head()


# Its time to apply encoding for categorical variable except "Employee name " because it is unique and we already have id column for the same.
# So we encode Job title and Agency column .

# In[ ]:


data.drop(columns=["EmployeeName"],inplace=True,axis=1)


# In[ ]:


from sklearn import preprocessing
data.apply(preprocessing.LabelEncoder().fit_transform(data['JobTitle']))


# In[ ]:


data.to_csv("input.csv")

