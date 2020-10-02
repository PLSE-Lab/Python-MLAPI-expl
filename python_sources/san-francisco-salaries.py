#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data=pd.read_csv('../input/Salaries.csv')


# In[ ]:


data.info()


# In[ ]:


data.drop(['Notes'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.set_index('Id')


# In[ ]:


data['JobTitle'].nunique()


# In[ ]:


data['JobTitle'].value_counts()


# In[ ]:


data.info()


# In[ ]:


for col in ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']:
    data[col] = pd.to_numeric(data[col], errors='coerce')


# In[ ]:


data.info()


# What is the average BasePay ? 

# In[ ]:


data['BasePay'].mean()


# What is the highest amount of OvertimePay in the dataset ?

# In[ ]:


data['OvertimePay'].max()


# What is the job title and totalpaybenefits of JOSEPH DRISCOLL ?

# In[ ]:


data[data['EmployeeName'] == 'JOSEPH DRISCOLL'][['JobTitle','TotalPayBenefits']]


# What is the name of highest paid person (including benefits)?

# In[ ]:


data.iloc[data['TotalPayBenefits'].idxmax()]


# What was the average (mean) BasePay of all employees per year? (2011-2014) ?

# In[ ]:


data.groupby('Year')['BasePay'].mean()


# How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)

# In[ ]:


sum(data[data['Year'] == 2013]['JobTitle'].value_counts() == 1)


# How many people have the word Chief in their job title?

# In[ ]:


sum(data['JobTitle'].str.lower().str.contains('chief', na=False))


# In[ ]:




