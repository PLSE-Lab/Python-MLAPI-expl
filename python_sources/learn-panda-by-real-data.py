#!/usr/bin/env python
# coding: utf-8

# # This is a short introduction to pandas, geared mainly for new users.
# **You can Learn how to read Data and elemantry exploratry Data Analysis by this kerner.
# I try to anayze a real bigdata step by stem in this kerner.**

# # Step1: Import all necessary Libraries as follow
# **Customarily, we import**

# In[ ]:


import numpy as np
import pandas as pd


# # Step2: Import the Data and ...

# 1- Read the Data and make a DataFrame

# In[ ]:


df=pd.read_csv('../input/20112018-salaries-for-san-francisco/Total.csv')


# 2-Check Head of the DataFrame (here: df)

# In[ ]:


df.head(10)


# 3- Check the Shape of the DataFrame (here: df)

# In[ ]:


df.shape


# Shows that, in this DataFrame, there are 312882 rows in 9 columns

# 4- Check how many records are in each columns and type of each columns

# In[ ]:


df.info()


# the above window shows that:
# 1- All columns have 312882 records
# 2- Columns (BasPay, OvertimePay, OtherPay and Benefits) are in object format. Therefore for analyzing these columns, we must change their type tp float64 at first. So,..

# In[ ]:


series_list = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']
for series in series_list:
    df[series] = pd.to_numeric(df[series], errors='coerce')


# In[ ]:


df.info()


# # Step3: EDA

# What is the average of BasePay

# In[ ]:


df['BasePay'].mean()


# what is the max of BasePay

# In[ ]:


df['BasePay'].max()


# Descrption parameter for BasePay:

# In[ ]:


df['BasePay'].describe()


# In[ ]:


df.describe()


# Exercise: Apply this code for other Columns

# What is the job title of GARY JIMENEZ

# In[ ]:


df[df['EmployeeName']=='Gary JIMENEZ']['JobTitle']


# In[ ]:


df[df['EmployeeName']=='Gary JIMENEZ']['BasePay']


# In[ ]:


df[df['TotalPayBenefits']==df['TotalPayBenefits'].max()]


# In[ ]:


df.groupby('Year').mean()['BasePay']


# In[ ]:


df['EmployeeName'].nunique()


# In[ ]:


df[['EmployeeName','Year']].groupby('Year').nunique()


# In[ ]:


df['JobTitle'].value_counts().head(5)


# In[ ]:




