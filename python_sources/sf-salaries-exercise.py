#!/usr/bin/env python
# coding: utf-8

# # SF Salaries Exercise 
# **Note: These exercises are from the Udemy course, [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/).**
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

# ** Import pandas as pd.**

# In[13]:


import numpy as np
import pandas as pd


# ** Read Salaries.csv as a dataframe called sal.**

# In[18]:


sal = pd.read_csv('../input/Salaries.csv', na_values='Not Provided')


# ** Check the head of the DataFrame. **

# In[19]:


sal.head()


# ** Use the .info() method to find out how many entries there are.**

# In[20]:


sal.info()


# **What is the average BasePay ?**

# In[21]:


sal['BasePay'].mean()


# ** What is the highest amount of OvertimePay in the dataset ? **

# In[22]:


sal['OvertimePay'].max()


# ** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **

# In[23]:


sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']


# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[24]:


sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']


# ** What is the name of highest paid person (including benefits)?**

# In[25]:


ind = sal['TotalPayBenefits'].idxmax()
sal.loc[ind]['EmployeeName']


# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[27]:


ind = sal['TotalPayBenefits'].idxmin()
sal.iloc[ind]


# ** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **

# In[28]:


sal.groupby('Year').mean()['BasePay']


# ** How many unique job titles are there? **

# In[29]:


sal['JobTitle'].nunique()


# ** What are the top 5 most common jobs? **

# In[31]:


sal['JobTitle'].value_counts().head()


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[32]:


(sal[sal['Year']==2013]['JobTitle'].value_counts()==1).sum()


# ** How many people have the word Chief in their job title? (This is pretty tricky) **

# In[33]:


sal['JobTitle'].apply(lambda str:('chief' in str.lower())).sum()


# # Great Job!
