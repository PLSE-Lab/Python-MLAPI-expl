#!/usr/bin/env python
# coding: utf-8

# # SF Salaries Exercise 
# 
# Exercises are from Udemy course Python for [Data Science and Machine Learning Bootcamp ](http://www.udemy.com/share/10008ABUIad11SRng=/)

# ** Import pandas as pd.**

# In[ ]:


import pandas as pd
import numpy as np


# ** Read Salaries.csv as a dataframe called sal.**

# In[ ]:


sal = pd.read_csv('../input/Salaries.csv')


# ** Check the head of the DataFrame. ** <br>
# head() function returns the first n rows for the object based on position. Default value is 5 rows

# In[ ]:


sal.head()


# ** Use the .info() method to find out how many entries there are.**
# <br>
# info() method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

# In[ ]:


sal.info()


# **What is the average BasePay ?** <br>
# mean() function returns the mean of the values for the requested axis. By default axis = 0 (rows)

# In[ ]:


# Let's try to execute mean() function directly
sal['BasePay'].mean()


#  Error "**unsupported operand type(s) for +: 'float' and 'str**'" means that values in BasePay column have string values. <br> 
#  Let's check it

# In[ ]:


sal['BasePay']


# We can see not float value 'Not Provided'. It means that we should exclude it from result

# In[ ]:


# Change value Not Provided to Nan
sal[sal['BasePay'].str.contains('Not Provided', na=False)] = np.NaN


# In[ ]:


sal['BasePay'].mean()


# Error  still exists **unsupported operand type(s) for +: 'float' and 'str'**. <br>
# Let' convert values in float type

# In[ ]:


sal['BasePay'] = sal['BasePay'].apply(lambda x : float(x))


# In[ ]:


sal['BasePay'].mean()


# ** What is the highest amount of OvertimePay in the dataset ? **

# In[ ]:


sal[~sal['OvertimePay'].str.contains('Not Provided', na=False)]['OvertimePay'].apply(lambda x : float(x)).max()


# ** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **

# In[ ]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']


# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[ ]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']


# ** What is the name of highest paid person (including benefits)?**

# In[ ]:


sal.iloc[sal['TotalPayBenefits'].idxmax()]


# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[ ]:


sal.iloc[sal['TotalPayBenefits'].idxmin()]


# ** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **

# In[ ]:


sal.groupby('Year')['BasePay'].mean()


# ** How many unique job titles are there? **

# In[ ]:


sal['JobTitle'].nunique()


# ** What are the top 5 most common jobs? **

# In[ ]:


sal['JobTitle'].value_counts().head()


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[ ]:


sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)


# ** How many people have the word Chief in their job title? (This is pretty tricky) **

# In[ ]:


sum(sal['JobTitle'].str.lower().str.contains('chief', na=False))


# ** Bonus: Is there a correlation between length of the Job Title string and Salary? **

# In[ ]:


sal['title_len'] = sal['JobTitle'].str.len()


# In[ ]:


sal[['title_len','TotalPayBenefits']].corr()


# # Great Job!
