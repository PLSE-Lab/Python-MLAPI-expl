#!/usr/bin/env python
# coding: utf-8

# # SF Salaries Exercise 
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

# ** Import pandas as pd.**

# In[ ]:


import pandas as pd
import numpy as np


# ** Read Salaries.csv as a dataframe called sal.**

# In[ ]:


sal=pd.read_csv("../input/Salaries.csv")


# ** Check the head of the DataFrame. **

# In[ ]:


sal.head()


# ** Use the .info() method to find out how many entries there are.**

# In[ ]:


sal.info()


# **What is the average BasePay ?**

# In[ ]:


# sal["BasePay"].mean()


# ** What is the highest amount of OvertimePay in the dataset ? **

# In[ ]:


# sal["OvertimePay"].max()


# ** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **

# In[ ]:


sal[sal["EmployeeName"]=="JOSEPH DRISCOLL"]["JobTitle"]


# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[ ]:


sal[sal["EmployeeName"]=="JOSEPH DRISCOLL"]["TotalPayBenefits"]


# ** What is the name of highest paid person (including benefits)?**

# In[ ]:


# sal[sal["TotalPayBenefits"].max()]
sal[sal["TotalPayBenefits"]==sal["TotalPayBenefits"].max()]


# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[ ]:


sal[sal["TotalPayBenefits"]==sal["TotalPayBenefits"].min()]


# ** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **

# In[ ]:


# sal.groupby("Year")["BasePay"].mean()


# ** How many unique job titles are there? **

# In[ ]:


sal["JobTitle"].nunique()


# ** What are the top 5 most common jobs? **

# In[ ]:


sal["JobTitle"].value_counts().head(5)


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[ ]:


sum(sal[sal["Year"]==2013]["JobTitle"].value_counts()==1)


# ** How many people have the word Chief in their job title? (This is pretty tricky) **

# In[ ]:


def check(job):
    if "chief" in job.lower():
        return True
    else:
        return False


# In[ ]:


sum(sal["JobTitle"].apply(lambda x: check(x)))


# ** Bonus: Is there a correlation between length of the Job Title string and Salary? **

# In[ ]:


sal["JT_len"]=sal["JobTitle"].apply(len)


# In[ ]:


sal.head(2)


# In[ ]:


sal[["JT_len","TotalPayBenefits"]].corr()


# # Great Job!
