#!/usr/bin/env python
# coding: utf-8

# **SF Salaries Exercise**
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the SF Salaries Dataset from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.
# 

# In[ ]:


import pandas as pd
import sqlite3


# To load sqlite data

# In[ ]:


con = sqlite3.connect("../input/database.sqlite")
df = pd.read_sql_query('SELECT * FROM Salaries', con)
df


# Check the head of the DataFrame.

# In[ ]:


df.head()


# Use the .info() method to find out how many entries there are.

# In[ ]:


df.info()


# What is the average BasePay ?

# In[ ]:


#since we have str 'Not Provided' wehave to remove that
df['BasePay'].unique()


# In[ ]:


#locating 'Not Provided'
df.loc[df['BasePay'] == 'Not Provided']


# In[ ]:


#dreplace 'Not Provided' with zero's
#df.replace('Not Provided',0,inplace = True)


# In[ ]:


df.drop([148646,148650,148651,148652], axis =0,inplace=True)


# In[ ]:


#mean of BasePay
df['BasePay'].unique()
#df['BasePay'].mean()


# What is the highest amount of OvertimePay in the dataset ?

# In[ ]:


df['OvertimePay'].max()


# What is the job title of JOSEPH DRISCOLL ?

# In[ ]:


df.loc[df['EmployeeName'] == 'JOSEPH DRISCOLL'] ['JobTitle']


# How much does JOSEPH DRISCOLL make (including benefits)?

# In[ ]:


df[df['EmployeeName'] =='JOSEPH DRISCOLL']['TotalPayBenefits']


# What is the name of highest paid person (including benefits)

# In[ ]:


df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]


# In[ ]:


df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]['EmployeeName']


# What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?

# In[ ]:


df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]


# In[ ]:


df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]['EmployeeName']


# What was the average (mean) TotalPay of all employees per year? (2011-2014) ?

# In[ ]:


df['Year'].unique()


# In[ ]:


df.groupby('Year').mean()['TotalPay']


# How many unique job titles are there?

# In[ ]:


df['JobTitle'].nunique()


# What are the top 5 most common jobs?

# In[ ]:


df['JobTitle'].value_counts().head(5)


# How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)

# In[ ]:


sum(df[df['Year'] ==2013]['JobTitle'].value_counts()==1)


# How many people have the word Chief in their job title? (This is pretty tricky)

# In[ ]:


def chiefStr(title):
    if 'chief' in title.lower():
        return True
    else:
        return False


# In[ ]:


sum(df['JobTitle'].apply(lambda x : chiefStr(x)))


# Bonus: Is there a correlation between length of the Job Title string and Salary?

# In[ ]:


df['title_len'] = df['JobTitle'].apply(len)


# In[ ]:


df[['title_len','TotalPayBenefits']].corr()


# In[ ]:





# In[ ]:




