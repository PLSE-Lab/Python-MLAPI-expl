#!/usr/bin/env python
# coding: utf-8

# # SF Salaries Exercise 
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

# ** Import pandas as pd.**

# In[ ]:


import pandas as pd


# ** Read Salaries.csv as a dataframe called sal.**

# In[ ]:


df = pd.read_csv('../input/Salaries.csv')


# ** Check the head of the DataFrame. **

# In[ ]:


df.head()


# ** Use the .info() method to find out how many entries there are.**

# In[ ]:


df.info()


# **What is the  average of first 10000 items in  BasePay ?**

# In[ ]:


df['BasePay'].head(1000).mean()


# ** What is the highest amount of TotalPayBenefits in the dataset ? **

# In[ ]:


df.loc[df['TotalPayBenefits'].idxmax()]['TotalPayBenefits']


# ** What is the job title of  JOSEPH DRISCOLL ?  **

# In[ ]:


df[df['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']


# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[ ]:


df[df['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']


# ** What is the name of highest paid person (including benefits)?**

# In[ ]:


#df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]['EmployeeName']
df.loc[df['TotalPayBenefits'].idxmax()]['EmployeeName']


# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[ ]:


df.loc[df['TotalPayBenefits'].idxmin()]['EmployeeName']


# ** What was the average (mean) TotalPay of all employees per year? (2011-2014) ? **

# In[ ]:


df.groupby('Year').mean()['TotalPay']


# ** How many unique job titles are there? **

# In[ ]:


len(df['JobTitle'].unique())
#df['JobTitle'].nunique()


# ** What are the top 7 most common jobs? **

# In[ ]:


df['JobTitle'].value_counts().head(7) #head default always return 5 top elements


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[ ]:


sum(df[df['Year']==2013]['JobTitle'].value_counts() == 1)


# ** How many people have the word Chief in their job title? (This is pretty tricky) **

# In[ ]:


def isInclude(title):
    if 'chief' in title.lower():
        return True
    else:
        return False    


# In[ ]:


sum(df['JobTitle'].apply(lambda x: isInclude(x)))


# ** Bonus: Is there a correlation between length of the Job Title string and Salary? **

# In[ ]:


df['title_len'] = df['JobTitle'].apply(len) 


# In[ ]:


df[['title_len','TotalPayBenefits']].corr() # No correlation.

