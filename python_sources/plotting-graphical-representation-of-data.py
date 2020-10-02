#!/usr/bin/env python
# coding: utf-8

# **Plotting graphical representation**
# 
# **Objective:** To present the graphical representation of data. The  data can be represented graphically. In fact, the graphical representation of statistical data is an essential step during statistical analysis.
# 
# Gaurav Mishra
# Dec, 12 2017

# Reading relevant libraries and data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# [](http://)Load the input file.

# In[2]:


data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()


# **Graphical Representation using various graph/plot**
# 
# ****Count Plot****

# In[3]:


sns.countplot(data.Age > 25)
plt.show()


# **Histogram Graph**
# 
# Attrition with Age

# In[4]:


atr_yes = data[data['Attrition'] == 'Yes']
atr_no = data[data['Attrition'] == 'No']
plt.hist(atr_yes['Age'])


# **Attrition rate: Job Level vs Gender**

# In[5]:


plt.figure(figsize=(12,8))
sns.barplot(x = data['Gender'], y = atr_yes['JobLevel'])


# Interestingly, As per Job level, Female attrition rate is little high.

# **Attrition rate: Job Level vs Job Role**

# In[6]:


sns.barplot(x = data['JobLevel'], y = atr_yes['JobRole'])


# **Box Plot**

# In[7]:


sns.boxplot(data['Gender'], data['MonthlyIncome'])
plt.title('MonthlyIncome vs Gender Box Plot', fontsize=12)      
plt.xlabel('MonthlyIncome', fontsize=12)
plt.ylabel('Gender', fontsize=12)
plt.show()


# **Mean of  Incomes  vs Genders**

# In[8]:


avg_male = np.mean(data.MonthlyIncome[data.Gender == 'Male'])
avg_female = np.mean(data.MonthlyIncome[data.Gender == 'Female'])
print(avg_female/avg_male)


# A woman makes 1.04 foreach1foreach1  a man earns.

# **Distribution plot**

# **Let's explore the income distribution for men and women:**

# In[9]:


sns.distplot(data.MonthlyIncome[data.Gender == 'Male'], bins = np.linspace(0,20000,60))
sns.distplot(data.MonthlyIncome[data.Gender == 'Female'], bins = np.linspace(0,20000,60))
plt.legend(['Males','Females'])


# Interestingly, the males histogram does not seem to be more to the right.

# **Now we will look at the different departments:**

# In[10]:


plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.title('Sales')
sns.distplot(data.MonthlyIncome[(data.Department == 'Sales') & (data.Gender == 'Male')])
sns.distplot(data.MonthlyIncome[(data.Department == 'Sales') & (data.Gender == 'Female')])
plt.xlabel('')

plt.subplot(3,1,2)
plt.title('R&D')
sns.distplot(data.MonthlyIncome[(data.Department == 'Research & Development') & (data.Gender == 'Male')])
sns.distplot(data.MonthlyIncome[(data.Department == 'Research & Development') & (data.Gender == 'Female')])
plt.xlabel('')

plt.subplot(3,1,3)
plt.title('HR')
sns.distplot(data.MonthlyIncome[(data.Department == 'Human Resources') & (data.Gender == 'Male')])
sns.distplot(data.MonthlyIncome[(data.Department == 'Human Resources') & (data.Gender == 'Female')])


# The average male at the sales department still earns less than the average female, though the difference is smaller compared to the other department

# **Income Distribution with Job Level**

# In[11]:


plt.figure(figsize = (10,10))
plt.subplot(2,1,1)
plt.plot(data.JobLevel,data.MonthlyIncome,'o', alpha = 0.01)
plt.xlabel('Job Level')
plt.ylabel('Monthly Income')


# **Total working years**

# In[12]:


sns.distplot(data.TotalWorkingYears, bins = np.arange(min(data.TotalWorkingYears),max(data.TotalWorkingYears),1))
plt.ylabel('Number of Employees')


# On an average employee wroks for 10 years

# **Scatter Plot**
# 
# Monthly / Daily vs Dally Rate Ratio

# In[13]:


_ = plt.scatter((data['MonthlyRate'] / data['DailyRate']), data['DailyRate'])
_ = plt.xlabel('Ratio of Monthly to Daily Rate')
_ = plt.ylabel('Daily Rate')
_ = plt.title('Monthly/Daily Rate Ratio vs. Daily Rate')
plt.show()


# **Joint plots**

# In[14]:


sns.jointplot(data.MonthlyIncome ,data.Age, kind = "scatter")   
plt.show()


# On an average, monthly income is less than 7k per month till age 40

# **Pair plots**

# In[15]:


cont_col= ['Age', 'PerformanceRating','MonthlyIncome','Attrition']
sns.pairplot(data[cont_col], kind="reg", diag_kind = "kde" , hue = 'Attrition' )
plt.show()

