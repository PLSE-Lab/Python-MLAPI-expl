#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Importance Of Outlier Cleaning
# 
# This notebook illustrates the risks of outliers in numerical data. You can investigate the differences in the salaries of a dirty and a cleaned dataset.
# 
# The data comes from a [Stack Overflow Survey](https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey) in 2018.
# 
# [1 Import The Libraries](#1)    
# [2 Load The Dataset](#2)    
# [3 Compare Dirty- and Clean-Dataset](#3)    
# [4 Create A Mean And Median Dataset](#4)    
# [5 Compare Mean And Median](#5)    
# [6 Computing The Mean Median Differences](#6)    
# [7 Compare Mean/Median Differences](#7)    
# [8 Conclusion](#8)    
# 
# ### 1 Import The Libaries

# In[1]:


# To do linear algebra
import numpy as np

# To store the data
import pandas as pd

# To create plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# To create nicer plots
import seaborn as sns


# ### 2 Load The Dataset
# 
# Two distinct datasets will be created from the survey data. 
# 
# <br>
# 
# **Dirty Dataset** 
# 
# This set contains all yearly salaries of male and female employees without missing values.
# 
# <br></br>
# 
# **Clean Dataset** 
# 
# This set contains all yearly salaries below 500.000$ of male and female employees without missing values.
# <br>
# (The limit will be explained in the next cell.)

# In[3]:


# Load the data
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False)

# Store the dirty and create a clean dataset
dirty_df = df[df.Gender.isin(['Male', 'Female'])][['Gender', 'ConvertedSalary']].dropna()
clean_df = df[(df.ConvertedSalary<500000) & (df.Gender.isin(['Male', 'Female']))][['Gender', 'ConvertedSalary']].dropna()


# ### 3 Compare Dirty- and Clean-Dataset
# 
# Both upper plots depict the dirty dataset and both lower plots depict the clean dataset.
# <br>
# In the distribution of the dirty salary you can see some outliers claiming to earn several hundreds of thousand dollar.
# 
# Since the tail of the distribution drops quickly, a limit for the clean dataset at 500.000$ is reasonable.
# 
# Comparing the mean salary of both sets, you can see a decrease of several thousand dollars.

# In[4]:


# Create the figure and the subplot-sizes
fig = plt.figure(1, figsize=(15,5))
gridspec.GridSpec(2,4)

# Plot: Unclean distribution
ax = plt.subplot2grid((2,4), (0,0), rowspan=1, colspan=3)
sns.distplot(dirty_df.ConvertedSalary, ax=ax)
plt.title('Dirty ConvertedSalary Distribution')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('Frequency')

# Plot: Unclean ConvertedSalary
ax = plt.subplot2grid((2,4), (0,3))
sns.pointplot(data=dirty_df, x='Gender', y='ConvertedSalary', ax=ax)
plt.title('Dirty Data')
plt.xlabel('Gender')
plt.ylabel(' Mean ConvertedSalary')
plt.ylim([0, 105000])

# Plot: Clean distribution
ax = plt.subplot2grid((2,4), (1,0), rowspan=1, colspan=3)
sns.distplot(clean_df.ConvertedSalary, ax=ax)
plt.title('Clean ConvertedSalary Distribution')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('Frequency')

# Plot: Clean ConvertedSalary
ax = plt.subplot2grid((2,4), (1,3))
sns.pointplot(data=clean_df, x='Gender', y='ConvertedSalary', ax=ax)
plt.title('Clean Data')
plt.xlabel('Gender')
plt.ylabel('Mean ConvertedSalary')
plt.ylim([0, 105000])

# Show the plot
fig.tight_layout()
plt.show()


# ### 4 Create A Mean And Median Dataset
# 
# Computing the mean and the median for the salary of both datasets, we can explore their susceptibility to outliers.

# In[5]:


# Compute mean/median with dirty data
dirty_salary = df[df.Gender.isin(['Male', 'Female'])][['Gender', 'ConvertedSalary']].dropna().groupby('Gender').agg({'ConvertedSalary':['mean', 'median']})
dirty_salary.columns = dirty_salary.columns.get_level_values(1)
dirty_salary = dirty_salary.stack().reset_index().rename(columns={'level_1':'Type', 0:'Salary'})
dirty_salary['Status'] = 'dirty'

# Compute mean/median with clean data
clean_salary = df[(df.Gender.isin(['Male', 'Female'])) & (df.ConvertedSalary<500000)][['Gender', 'ConvertedSalary']].dropna().groupby('Gender').agg({'ConvertedSalary':['mean', 'median']})
clean_salary.columns = clean_salary.columns.get_level_values(1)
clean_salary = clean_salary.stack().reset_index().rename(columns={'level_1':'Type', 0:'Salary'})
clean_salary['Status'] = 'clean'

# Combine both datasets
salary = pd.concat([dirty_salary, clean_salary]).reset_index(drop=True)


# ### 5 Compare Mean And Median
# 
# While the mean is highly susceptible to the outliers in the datasets, the median has a more stable behaviour.

# In[6]:


# Compare the mean/median grouped by Status and Gender
sns.factorplot(data=salary, hue='Status', x='Type', y='Salary', col='Gender')
plt.show()


# ### 6 Computing The Mean Median Differences
# 
# Percental differences between the mean/median salaires of the dirty and the clean dataset are being computed here.

# In[7]:


# Compute the difference between the dirty and the clean dataset
dirty_salary_values = salary[salary.Status=='dirty'].Salary.values
clean_salary_values = salary[salary.Status=='clean'].Salary.values

difference = salary.copy().head(4).drop('Status', axis=1).rename(columns={'Salary':'Decrease'})
difference['Decrease'] = (dirty_salary_values - clean_salary_values) / dirty_salary_values


# ### 7 Compare Mean/Median Differences
# 
# By removing the outliers from the datasets, you can see the mean salaries decrease by roughly 32% while the median salaries decrease only by approximately 4%.

# In[8]:


# Compare thedifference in the mean/median grouped by Status and Gender
sns.barplot(data=difference, hue='Gender', x='Type', y='Decrease')
plt.title('Percental decrease between dirty and clean data')
plt.ylabel('Percental decrease')
plt.show()


# ### 8 Conclusion
# 
# Outliers can skew your whole exploration!
# 
# To tackle this problem you can either clean the outliers or use stable aggregation functions.
# <br>
# Be carful with numerical outliers and aggregating with the mean function.
# 
# For more information on outliers you can read this [R-notebook](https://www.kaggle.com/rtatman/data-cleaning-challenge-outliers) from the Kaggle-Team.
# 
# Please consider an upvote if this notebook has been helpful for you.
# <br>
# Have a good day!

# In[ ]:




