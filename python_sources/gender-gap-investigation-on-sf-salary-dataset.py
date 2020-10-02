#!/usr/bin/env python
# coding: utf-8

# # Gender Gap Investigation on SF Salary Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Data Import and Initial Formating
# For this Kernel I am going to use 2 different datasets:
# - The publically available SF Salaries dataset

# In[ ]:


salaries_df = pd.read_csv('../input/sf-salaries/Salaries.csv', low_memory=False)
salaries_df.info()


# - A gender by name data set that contains names with the most probable gender for that name 

# In[ ]:


gender_by_name_df = pd.read_csv('../input/gender-by-name/gender.csv')
gender_by_name_df.info()


# I am going to start by making sure I have the right data types and that I fill possible gaps in the SF salaries dataset

# In[ ]:


for column in ['BasePay', 'OvertimePay', 'TotalPay', 'TotalPayBenefits']:
    salaries_df[column] = pd.to_numeric(salaries_df[column], errors='coerce')
    salaries_df[column].fillna(value=np.float64(0))
    
salaries_df.head()


# ## Merging the datasets to get the gender
# 
# The second step is to merge the two data frames to create a new dataframe that contains all SF Salaries information and the gender associated to each record

# In[ ]:


salaries_df2 = salaries_df.assign(name=salaries_df['EmployeeName'].apply(lambda x: x.split()[0].upper()))
salaries_gender_df = pd.merge(salaries_df2, gender_by_name_df, how='inner', on='name').drop('name', axis=1)
salaries_gender_df.info()
salaries_gender_df.head()


# In[ ]:


salaries_gender_groupby_gender=salaries_gender_df.groupby('gender')
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='EmployeeName', data=salaries_gender_groupby_gender.count().reset_index())


# Merging the table with gender_by_name dataset did a pretty good job of figuring out the gender for each record, however there is a number of rows marked as 'Unisex'. I am going to filter them to just keep Female and Male records.

# In[ ]:


salaries_gender_df=salaries_gender_df[(salaries_gender_df['gender'] == 'Male')|(salaries_gender_df['gender'] == 'Female') ]


# ## Looking at the average Salaries per Gender

# We start by grouping the data by gender so we can check the averages per gender

# In[ ]:


salaries_gender_groupby_gender=salaries_gender_df.groupby('gender')


# Let's start by looking the percentage of average salary that females get compared with males

# In[ ]:


avgpay_benefits = salaries_gender_groupby_gender.mean()['TotalPayBenefits']
avgpay_female = avgpay_benefits['Female']
avgpay_male = avgpay_benefits['Male']
str(round(avgpay_female*100/avgpay_male)) + '%'


# Let's see now the averages per gender in a bar diagram

# In[ ]:



plt.figure(figsize=(10, 6))
sns.barplot(x='gender',y='TotalPayBenefits',data=salaries_gender_groupby_gender.mean().reset_index())
plt.show()


# And in a distribution plot

# In[ ]:


plt.figure(figsize=(20, 8))
sns.distplot(salaries_gender_df[salaries_gender_df['gender']=='Female']['TotalPayBenefits'])
sns.distplot(salaries_gender_df[salaries_gender_df['gender']=='Male']['TotalPayBenefits'])
plt.show()


# # Analyzing the salaries grouping by Job Title

# We start by getting the group by salaries by JobTitle and Gender, and getting the count of EmployeeName and the mean of 'BasePay' and 'TotalPayBenefits'

# In[ ]:


salaries_per_job_and_gender = salaries_gender_df.groupby(['JobTitle','gender']).agg({'EmployeeName':'count', 'BasePay': 'mean', 'TotalPayBenefits':'mean'})       .rename(columns={'EmployeeName':'employeeCount','BasePay': 'BasePayAvg', 'TotalPayBenefits':'totalPayAvg'})       .reset_index()
salaries_per_job_and_gender.info()
salaries_per_job_and_gender.head()


# I realized that there is some number of jobs that have an small number of workers or a very small number of workers of one specific gender. So in order to remove outliers, I will just keep the records that have at least 30 workers

# In[ ]:


salaries_per_job_and_gender = salaries_per_job_and_gender[salaries_per_job_and_gender['employeeCount'] > 30]


# The following code is going to divide the table in one table for male gender and another for female gender, and then it is going to merge them by JobTitle, generating a single table with one record per job title and different columns for the male and female averages.

# In[ ]:


female_salaries_per_job = salaries_per_job_and_gender[salaries_per_job_and_gender['gender'] == 'Female'][['JobTitle', 'BasePayAvg', 'totalPayAvg']]
female_salaries_per_job.columns = ['JobTitle', 'FemaleBasePay', 'FemalePayBenefits']
male_salaries_per_job = salaries_per_job_and_gender[salaries_per_job_and_gender['gender'] == 'Male'][['JobTitle', 'BasePayAvg', 'totalPayAvg']]
male_salaries_per_job.columns = ['JobTitle', 'MaleBasePay', 'MalePayBenefits']
salaries_per_job_df = pd.merge(female_salaries_per_job, male_salaries_per_job, on='JobTitle', how='inner')
salaries_per_job_df.head()


# We calculate again the percentage of average salary that females get compared with males, this time by job position. I will sort them by percentage and I will print a bigger sample of the head and tail.

# In[ ]:


salaries_per_job_df = salaries_per_job_df.assign(DiffBasePay=salaries_per_job_df['FemaleBasePay']*100/salaries_per_job_df['MaleBasePay'])
salaries_per_job_df = salaries_per_job_df.assign(DiffPayBenefits=salaries_per_job_df['FemalePayBenefits']*100/salaries_per_job_df['MalePayBenefits'])
salaries_per_job_df.sort_values(by='DiffPayBenefits').head(20)


# In[ ]:


salaries_per_job_df.sort_values(by='DiffPayBenefits').tail(20)


# In 'DiffBasePay' and 'DiffPayBenefits', possitive number mean that the average for female is higher than the average for Male

# To get a better picture, let's check the 'median' and 'mean' of 'DiffBasePay' and 'DiffPayBenefits'

# In[ ]:


salaries_per_job_df.median()[['DiffBasePay', 'DiffPayBenefits']]


# In[ ]:


salaries_per_job_df.mean()[['DiffBasePay', 'DiffPayBenefits']]


# Now, we can check again the distribution of the differences per gender in 'Base Pay' and 'Pay with Benefits' by Job Title

# In[ ]:


plt.figure(figsize=(16, 10))
sns.distplot(salaries_per_job_df['DiffBasePay'],bins=30)
plt.show()


# 

# In[ ]:


plt.figure(figsize=(16, 10))
sns.distplot(salaries_per_job_df['DiffPayBenefits'],kde=True,bins=30)
plt.show()


# Finally we observe a small difference between DiffBasePay and DiffPayBenefits for both median and mean, so this last chart shows the relation between BasePay and OvertimePay per gender

# In[ ]:


sns.lmplot(x='BasePay', y='OvertimePay', hue='gender', data=salaries_gender_df, fit_reg=False, height=7, aspect=1.4)
plt.show()


# In[ ]:




