#!/usr/bin/env python
# coding: utf-8

# # San Francisco Salaries (2011-2014)

# In[ ]:


import pandas as pd


# ** Reading in the data **

# In[ ]:


sal = pd.read_csv('../input/Salaries.csv')
sal['BasePay'] = pd.to_numeric(sal['BasePay'], errors='coerce')
sal['OvertimePay'] = pd.to_numeric(sal['OvertimePay'], errors='coerce')
sal['OtherPay'] = pd.to_numeric(sal['OtherPay'], errors='coerce')
sal['Benefits'] = pd.to_numeric(sal['Benefits'], errors='coerce')


# ** Checking the format of the data **

# In[ ]:


sal.head()


# In[ ]:


sal.info()


# ** Finding the average base pay **

# In[ ]:


print("Average Base Pay: ${}".format(round(sal['BasePay'].mean(), 2)))


# ** Finding the highest base pay **

# In[ ]:


print("The highest base pay is ${}".format(round(sal['BasePay'].max(), 2)))


# ** What is the highest amount of OvertimePay in the dataset ? **

# In[ ]:


sal['OvertimePay'].max()


# ** Data for the person with the highest total compensation**

# In[ ]:


sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]


# ** Data for the person with the lowest total compensation **

# In[ ]:


sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]


# For some reason this person has no base pay, no overtime pay, no benefits, and actually owes $618.13.

# ** Average base pay over the entire time horizon (2011-2014) as well as a per-year basis **

# In[ ]:


print("The average base pay from 2011-2014 was ${}".format(round(sal['BasePay'].mean(), 2)))


# In[ ]:


sal.groupby('Year').mean()['BasePay']


# ** Number of unique job titles **

# In[ ]:


print("There were {} unique job titles in this data set.".format(sal['JobTitle'].nunique()))


# ** Top 10 most common job titles **

# In[ ]:


sal['JobTitle'].value_counts()[:10]


# ** Job Titles represented by only one person in 2013**

# In[ ]:


len(sal[sal['Year'] == 2013]['JobTitle'].value_counts()[sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1])


# In[ ]:


sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)


# ** Checking for correlations **

# Title Length vs Base Pay

# In[ ]:


sal['LenTitle'] = sal['JobTitle'].apply(len)

sal[['LenTitle', 'BasePay']].corr()


# Title Length vs Other Pay

# In[ ]:


sal['LenTitle'] = sal['JobTitle'].apply(len)
sal[['LenTitle', 'OtherPay']].corr()


# There is seemingly no correlation.

# ** Average Police total compensation vs Fire Department total compensation **

# In[ ]:


police_mean = sal[sal['JobTitle'].str.lower().str.contains('police')]['TotalPayBenefits'].mean()
fire_mean = sal[sal['JobTitle'].str.lower().str.contains('fire')]['TotalPayBenefits'].mean()
print("On average, people whose title includes 'police' make ${:,} in total compensation.".format(round(police_mean,2)))
print("On average, people whose title includes 'fire' make ${:,} in total compensation. \n".format(round(fire_mean,2)))

pct_diff = (fire_mean - police_mean) * 100 / police_mean
print("People whose title includes 'fire' have a {:.2f}% higher total compensation than those whose title includes 'police'.".format(pct_diff))

