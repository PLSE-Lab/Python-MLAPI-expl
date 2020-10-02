#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Reading Data

# In[ ]:


naukri = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


naukri.head()


# In[ ]:


naukri.tail()


# ## Structure and Summary

# In[ ]:


naukri.shape


# In[ ]:


naukri.info()


# In[ ]:


naukri.isnull().sum()


# ## EDA

# ### Removing Irrelevant Columns

# I believe 'Crawl Timestamp' and 'Functional Area' are irrelevant columns for our analysis so I'll drop these columns

# In[ ]:


naukri = naukri.drop(['Crawl Timestamp','Functional Area'], axis = 1)
naukri.head()


# ### Checking if any duplicate values are present in 'Uniq Id'

# In[ ]:


naukri['Uniq Id'].is_unique


# As every single Uniq Id is unique, that means we don't have any duplicate rows and hence we won't drop any rows.

# ### Top 10 'Job Experience Required'

# In[ ]:


job_exp_top_10 = pd.DataFrame(naukri['Job Experience Required'].value_counts()).head(10)
plt.figure(figsize=(10,8))
sns.barplot(data = job_exp_top_10, y = job_exp_top_10.index, x = 'Job Experience Required')
plt.title('Top 10 Range of Job Experience Required', size = 18)
plt.ylabel('Range of Job Experience', size = 15)
plt.xlabel('No. of Jobs Advertised', size = 15)
plt.show()


# As we can see, majority of the companies require employees between 2-5 and 2-7 years of experience. Hence majority of the candidates that are required should be either entry level with some experience, or junior level.

# ### Percentage of Top 10 Role Categories

# In[ ]:


role_top_10 = pd.DataFrame(naukri['Role Category'].value_counts(normalize = True).head(10) * 100)
plt.figure(figsize=(10,8))
plt.title('Top 10 Role Categories', size = 18)
sns.barplot(data = role_top_10, y = role_top_10.index, x = 'Role Category')
plt.ylabel('Role Categories',size = 15)
plt.xlabel('Percentage of roles advertised', size = 15)
plt.show()


# We can see that 'Programming and Design' Roles take more than 30% of all job roles advertised on Naukri.com
# 
# This is not shocking as a lot of foreign companies outsource their IT services/software development from India.

# ### Top 10 locations that are hiring in the 'Programming and Design' Role category

# In[ ]:


loc_top_10 = pd.DataFrame(naukri[naukri['Role Category'] == 'Programming & Design'].Location.value_counts().head(10)) #percentage of top 10 Role Categories advertised on Naukri.com
plt.figure(figsize=(10,8))
plt.title('Top 10 Locations in the "Programming and Design" Role Category', size = 18)
sns.barplot(data = loc_top_10, y = 'Location', x = loc_top_10.index)
plt.ylabel('No. of Job Opportunities in various cities',size = 15)
plt.xlabel('Locations', size = 15)
plt.show()


# Bangalore, being the IT Hub of India, has more job opportunities for 'Programming & Design' Roles by a very high margin when compared to other cities.

# ### Percentage of Top 10 Industries that have advertised on Naukri.com

# In[ ]:


industry_top_10 = pd.DataFrame(naukri.Industry.value_counts(normalize = True).head(10) * 100)
plt.figure(figsize=(10,8))
plt.title('Top 10 Industries', size = 18)
sns.barplot(data = industry_top_10, y = industry_top_10.index, x = 'Industry')
plt.ylabel('Industries',size = 15)
plt.xlabel('Percentage of Jobs advertised by Industries', size = 15)
plt.show()


# It's no surprise that IT Software and Software Services are the industries that have advertised the most on Naukri.com, as we have seen before that Programming and Design Roles are the most advertised Role categories as well.

# In[ ]:




