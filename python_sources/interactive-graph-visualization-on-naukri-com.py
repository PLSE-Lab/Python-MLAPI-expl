#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing essential libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Reading CSV file
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


df.head()


# In[ ]:


#Overall information about the dataset
df.info()


# In[ ]:


#This shows the unique value,total count, top values and top value frequency per column
df.describe()


# Graph for showing the null values in a each column, yellow markings shows the null entry.

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# From above heatmap, you can see that some of the column values are null.

# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


df['Industry'].value_counts()[0:9].iplot(kind='bar')


# IT-software, Human resources, call centers, banking and education industry seems to be having the highest number of jobs available on Naukri website.

# In[ ]:


df['Location'].value_counts()[0:9].iplot(kind='bar')


# This graph represents that most of the jobs are available in Bangalore, Mumbai, Pune and Hydrabad area. Bangalore totally has around 4986 jobs whereas, other locations has less than 3350 jobs.

# In[ ]:


softwarejobs = df[df['Industry'] == 'IT-Software, Software Services']
swjobs_locationCount = softwarejobs['Location'].value_counts()
swjobs_locationCount[0:9].iplot(kind='bar',colors='red')


# Above graph shows us the top locations providing 'IT software' industry jobs in which again Bangaluru has topped with more than 2100 jobs, whereas Hydrabad and Pune provides the same job opportunity with 950 jobs.
# 
# From above two graphs, you can say that more than 40% jobs in Bangaluru are related to IT industry.

# In[ ]:


df['Job Salary'].value_counts()[0:9].iplot(kind='bar')


# Most of the job salaries has not been disclosed by Recruiter.

# In[ ]:


df['Job Experience Required'].value_counts()[0:9].iplot(kind='bar')


# To summarize above graph in short, There are around 3000 jobs needs more than 1 year of job experience, 3600 jobs which needs more than 2 years experience, 2300 jobs with more than 3 years of experience.

# In[ ]:


df['Key Skills'].value_counts()[0:9].iplot(kind='bar')


# For around 100 jobs, required set of skills are counselor, teaching or customer services. But this graph does not give us a full picture of all 30000 jobs, because according to above graphs, most of the available jobs are related to IT industry and on this website, if you look at the provided data, key skills required in this industry have been described differently in each job description which is why we have to look at other attributes to understand this data more. So instead, let's have a look at 'Role Category' attribute.

# In[ ]:


df['Role Category'].value_counts()[0:9].iplot(kind='bar')


# This graph gives a better picture compared to 'Key Skills' attribute. Here you can see that, software related jobs with role category 'Programming and design' seems to be leading.

# In[ ]:


df['Role'].value_counts()[0:9].iplot(kind='bar')


# This graph gives an overall idea about job positions available on the website in which again software industry positions are highly demanding.

# In short, according to given data, Most of the jobs on Naukri.com are related IT and software industry which are hiring for programming and design role category and Most of the job recruiters are from Bangaluru region.
