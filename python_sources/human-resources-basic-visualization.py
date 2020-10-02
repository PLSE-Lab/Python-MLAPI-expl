#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pd.set_option('display.max_columns', 300)


# In[ ]:


hr = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv',index_col = 'EmpID')


# In[ ]:


hr.head(5)


# In[ ]:


hr.tail(5)


# In[ ]:


hr.shape


# # Clean Data

# In[ ]:


hr.dropna(axis = 0 ,how = 'all',inplace = True)


# In[ ]:


hr.shape


# In[ ]:


hr.info()


# In[ ]:


hr.describe()


# In[ ]:


hr.isnull().sum()


# In[ ]:


hr.drop(columns = ['Employee_Name','MaritalStatusID','HispanicLatino','Zip','Position','MarriedID','Termd','GenderID'], inplace = True)


# # gender distribution 

# In[ ]:


sns.countplot(x ='Sex',data=hr)


# 
# Our departments have more female than male 

# # age distribution

# In[ ]:


def calc_age(entity):
    current_year = pd.to_datetime('today').year
    birth_year = pd.to_datetime(entity).year
    if birth_year>current_year:
        birth_year = birth_year-100
    return current_year - birth_year


# In[ ]:



hr['age'] = hr['DOB'].map(calc_age)


# In[ ]:


sns.distplot(a=hr['age'],kde = False);


# the majority of our employees ages is between 35 and 40

# # Gender distribution  for each Department

# In[ ]:





# In[ ]:


plt.figure(figsize=(100,45))
sns.set(font_scale=5)
sns.countplot(x='Department',data=hr,hue='Sex');


# Production Department and Software Engineering hhave more female than male where IT/IS have more male

# # some  Author inspiration

# # Is there any relationship Manager and his Employees PerformanceScore?

# In[ ]:


hr['PerformanceScore'].value_counts()


# In[ ]:


grouped_data = hr.groupby(['ManagerName', 'PerformanceScore']).size().reset_index()
grouped_data.columns = ['ManagerName','PerformanceScore','Count']


# In[ ]:


grouped_data=grouped_data.pivot(columns='PerformanceScore', index='ManagerName', values='Count')


# In[ ]:



grouped_data.plot(kind='bar',stacked = True,figsize=(50,30))


# some managers like 'Michael Albert' have employees with needs improvement performance where  'David Stanley' Manager has high number of emploeyees with fully meet performane

# In[ ]:


grouped_data = hr.groupby(['ManagerName', 'EmpSatisfaction']).size().reset_index()
grouped_data.columns = ['ManagerName','EmpSatisfaction','Count']
grouped_data=grouped_data.pivot(columns='EmpSatisfaction', index='ManagerName', values='Count')


# In[ ]:


grouped_data.plot(kind='bar',stacked=True,figsize=(50,30))


# # Is there relationship between payrate and performance score

# In[ ]:


plt.figure(figsize=(16,6.5))
sns.set(font_scale=2)
sns.regplot(x='EmpSatisfaction',y='PayRate',data = hr);


# There is no clear reltionship that increasing pay rate will increase employees satisfication

# In[ ]:


hr['MaritalDesc'].value_counts()


# # marital statues distribution

# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(x = hr['MaritalDesc'])


# In[ ]:


grouped_data = hr.groupby(['MaritalDesc', 'PerformanceScore']).size().reset_index()
grouped_data.columns = ['MaritalDesc','PerformanceScore','Count']
grouped_data=grouped_data.pivot(columns='PerformanceScore', index='MaritalDesc', values='Count')


# In[ ]:


grouped_data.plot(kind = 'bar',stacked=True,figsize=(30,14),)


# most emplyees either married or single

# # is marital status effect thr performance score

# since we have a low number of emplyees who are seperated,devorsed or seperated we can't till , we need to collect more data

# # Best resourse for gathering Emplyees

# In[ ]:


plt.figure(figsize=(15,20))
sns.set(font_scale=2)
sns.countplot(y = 'RecruitmentSource',data = hr,order = hr['RecruitmentSource'].value_counts().index)


# most of our emplyees came from Employee referrial or diversty job fair

# # Is RaceDesc have effect on emplyees performance                
# 

# In[ ]:


hr['RaceDesc'].value_counts()


# In[ ]:


grouped_data = hr.groupby(['RaceDesc', 'PerformanceScore']).size().reset_index()
grouped_data.columns = ['RaceDesc','PerformanceScore','Count']
grouped_data=grouped_data.pivot(columns='PerformanceScore', index='RaceDesc', values='Count')


# In[ ]:


grouped_data.plot(kind = 'bar',stacked=True,figsize = (20,17))


# 
