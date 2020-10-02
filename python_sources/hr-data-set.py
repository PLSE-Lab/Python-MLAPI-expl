#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## **DATA OVERVIEW**

# In[ ]:


data = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


print('Employee stayed : ',data[data['left'] == 0]['left'].value_counts())
print('Employee left : ',data[data['left'] == 1]['left'].value_counts())


# > *Number of employees left : **3571***
# 
# > *Number of employees stayed : **11428***

# ## REASONS THAT SEEMS RATIONAL ##
# 
# Below may be the possible reasons for declining employee number :
# 
# * If they are not promoted in last five years
# * If given less payroll
# * Unortunate work accident
# 
# 
# 

#  ###  **1 . PROMOTION OFFERED** ###

# In[ ]:


fig = plt.figure(figsize = (8,4))
sns.countplot(data['promotion_last_5years'], hue=data['left'], palette='Paired')
plt.suptitle('WHETHER PROMOTED IN THE LAST FIVE YEARS', fontsize=14, fontweight='bold')
plt.legend(title='Left')
plt.xlabel('Promotion in last 5years')
plt.ylabel('Number of employees')


# It is clear that the employees ( little less than 4000) might have left the company as they are not satisfied with their promotion process.

#  ###  **2. SALARY EXPECTATION** ###

# In[ ]:


fig = plt.figure(figsize = (8,4))
sns.countplot(data['salary'], hue=data['left'], palette='coolwarm')
plt.suptitle('SALARY LEVEL', fontsize=14, fontweight='bold')
plt.legend(title='Left')
plt.xlabel('Salary offered')
plt.ylabel('Number of employees')


# It seems employees who were offered low to medium salary packages left the company.

# ###  **3. WORK ACCIDENT** ###

# In[ ]:


fig = plt.figure(figsize = (8,4))
sns.countplot(data['Work_accident'], hue=data['left'], palette='Paired')
plt.suptitle('HAD ANY WORK ACCIDENT', fontsize=14, fontweight='bold')
plt.legend(title='Left')
plt.xlabel('Work accident')
plt.ylabel('Number of employees')


# Many employees left the company even though they have not had any work accident. So, work accident may not generalise the reason for employee leaving the company.

# ## ANALYSING WITH REASONS
# 
# From the above three categorical data, we can reasonably assme that "Promotion" and "Salary" may be the main reasons for employees leaving the company. Lets justify this assumption with further analysis:

#  **LAST EVALUATION (VS) AVERAGE MONTHLY HOURS**

# In[ ]:


sns.lmplot(x='last_evaluation', y='average_montly_hours', data=data, hue='left', fit_reg=False)


# In[ ]:


sns.lmplot(x='last_evaluation', y='average_montly_hours', data=data, col='left', hue='promotion_last_5years', fit_reg=False)


# We have 2 major cluster of people leaving the company:
# 
# * People who worked for more hours and got evaluation point greater than 0.75
# 
# * People who worked for less hours and having evaluation point  in between 0.4 and 0.6
# 
# In both cases, people are not offered promotion in the last five years

# **NUMBER OF PROJECT (VS) AVERAGE MONTHLY HOURS**

# In[ ]:


sns.lmplot(x='number_project', y='average_montly_hours', data=data, col='left', hue='promotion_last_5years', fit_reg=False, palette='coolwarm')


# In[ ]:


sns.lmplot(x='number_project', y='average_montly_hours', data=data, col='left', hue='salary', fit_reg=False, palette='coolwarm')


#  By comparing the above two graphs, we can intrepret that people who had worked in more number of projects and spent quality amont of working hours had left the organisation. In this comparision the reason is apparently due to medium salary and promotion process. 

# **TIME SPENT IN COMPANY (VS) AVERAGE MONTHLY HOURS**

# In[ ]:


sns.lmplot(x='time_spend_company', y='average_montly_hours', data=data, col='left', hue='promotion_last_5years', fit_reg=False)


# In[ ]:


sns.lmplot(x='time_spend_company', y='average_montly_hours', data=data, col='left', hue='salary', fit_reg=False)


# People who worked less than or equal to 6 years had left the organisation due to less salary or promotion.

#  **SATISFACTION LEVEL**

# In[ ]:


empLeft = data[data['left'] == 0]
empStayed = data[data['left'] == 1]


# In[ ]:


fig = plt.figure(figsize = (10,8))
plt.hist(empLeft['satisfaction_level'], bins=20, alpha = 0.7, label='Employee Left',color='black')
plt.hist(empStayed['satisfaction_level'],bins=20,alpha = 0.2, label='Employee Stayed',color='blue')
plt.legend()
plt.xlabel('Employees satifaction level')
#plt.ylabel('No of employees')


# Obviously , many people who had shown less that 50% satisfaction left the company.

# **DEPARTMENT** 

# In[ ]:


fig = plt.figure(figsize=(12,6))
sns.countplot(data['sales'], hue=data['left'])


# ## **CONCLUSION**
# 
# From this data analysis, we can be sure that the major reasons for people leaving company is due to :
# 
# * Promotion
# * Salary
# 
# Due to this majority of the people are leaving from the ***Sales, Accounting and Technical ***departments compared to other departments.
# 
# Even salary seems to have slighly less impact on employees than promotion.. So company must consider changing their promotion process. If that is not rising the employees satisfaction level then they can consider revising the salary packages.
