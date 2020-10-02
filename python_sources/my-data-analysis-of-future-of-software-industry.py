#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Gathering Data

# In[ ]:


data=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")


# In[ ]:


data.head()


# ASSESSING DATA

# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.corr()


# In[ ]:


data.info()


# In[ ]:


data['Employment'].value_counts()


# In[ ]:


data['MainBranch'].value_counts()


# In[ ]:


data_student=data[data['MainBranch']=='I am a student who is learning to code']


# In[ ]:


data_student_india=data_student[data_student['Country']=='India']


# In[ ]:


data_student_india.shape


# DATA CLEANING

# In[ ]:


data_student_india.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs'],axis=1,inplace=True)


# In[ ]:


data_student_india.drop(['WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat'],axis=1,inplace=True)


# In[ ]:


data_student_india.drop(['OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)


# In[ ]:


data_student_india_age=data_student_india[data_student_india['Age']>22]


# In[ ]:


data_student_india_age.head()


# EXPLORATORY DATA ANALYSIS

# In[ ]:


data_student_india_age['UndergradMajor'].value_counts().plot(kind='bar')


# In[ ]:


ct=pd.crosstab(data_student_india_age['SOFindAnswer'],data_student_india_age['SOTimeSaved'])


# In[ ]:


sns.heatmap(ct)


# In[ ]:


ct.plot(kind='bar')


# In[ ]:


sns.barplot(x='Age',y='SocialMedia',data=data_student_india)


# In[ ]:


sns.barplot(x='Age',y='EdLevel',data=data_student_india)


# In[ ]:


sns.pairplot(data_student_india_age)


# In[ ]:


sns.distplot(data_student_india_age['Age'],bins=30)


# In[ ]:


data_student_india_age.columns


# In[ ]:


sns.barplot(x='Gender',y='Age',data=data_student_india_age,estimator=np.std)


# In[ ]:


sns.violinplot(x='YearsCode',y='Age',data=data_student_india_age,split=True)


# In[ ]:


sns.stripplot(x='BlockchainIs',y='Age',data=data_student_india_age,jitter=True)


# In[ ]:


sns.barplot(x='Age',y='YearsCode',data=data_student_india_age)


# In[ ]:


sns.barplot(x='Age',y='Ethnicity',data=data_student_india_age)


# In[ ]:


sns.heatmap(pd.crosstab(data_student_india_age['DatabaseWorkedWith'],data_student_india_age['Ethnicity']))


# In[ ]:


sns.barplot(x='Age',y='ITperson',data=data_student_india_age)


# In[ ]:


data_student_india_age['Ethnicity'].value_counts().plot(kind='pie',autopct='%0.2f')


# In[ ]:




