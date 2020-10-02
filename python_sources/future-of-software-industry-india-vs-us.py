#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data['Employment'].value_counts()


# In[ ]:


data['BetterLife'].value_counts()


# In[ ]:


data['ITperson'].value_counts()


# In[ ]:


data['MainBranch'].value_counts()


# In[ ]:


data['Hobbyist'].value_counts()


# In[ ]:


data['Respondent'].value_counts()


# In[ ]:


data['UndergradMajor'].value_counts()


# In[ ]:


data['Age'].value_counts()


# In[ ]:


data['Country'].value_counts()


# In[ ]:


data['YearsCode'].value_counts()


# In[ ]:


data_student=data[data['MainBranch']=='I am a student who is learning to code']


# In[ ]:


data_student_india=data_student[data_student['Country']=='India']


# In[ ]:


data_student_US=data_student[data_student['Country']=='United States']


# In[ ]:


data_hobby=data[data['MainBranch']=='I code primarily as a hobby']


# In[ ]:


data_hobby_india=data_hobby[data_hobby['Country']=='India']


# In[ ]:


data_hobby_US=data_hobby[data_hobby['Country']=='United States']


# In[ ]:


data_student_india.shape


# In[ ]:


data_student_US.shape


# In[ ]:


data_hobby_india.shape


# In[ ]:


data_hobby_US.shape


# A Comparision between India and US

# In[ ]:


data_hobby_india=data_hobby[data_hobby['Country']=='India']
sns.barplot(y='YearsCode',x='Age',data=data_hobby_india,color='red')
data_hobby_US=data_hobby[data_hobby['Country']=='United States']
sns.barplot(y='YearsCode',x='Age',data=data_hobby_US,color='blue')


# In[ ]:


data_student_india.columns


# Data cleaning

# In[ ]:


data_student_india.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)


# In[ ]:


data_student_US.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)


# In[ ]:


data_student_india.shape


# In[ ]:


data_student_US.shape


# In[ ]:


data_student_india_age=data_student_india[data_student_india['Age']>20]


# In[ ]:


data_student_US_age=data_student_US[data_student_US['Age']>20]


# EDA

# In[ ]:


data_student_india_age['UndergradMajor'].value_counts().plot(kind='barh')


# In[ ]:


data_student_US_age['UndergradMajor'].value_counts().plot(kind='barh',color='red')


# In[ ]:


data['LanguageWorkedWith'].value_counts()


# In[ ]:





# In[ ]:


data_student_india=data_student[data_student['Country']=='India']
sns.barplot(x='Age',y='SocialMedia',data=data_student_india)


# In[ ]:


data_student_US=data_student[data_student['Country']=='United States']
sns.barplot(x='Age',y='SocialMedia',data=data_student_US)


# In[ ]:


sns.barplot(x='Age',y='ITperson',data=data_student_india_age)


# In[ ]:


sns.barplot(x='Age',y='ITperson',data=data_student_US_age)


# In[ ]:


sns.barplot(x='Age',y='Ethnicity',data=data_student_india_age)


# In[ ]:


sns.barplot(x='Age',y='Ethnicity',data=data_student_US_age)


# In[ ]:


sns.barplot(x='Age',y='EdLevel',data=data_student_india_age)


# In[ ]:


sns.barplot(x='Age',y='EdLevel',data=data_student_US_age)


# In[ ]:


sns.barplot(y='Gender',x='Age',data=data_student_india_age)


# In[ ]:


sns.barplot(y='Gender',x='Age',data=data_student_US_age)


# In[ ]:


data_student_india.columns


# In[ ]:


data_student_india_age['BetterLife'].value_counts()


# In[ ]:


sns.barplot(y='Employment',x='Age',data=data_student_india)


# In[ ]:


sns.barplot(y='Employment',x='Age',data=data_student_US)


# In[ ]:


sns.barplot(y='Dependents',x='Age',data=data_student_india_age)


# In[ ]:


sns.barplot(y='Dependents',x='Age',data=data_student_US_age)


# In[ ]:


data_student_india_age['BetterLife'].value_counts().plot(kind='pie',autopct='%0.2f')


# In[ ]:


data_student_US_age['BetterLife'].value_counts().plot(kind='pie',autopct='%0.2f')


# In[ ]:




