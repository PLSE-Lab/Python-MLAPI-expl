#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')
data1=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')


# In[ ]:


data.head()


# In[ ]:


data1.head()


# In[ ]:


data.shape


# In[ ]:


data.info


# In[ ]:


data.columns


# In[ ]:


data.corr()


# In[ ]:


data.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)


# In[ ]:


data


# In[ ]:


data['Employment'].value_counts()


# In[ ]:


data['Employment'].value_counts().plot(kind='bar')


# In[ ]:


data['MainBranch'].value_counts()


# In[ ]:


data['MainBranch'].value_counts().plot(kind='pie', autopct='%0.2f')


# In[ ]:


data['Country'].value_counts().head(10)


# In[ ]:


data['Country'].value_counts().head(10).plot(kind='bar')


# In[ ]:


data['Age'].value_counts().head(10).plot(kind='bar')


# In[ ]:


sns.barplot(x='Age',y= 'Employment',data=data)


# In[ ]:


data['UndergradMajor'].value_counts().head(10).plot(kind='bar')


# In[ ]:


data['SocialMedia'].value_counts().head(10)


# In[ ]:


data['SocialMedia'].value_counts().head(10).plot(kind='bar')


# In[ ]:


sns.barplot(x='Age',y= 'SocialMedia',data=data)


# In[ ]:


data['EdLevel'].value_counts()


# In[ ]:


data['EdLevel'].value_counts().plot(kind='bar')


# In[ ]:


sns.barplot(x='Age',y= 'EdLevel',data=data)


# In[ ]:




