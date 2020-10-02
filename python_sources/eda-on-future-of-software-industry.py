#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# **Gathering Data** 

# In[ ]:


data=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')


# In[ ]:


df=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')


# **Assessing The data**

# In[ ]:


data.head()


# In[ ]:


df.head()


# we will be working with the first data only, as the second one is the explaination of the columns of first data.

# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.info()


# **Cleaning The Data**

# In[ ]:


data.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)


# In[ ]:


data.shape


# **The EDA**

# In[ ]:


data['Employment'].value_counts()


# In[ ]:


data['Employment'].value_counts().head(3).plot(kind='pie',autopct='%0.2f')


# In[ ]:


data['MainBranch'].value_counts()


# In[ ]:


data['MainBranch'].value_counts().head(3).plot(kind='pie',autopct='%0.2f')


# In[ ]:


a=data['MainBranch'].value_counts()


# In[ ]:


b=data['MainBranch'].value_counts()>3500


# In[ ]:


c=a[b].index.tolist()
c


# In[ ]:


data=data[data['MainBranch'].isin(c)]


# In[ ]:


data.shape


# In[ ]:


data['Hobbyist'].value_counts()


# In[ ]:


data['Hobbyist'].replace({'Yes':1,'No':0},inplace=True)


# In[ ]:


data['Country'].value_counts().head(10)
#top 10 Country with most number of upcoming developers


# In[ ]:


data_india=data[data['Country']=='India']
sns.barplot(y='YearsCode',x='Age',data=data_india,color='red')
data_US=data[data['Country']=='United States']
sns.barplot(y='YearsCode',x='Age',data=data_US,color='blue')
data_germany=data[data['Country']=='Germany']
sns.barplot(y='YearsCode',x='Age',data=data_US,color='green')


# In[ ]:


data['UndergradMajor'].value_counts().head(5).plot(kind='pie',autopct='%0.2f')
#top 5 streams people are studying as their main field


# In[ ]:


data['Age'].value_counts().head(10).plot(kind='bar')
#top 10 age groups who codes


# In[ ]:


sns.barplot(x='Age',y= 'Employment',data=data)


# In[ ]:


data['BetterLife'].value_counts().plot(kind='pie',autopct='%0.2f')


# In[ ]:


data['Gender'].value_counts()


# In[ ]:


data['Gender'].value_counts().head(2).plot(kind='pie',autopct='%0.2f')
#the two genders who codes most


# In[ ]:


data['OpenSourcer'].value_counts()


# In[ ]:


data['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')
#checking for percentage


# In[ ]:


data_india=data[data['Country']=='India']
data_us=data[data['Country']=='United States']
data_germany=data[data['Country']=='Germany']


# In[ ]:


#using the top 3 countries for comparison


# In[ ]:


data_india['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')


# In[ ]:


data_us['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')


# In[ ]:


data_germany['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')


# In[ ]:


data['UndergradMajor'].value_counts().head(5).plot(kind='pie',autopct='%0.2f')
#top 5 subjects which people opt as their main field


# In[ ]:


data_india['EdLevel'].value_counts().head(5)


# In[ ]:


data_us['EdLevel'].value_counts().head(5)


# In[ ]:


data_germany['EdLevel'].value_counts().head(5)


# In[ ]:


data['SocialMedia'].value_counts().head(5).plot(kind='pie',autopct='%0.2f')


# In[ ]:





# In[ ]:




