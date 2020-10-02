#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting graphs and variations
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting heatmaps and graphs


    


# Gathering the data(Reading the csv file)

# In[ ]:


data=pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')
    


# Assessing the data

# In[ ]:


data.shape


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data['Employment'].value_counts()


# In[ ]:


data['MainBranch'].value_counts()


# Cleaning the data

# In[ ]:


data.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)


# In[ ]:


data.columns


# In[ ]:


data.drop(['Ethnicity','Dependents'],axis=1,inplace=True)


# In[ ]:


data['Age'].value_counts()


# Exploratory Data Analysis(EDA)

# Finding out the top most countries having the most developed software industry

# In[ ]:


data_developer=data[data['MainBranch']=='I am a developer by profession']
data_developer


# In[ ]:


data_developer_age=data_developer[data_developer['Age']>25]
data_developer_age


# In[ ]:


data_developer_age['Country'].head(50).value_counts().plot(kind='bar')


# In[ ]:


data_developer_age['Employment'].head(10).value_counts().plot(kind='bar')


# In[ ]:


data_learner=data[data['MainBranch']=='I am a student who is learning to code']
data_learner_age=data_learner[data_learner['Age']>20]

data_learner_age['Country'].head(50).value_counts().plot(kind='bar')
plt.xticks(rotation=90)


# Exploratory Data Analysis for top two countries
# 

# 1.For United States

# In[ ]:


data_us=data[data['Country']=='United States']
data_us_age=data_us[data_us['Age']>25]


# In[ ]:


sns.barplot(x='Age',y='EdLevel',data=data_us_age)


# In[ ]:


sns.barplot(x='Age',y='MainBranch',data=data_us_age)


# In[ ]:


sns.barplot(x='Age',y= 'Employment',data=data_us_age)


# In[ ]:


c=pd.crosstab(data_us_age['SOFindAnswer'],data_us_age['SOTimeSaved'])
sns.heatmap(c,cmap='autumn',linewidths=0.8)


# In[ ]:


data_us_age['YearsCode'].head(50).value_counts().plot(kind='pie')


# In[ ]:


sns.barplot(x='Age',y='UndergradMajor',data=data_us_age)


# In[ ]:


sns.barplot(x='Age',y='ITperson',data=data_us_age)


# In[ ]:


data_us_age['DatabaseWorkedWith'].head(50).value_counts().plot(kind='bar')


# In[ ]:


data_us_age['DatabaseDesireNextYear'].head(50).value_counts().plot(kind='bar')


# In[ ]:


data_us_age['LanguageDesireNextYear'].head(10).value_counts().plot(kind='bar')


# In[ ]:


sns.pairplot(data_us_age)


# 2.For India

# In[ ]:


data_india=data[data['Country']=='India']
data_india_age=data_us[data_us['Age']>25]
sns.barplot(x='Age',y='EdLevel',data=data_india_age)


# In[ ]:


sns.barplot(x='Age',y='MainBranch',data=data_india_age)


# In[ ]:


sns.barplot(x='Age',y= 'Employment',data=data_us_age)


# In[ ]:


c=pd.crosstab(data_india_age['SOFindAnswer'],data_india_age['SOTimeSaved'])
sns.heatmap(c,cmap='spring',linewidths=0.8)


# In[ ]:


sns.barplot(x='Age',y='UndergradMajor',data=data_india_age)


# In[ ]:


sns.barplot(x='Age',y='ITperson',data=data_india_age)


# In[ ]:


data_india_age['DatabaseWorkedWith'].head(50).value_counts().plot(kind='bar')


# In[ ]:


data_india_age['DatabaseDesireNextYear'].head(50).value_counts().plot(kind='bar')


# In[ ]:


data_india_age['LanguageDesireNextYear'].head(10).value_counts().plot(kind='bar')


# In[ ]:


sns.pairplot(data_india_age)


# In[ ]:


data_india_age['PlatformDesireNextYear'].head(10).value_counts().plot(kind='pie')



# In[ ]:


data_india_age['WebFrameDesireNextYear'].head(50).value_counts().plot(kind='bar')


# In[ ]:


sns.barplot(x='Age',y='SocialMedia',data=data_india_age)


# In[ ]:


data_india_age['JobFactors'].head(20).value_counts().plot(kind='bar')


# *By Chandrima Sarkar
# sarkarchandrima12@gmail.com*

# In[ ]:





# 

# 
