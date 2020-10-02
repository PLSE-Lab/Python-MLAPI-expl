#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting graphs and variations
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting heatmaps and graphs


# Reading the csv file

# In[ ]:


data=pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')


# Data cleaning

# In[ ]:


data.shape #checking the shape of the dataset


# In[ ]:


data #knowing my dataset


# In[ ]:


data.columns #learning about the columns of my dataset


# We can see several unwanted columns. We will drop them.

# In[ ]:


data.drop(columns={'Respondent','Hobbyist','Employment','Student','EdLevel','UndergradMajor','EduOther','OrgSize','DevType','YearsCode','Age1stCode','YearsCodePro','JobSat','MgrIdiot','MgrMoney','MgrWant','JobSeek','LastHireDate','LastInt','FizzBuzz','JobFactors','ResumeUpdate','CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','WorkPlan','WorkChallenge','WorkRemote','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','MiscTechWorkedWith','MiscTechDesireNextYear','DevEnviron','OpSys','Containers','BlockchainOrg','BlockchainIs','BetterLife','ITperson','OffOn','Extraversion','ScreenName','SOVisit1st','SOVisitFreq','SOVisitTo','SOFindAnswer','SOTimeSaved','SOHowMuchTime','SOAccount','SOPartFreq','SOJobs','EntTeams','SOComm','WelcomeChange','OpenSource','SONewContent','Trans','Sexuality','Ethnicity','Dependents','SurveyLength','SurveyEase'},inplace = True)


# In[ ]:


data.head(5) #now checking our dataset again


# In[ ]:


data.shape #we can see that the no. of columns is reduced


# In[ ]:


data.info()


# In[ ]:


data.describe()


# **Assessing The Data**
# 
# Type Conversion
# 
# 1. MainBranch to Category type
# 2. OpenSourcer to Category type as 'Yes' or 'No'.
# 3. CareerSat to category type as this contain only 5 type of string.
# 4. WorkLoc to Category type as 'Ofice' or 'Home'.
# 5. Age to Integer type as age is alwees a whole number.
# 6. Gender to Category type as 'Male' or 'Female'
# 7. Quantity of incompleteness issue(almost every column contains empty data so we will try to fill them)
# 
# 
# NOTE
# * MainBranch Will remove all the empty data as Non-Developer.
# * Country We don't have other option as to remove the entry row without Country column.
# * CareerSat We will take a middle value for empty entry as all satisfied will comment on their satisfaction.
# * WorkWeekHrs to not disturb the observation we will replace the empty entry with mean of that column.
# * Gender we will take the empty entry as Transgender as they sometimes do not state their Gender.
# * Cleaning The Data.
# * So now we know the datatypes of the columns and hence now we will be converting them to their required data tyes.

# In[ ]:


data['MainBranch'].value_counts()


# In[ ]:


MainBranch = []
for i in data['MainBranch']:
    if i == 'I am a developer by profession':
        MainBranch.append('Developer')
    elif i == 'I am a student who is learning to code':
        MainBranch.append('Student')
    elif i == 'I am not primarily a developer, but I write code sometimes as part of my work':
        MainBranch.append('Partial Developer')
    elif i == 'I code primarily as a hobby':
        MainBranch.append('Hobby') 
    elif i == 'I used to be a developer by profession, but no longer am':
        MainBranch.append('Ex Developer')
    else:
        MainBranch.append('Student')
data['MainBranch'] = MainBranch
data['MainBranch'] = data['MainBranch'].astype('category',inplace=True)


# In[ ]:


data['OpenSourcer'].value_counts()


# In[ ]:


OpenSourcer = []
for i in data['OpenSourcer']:
    if (i == 'Never' ) or (i=='Less than once per year') or (i==''):
        OpenSourcer.append('No')
    else:
        OpenSourcer.append('Yes')
data['OpenSourcer'] = OpenSourcer
data['OpenSourcer'] = data['OpenSourcer'].astype('category')


# In[ ]:


data['CareerSat'].value_counts()


# In[ ]:


data['WorkLoc'].value_counts()


# In[ ]:


WorkLoc = []
for i in data['WorkLoc']:
    if i == 'Office':
        WorkLoc.append(i)
    else:
        WorkLoc.append('Home')
data['WorkLoc'] = WorkLoc
data['WorkLoc'] = data['WorkLoc'].astype('category')


# In[ ]:


data['Age'].value_counts()


# In[ ]:


Age = []
count = 0
for i in data['Age']:
    try:
        Age.append(int(i))
    except:
        Age.append(i)
        count += 1
data['Age'] = Age


# In[ ]:


count


# In[ ]:


data['Gender'].value_counts()


# In[ ]:


Gender = []
for i in data['Gender']:
    if (i=='Man') or (i== 'Man;Non-binary, genderqueer, or gender non-conforming'):
        Gender.append('Male')
    elif (i=='Woman') or (i=='Woman;Non-binary, genderqueer, or gender non-conforming') or (i=='Woman;Man;Non-binary, genderqueer, or gender non-conforming'):
        Gender.append('Female')
    else:
        Gender.append('Transgender')
data['Gender'] = Gender
data['Gender'] = data['Gender'].astype('category')


# **QUESTION : Which are the top most countries having the most developed software industry?**

# In[ ]:


data['Country'].head(50).value_counts().plot(kind='bar')


# **Analysis for India**

# In[ ]:


data_india=data[data['Country']=='India']
data_india_age=data_india[data_india['Age']>25]


# **AGE Vs MAINBRANCH**

# In[ ]:


sns.barplot(x='Age',y='MainBranch',data=data_india_age)


# **Developers love to work with which databases?**

# In[ ]:


data_india_age['DatabaseWorkedWith'].head(50).value_counts().plot(kind='bar')


# **What will be the databases that developers will be working with in the upcoming years?**

# In[ ]:


data_india_age['DatabaseDesireNextYear'].head(50).value_counts().plot(kind='bar')


# In[ ]:


data_india_age['LanguageDesireNextYear'].head(10).value_counts().plot(kind='bar')


# In[ ]:


sns.pairplot(data_india_age)


# ****What is the platform preferred by the developers in india?****

# In[ ]:


data_india_age['PlatformDesireNextYear'].head(10).value_counts().plot(kind='pie')


# **What will be the preferred webframe in the upcoming years in India?**

# In[ ]:


data_india_age['WebFrameDesireNextYear'].head(50).value_counts().plot(kind='bar')


# **What are the social medias used by the developers of india?**

# In[ ]:


sns.barplot(x='Age',y='SocialMedia',data=data_india_age)


# **Question :How many Developers are writing code for opensource?
# **

# In[ ]:


x = data.groupby('MainBranch')['OpenSourcer'].value_counts()
x = x.to_frame('Number_of_Developers')
x = x.reset_index()
sns.barplot(x='MainBranch',y='Number_of_Developers',hue='OpenSourcer',data=x)


# **Question :How much money do the Developers get throughout the globe?
# **

# In[ ]:


y = data['Country'].value_counts()[:30]
total = data.groupby('Country')['ConvertedComp'].sum()
data_plot1 = (total/y).sort_values(ascending=False)[:30]
data_plot = data_plot1.reset_index()
data_plot.rename(columns={0:'Average_income_in_USD','index':'Country_Name'},inplace=True)
sns.barplot(y='Country_Name',x='Average_income_in_USD',data = data_plot)


# **Question :How much are the Developers satisfied with their jobs throughout the globe?
# **

# In[ ]:


data_dev = data[data['MainBranch']=='Developer']
plot_data=data_dev['CareerSat'].value_counts().sort_index().reset_index()
plot_data.rename(columns={'index':'','CareerSat':'Number of Developers'},inplace=True)
sns.barplot(y='',x='Number of Developers',data=plot_data)


# **Question :Which type of work location is preferred by the Developers?
# **

# In[ ]:


data_plot_number = data_dev['WorkLoc'].value_counts().values
data_plot_name = data_dev['WorkLoc'].value_counts().index 
plt.pie(data_plot_number,labels=data_plot_name,autopct='%1.3f%%',)


# **overall analysis of developers who are above 25 years of age**

# In[ ]:


d=data[data['Age'] >25]


# **AGE Vs MAINBRANCH**

# In[ ]:


sns.barplot(x='Age',y='MainBranch',data=d)


# **Preferred database worked with by the developers all over the world**

# In[ ]:


d['DatabaseWorkedWith'].head(50).value_counts().plot(kind='bar')


# **What will the preferred language by the developers in the upcoming years?**

# In[ ]:


d['LanguageDesireNextYear'].head(10).value_counts().plot(kind='bar')


# **What will be the preferred database in the upcoming years?**

# In[ ]:


d['DatabaseDesireNextYear'].head(50).value_counts().plot(kind='bar')


# In[ ]:


sns.pairplot(d)


# In[ ]:


By Sharika Anjum Mondal

