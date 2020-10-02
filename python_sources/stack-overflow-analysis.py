#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing Dataset

# In[ ]:


data=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")


# Exploring the data

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.drop(columns={'Respondent','OpenSource','Student','EdLevel','EduOther','OrgSize','YearsCode','Age1stCode','MgrIdiot','MgrMoney','MgrWant','JobSeek','LastHireDate','LastInt','JobFactors','ResumeUpdate','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkRemote','ImpSyn','UnitTests','PurchaseHow','PurchaseWhat','OffOn','SocialMedia','Extraversion','ScreenName','SOVisit1st','SOVisitFreq','SOVisitTo','SOFindAnswer','SOTimeSaved','SOHowMuchTime','SOAccount','SOPartFreq','SOJobs','EntTeams','SOComm','SONewContent','Gender','Trans','Sexuality','Ethnicity','Dependents','SurveyLength','SurveyEase','FizzBuzz','CurrencySymbol','WelcomeChange','WorkPlan','WorkChallenge'},inplace=True)


# In[ ]:


data.shape


# **ANALYSIS ON MAIN BRANCH**

# In[ ]:


data['MainBranch'].replace({'I am a student who is learning to code':'Student','I am not primarily a developer, but I write code sometimes as part of my work':'Not a Developer','I am a developer by profession':'Developer', 'I code primarily as a hobby':'Code as hobby','I used to be a developer by profession, but no longer am':'Was a Developer'},inplace=True)


# In[ ]:


main_branch=data['MainBranch'].dropna()
plt.figure(figsize=(20,5))
sns.countplot(x=main_branch)
plt.xlabel('Branch Type', fontsize=10)
plt.ylabel('Count')
plt.title("Main Branch")
plt.show()


# In[ ]:


data_dev=data[data['MainBranch']=='Developer']


# **ANALYSIS ON HOBBYIST**

# In[ ]:


sns.countplot(x='Hobbyist',data=data_dev)
plt.xlabel('Answers')
plt.ylabel('Count')
plt.title("Do you code as a hobby?")
plt.show()


# **ANALYSIS ON OPENSOURCER**

# In[ ]:


plt.figure(figsize=(20,7))
sns.countplot(x='OpenSourcer',data=data_dev)
plt.xlabel('Frequency of contribution to OpenSource', fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title("How often do you contribute to open source?",fontsize=20)
plt.show()


# **ANALYSIS ON YEARS OF CODE AS A PROFESSIONAL**

# In[ ]:


data_dev['YearsCodePro'].unique()


# In[ ]:


data_dev['YearsCodePro'].replace({'Less than 1 year':0,'More than 50 years':51},inplace=True)
years_code=data_dev['YearsCodePro'].dropna()
years_code=years_code.astype(int)
max_year=years_code.max()
if(max_year==51):
    print ('Maximum Years of coding as a professional is more than 50 years')
else:
    print ('Maximum Years of coding as a professional is ',max_year,'years')


# In[ ]:


data_dev['YearsCodePro'].value_counts().head(2)


# In[ ]:


min_year=years_code.min()
if(min_year==0):
    print ('Minimum Years of coding as a professional is less than 1 year')
else:
    print ('Minimum Years of coding as a professional is ',min_year,'years')


# **ANALYSIS ON EMPLOYMENT STATUS OF DEVELOPERS**

# In[ ]:


plt.figure(figsize=(25,8))
sns.countplot(x='Employment',data=data_dev)
plt.xlabel('Employment Status', fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title("Employment Status of Developers",fontsize=20)
plt.show()


# **ANALYSIS IN UNDERGRAD MAJOR**

# In[ ]:


data_dev['UndergradMajor'].value_counts().head()


# **ANALYSIS IN DEVELOPMENT TYPE**

# In[ ]:


data_dev['DevType'].value_counts().head()


# **ANALYSIS ON CAREER SATISFACTION**

# In[ ]:


plt.figure(figsize=(20,7))
sns.countplot(x='CareerSat',data=data_dev)
plt.xlabel('Career Satisfaction Status', fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title("Career Satisfaction Status of Developers",fontsize=20)
plt.show()


# In[ ]:


#marking career satifaction on a scale of 0 to 4, 2 being neutral
data_dev['CareerSat'].replace({'Slightly satisfied':1, 'Very satisfied':4, 'Very dissatisfied':0,
       'Slightly dissatisfied':3, 'Neither satisfied nor dissatisfied':2},inplace=True)


# In[ ]:


career_sat=data_dev.groupby('Country')['CareerSat'].sum().sort_values(ascending=False).head().reset_index()
career_sat.plot.bar(x='Country',y='CareerSat')
plt.xlabel('Country',fontsize=10)
plt.ylabel("Career Satisfaction Rating Sum",fontsize=10)
plt.show()


# **ANALYSIS ON JOB SATISFACTION**

# In[ ]:


plt.figure(figsize=(20,7))
sns.countplot(x='JobSat',data=data_dev)
plt.xlabel('Job Satisfaction Status', fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title("Job Satisfaction Status of Developers",fontsize=20)
plt.show()


# In[ ]:


#marking job satifaction on a scale of 0 to 4, 2 being neutral
data_dev['JobSat'].replace({'Slightly satisfied':1, 'Very satisfied':4, 'Very dissatisfied':0,
       'Slightly dissatisfied':3, 'Neither satisfied nor dissatisfied':2},inplace=True)


# In[ ]:


career_sat=data_dev.groupby('Country')['JobSat'].sum().sort_values(ascending=False).head().reset_index()
career_sat.plot.bar(x='Country',y='JobSat')
plt.xlabel('Country',fontsize=10)
plt.ylabel("Job Satisfaction Rating Sum",fontsize=10)
plt.show()


# **ANALYSIS ON WORK WEEK HOURS**

# In[ ]:


work_hrs=data_dev.groupby('Country')['WorkWeekHrs'].sum().sort_values(ascending=False).head().reset_index()


# In[ ]:


work_hrs.plot.bar(x='Country',y='WorkWeekHrs')
plt.xlabel('Country',fontsize=10)
plt.ylabel("Work hrs/week",fontsize=10)
plt.show()


# In[ ]:


data.columns


# **ANALYSIS ON WORK LOCATION**

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='WorkLoc',data=data_dev)
plt.xlabel('Work Location', fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title("Preferred Work Location",fontsize=20)
plt.show()


# In[ ]:


work_loc=data_dev['WorkLoc'].value_counts().reset_index()
colors=['purple','violet','pink']
plt.figure(figsize=(10,8))
plt.pie(work_loc['WorkLoc'],labels=work_loc['index'],colors=colors,startangle=90, autopct='%.1f%%')
plt.show()


# **ANALYSIS ON CODE REVIEW**

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='CodeRev',data=data_dev)
plt.xlabel('Code Review Status', fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title("Do you review code as part of your work?",fontsize=20)
plt.show()


# **ANALYSIS ON CODE REVIEW HOURS**

# In[ ]:


career_sat=data_dev.groupby('Country')['CodeRevHrs'].sum().sort_values(ascending=False).head().reset_index()
career_sat.plot.bar(x='Country',y='CodeRevHrs')
plt.xlabel('Country',fontsize=10)
plt.ylabel("Code Reviw Hours",fontsize=10)
plt.show()


# **ANALYSIS ON TECHNICAL TRENDS**

# In[ ]:


def generate_word_column(column,colour):
    
    words = ''
    for i in data_dev[column]:
        try:
            a=i.split(';')
            for j in a:
                words+=' '+ j
        except:
            a=-999

    word_cloud = WordCloud(background_color=colour,max_font_size=300,width=2000, height=1080).generate(words)
    plt.figure(figsize=(20,8))
    plt.imshow(word_cloud)


# In[ ]:


generate_word_column('LanguageWorkedWith','Black')


# In[ ]:


generate_word_column('LanguageDesireNextYear','Pink')


# In[ ]:


generate_word_column('DatabaseWorkedWith','Black')


# In[ ]:


generate_word_column('DatabaseDesireNextYear','Pink')


# In[ ]:


generate_word_column('PlatformWorkedWith','Black')


# In[ ]:


generate_word_column('PlatformDesireNextYear','Pink')


# In[ ]:


generate_word_column('WebFrameWorkedWith','Black')


# In[ ]:


generate_word_column('WebFrameDesireNextYear','Pink')


# In[ ]:


generate_word_column('MiscTechWorkedWith','Black')


# In[ ]:


generate_word_column('MiscTechDesireNextYear','Pink')


# In[ ]:


generate_word_column('DevEnviron','White')


# **ANALYSIS ON OPERATING SYSTEM**

# In[ ]:


generate_word_column('OpSys','White')


# In[ ]:


data_dev['OpSys'].value_counts()


# **ANALYSIS ON BLOCKCHAIN ORGANIZATION**

# In[ ]:


data_dev['BlockchainOrg'].value_counts()


# **ANALYSIS ON AGE**

# In[ ]:


age=data_dev['Age'].dropna()
plt.figure(figsize=(10,5))
sns.distplot(age)

