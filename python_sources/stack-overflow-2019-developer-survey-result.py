#!/usr/bin/env python
# coding: utf-8

# # The Future of Software Industry

# You would think that we might have developed enough software by now in 2019. It appears not. There is still massive growth in the software industry and tech analysts are spending even more time trying to fashion their predictions for and of the road ahead. So in what direction is the software industry heading, lets have a look through the **Stack Overflow 2019 Developer Survey Results**.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
import collections as cl
from datetime import datetime


# numpy and pandas to work with the Dataset.<br>
# seaborn for Visualization of the data.<br>

# Throughout this Code, I will try to find out the answer to the following questions.

# ### Questions!

# - How many Developers are writing code for opensource?
# - How much money do the Developers get throughout the globe?
# - How much are the Developers satisfied with their jobs throughout the globe?
# - What is the relation between the age of a developer and his work?
# - What are the social media sites that the developers use?
# - Which is the most popular Operating System among the Developers?
# - Why do most of the Developers update their RESUME frequently?
# - What are the platforms that the Developers had used extensively and what platforms do they want to use in future?
# - What are the programming languages that the Developers had worked with and what languages do they want to use in future?
# - What are the Database Servers that the Developers use to while coding and what Database Servers they want to use in future?
# - What are the WebFrames that the Developers use and what WebFrame they want to use in future?
# - Which type of work location is preferred by the Developers?

# In[ ]:


data = pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


desc = pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')
desc.values


# #### For this Analysis, we need a few columns of this dataset where we will extract only the required data.

# In[ ]:


data.drop(columns={'Respondent','Hobbyist','Employment','Student','EdLevel','UndergradMajor','EduOther','OrgSize','DevType','YearsCode','Age1stCode','YearsCodePro','JobSat','MgrIdiot','MgrMoney','MgrWant','JobSeek','LastHireDate','LastInt','FizzBuzz','JobFactors','CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','WorkPlan','WorkChallenge','WorkRemote','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','MiscTechWorkedWith','MiscTechDesireNextYear','DevEnviron','Containers','BlockchainOrg','BlockchainIs','BetterLife','ITperson','OffOn','Extraversion','ScreenName','SOVisit1st','SOVisitFreq','SOVisitTo','SOFindAnswer','SOTimeSaved','SOHowMuchTime','SOAccount','SOPartFreq','SOJobs','EntTeams','SOComm','WelcomeChange','OpenSource','SONewContent','Trans','Sexuality','Ethnicity','Dependents','SurveyLength','SurveyEase'},inplace = True)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe()


# With an overall overview of the data, it is clear that there are various incompleteness with the data and they are as follows.

# ### Assessing The Data

# Type Convertion
# - MainBranch to Category type
# - OpenSourcer to Category type as 'Yes' or 'No'.
# - ResumeUpdate to Category type.
# - CareerSat to category type as this contain only 5 type of string.
# - WorkLoc to Category type as 'Office' or 'Home'.
# - OpSys to Category type.
# - Age to Integer type as age is always a whole number.
# - Gender to Category type as 'Male' or 'Female'.

# Quantity of incompleteness which needs to be sorted else it will hamper the analysis
# - MainBranch Will remove all the empty data as Non-Developer.
# - Country We don't have other option as to remove the entire row without Country column.
# - WorkWeekHrs We will replace the empty entry with mean of that column.
# - Gender We will take the empty entry as Transgender as they sometimes do not state their Gender.

# ### Cleaning The Data.

# So now we know the datatypes of the columns and hence now we will be converting them to their required data tyes. 

# #### Let's Start with MainBranch to Category type and from MainBranch we will remove all the empty data as Student.

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
        MainBranch.append('Semi_Developer')
    elif i == 'I code primarily as a hobby':
        MainBranch.append('Hobby') 
    elif i == 'I used to be a developer by profession, but no longer am':
        MainBranch.append('Ex_Developer')
    else:
        MainBranch.append('Student')
data['MainBranch'] = MainBranch
data['MainBranch'] = data['MainBranch'].astype('category',inplace=True)


# #### Converting ResumeUpdate to Category type.

# In[ ]:


data['ResumeUpdate'].value_counts()


# In[ ]:


data['ResumeUpdate'] = data['ResumeUpdate'].astype('category')


# #### Next is OpenSourcer to Category type as 'Yes' or 'No'.

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


# #### Next we will convert CareerSat to category type.

# In[ ]:


data['CareerSat'].value_counts()


# In[ ]:


data['CareerSat'] = data['CareerSat'].astype('category')


# #### Next we will convert WorkLoc to Category type.

# In[ ]:


data['WorkLoc'].value_counts()


# In[ ]:


WorkLoc = []
for i in data['WorkLoc']:
    if i == 'Office':
        WorkLoc.append(i)
    elif i == 'Home':
        WorkLoc.append('Home')
    else:
        WorkLoc.append('Other')
data['WorkLoc'] = WorkLoc
data['WorkLoc'] = data['WorkLoc'].astype('category')


# #### Now we will convert OpSys to Category type.

# In[ ]:


data['OpSys'].value_counts()


# In[ ]:


data['OpSys'] = data['OpSys'].astype('category')


# #### Next we will convert Age to Integer type as age is always a whole number.

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


# #### Now it is time to convert Gender to Category type as 'Male' or 'Female' and we will take the empty entry as 'Transgender' as they sometimes do not prefer to state their Gender.

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


# #### For Country, We don't have other option as to remove the entire row without Country column.

# In[ ]:


Country = data['Country'].value_counts().index
filters = []
for i in data['Country']:
    filters.append(i in Country)
    
data = data[filters]


# In[ ]:


data.info()


# In[ ]:


data.shape


# ### 1. How many Developers are writing code for opensource?

# In[ ]:


dev = data.groupby('MainBranch')['OpenSourcer'].value_counts()
dev = dev.to_frame('Number_of_Developers')
dev = dev.reset_index()
sns.barplot(x='MainBranch',y='Number_of_Developers',hue='OpenSourcer',data=dev)


# ### 2. How much money do the Developers get throughout the globe?

# In[ ]:


num = data['Country'].value_counts()[:30]
total = data.groupby('Country')['ConvertedComp'].sum()
data_plot1 = (total/num).sort_values(ascending=False)[:30]
data_plot = data_plot1.reset_index()
data_plot.rename(columns={0:'Average income in USD','index':'Country Name'},inplace=True)
sns.barplot(y='Country Name',x='Average income in USD',data = data_plot)


# ### 3. How much are the Developers satisfied with their jobs throughout the globe?

# In[ ]:


data_dev = data[data['MainBranch']=='Developer']
plot_data=data_dev['CareerSat'].value_counts().reset_index()
plot_data.rename(columns={'index':'Amount of Satisfaction','CareerSat':'Number of Developers'},inplace=True)
sns.barplot(y='Amount of Satisfaction',x='Number of Developers',data=plot_data)


# ### 4. What id the relation between the age of a developer and his work?

# In[ ]:


plot_data = data_dev['Age'].value_counts().reset_index()
plot_data.rename(columns={'index':'Age','Age':'Number of Developers'},inplace=True)
sns.lineplot(x='Age', y='Number of Developers', data=plot_data)


# ### 5. What are the social media sites that the developer uses?

# In[ ]:


data_plot = data_dev['SocialMedia'].value_counts().reset_index()
data_plot.rename(columns={'index':'Name of SocialMedia','SocialMedia':'Number of Users'},inplace=True)
sns.barplot(x='Number of Users',y='Name of SocialMedia',data=data_plot)


# ### 6. Which is the most popular Operating System among the Developers?

# In[ ]:


data_plot = data_dev['OpSys'].value_counts().reset_index()
data_plot.rename(columns={'index':'Name of Operating System','OpSys':'Number of Users'},inplace=True)
sns.barplot(x='Number of Users',y='Name of Operating System',data=data_plot)


# ### 7. Why do most of the Developers update their RESUME?

# In[ ]:


data_plot = data_dev['ResumeUpdate'].value_counts().reset_index()
data_plot.rename(columns={'index':'Reason for updating RESUME','ResumeUpdate':'Number of Developers'},inplace=True)
sns.barplot(x='Number of Developers',y='Reason for updating RESUME',data=data_plot)


# In[ ]:


def generate_word_column_for_the_column_of(column):
    column_name = column
    os_now_all_word = ''
    for i in data_dev[column_name]:
        try:
            a=i.split(';')
            for j in a:
                os_now_all_word+=' '+ j
        except:
            a=5

    cloud = WordCloud(background_color="white",max_font_size=250,width=960, height=1080).generate(os_now_all_word)

    cloud.to_file(column_name + '.png')
    plt.imshow(cloud)


# ###  8. What are the platforms that the developers had used extensively and what platforms do they want to use in Future?

# In[ ]:


generate_word_column_for_the_column_of('PlatformWorkedWith')


# In[ ]:


generate_word_column_for_the_column_of('PlatformDesireNextYear')


# ### 9. What are the programming languages that the developers had worked with and what languages do they want to use in Future?

# In[ ]:


generate_word_column_for_the_column_of('LanguageWorkedWith')


# In[ ]:


generate_word_column_for_the_column_of('LanguageDesireNextYear')


# ### 10. What are the Database Servers that the Developers use to while coding and what Database Servers do they want to use in Future?

# In[ ]:


generate_word_column_for_the_column_of('DatabaseWorkedWith')


# In[ ]:


generate_word_column_for_the_column_of('DatabaseDesireNextYear')


# ### 11. What are the WebFrames that the Developers use and what WebFrame do they want to use in Future?

# In[ ]:


generate_word_column_for_the_column_of('WebFrameWorkedWith')


# In[ ]:


generate_word_column_for_the_column_of('WebFrameDesireNextYear')


# ### 12. Which type of work location is preferred by the Developers?

# In[ ]:


data_plot_number = data_dev['WorkLoc'].value_counts().values
data_plot_name = data_dev['WorkLoc'].value_counts().index 
plt.pie(data_plot_number,labels=data_plot_name,autopct='%1.1f%%',)


# ### This concludes my analysis, I have tried to answer all the mentioned questions and I think all those questions are more than enough for analysis in this dataset. Thank you everyone for going through the complete analysis.

# In[ ]:




