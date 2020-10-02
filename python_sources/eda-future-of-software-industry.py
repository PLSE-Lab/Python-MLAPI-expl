#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


# In[ ]:


data=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")
data.head()


# In[ ]:


df=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv")
df.head()


# In[ ]:


data.columns   #Seeing the available Features.


# In[ ]:


data.info()   #Getting an info about the columns to see instances of missing data.


# In[ ]:


#Removing the features that we are not going to use for our analysis.
data.drop(['Extraversion', 'ScreenName', 'SOVisit1st',
       'SOVisitFreq', 'SOVisitTo', 'SOFindAnswer', 'SOTimeSaved',
       'SOHowMuchTime', 'SOAccount', 'SOPartFreq', 'SOJobs', 'EntTeams',
       'SOComm', 'WelcomeChange', 'SONewContent','CodeRevHrs', 'UnitTests', 'PurchaseHow', 'PurchaseWhat'
       , 'DatabaseWorkedWith',
       'DatabaseDesireNextYear', 'PlatformWorkedWith',
       'PlatformDesireNextYear', 'WebFrameWorkedWith',
       'WebFrameDesireNextYear', 'MiscTechWorkedWith',
       'MiscTechDesireNextYear', 'DevEnviron', 'OpSys', 'Containers',
       'BlockchainOrg', 'BlockchainIs', 'BetterLife', 'ITperson', 'OffOn','CurrencyDesc',
       'CompTotal', 'CompFreq', 'ConvertedComp', 'WorkPlan',
       'WorkChallenge', 'WorkRemote', 'WorkLoc', 'ImpSyn', 'CodeRev', 'UndergradMajor',
       'EduOther', 'OrgSize', 'DevType', 'Age1stCode',
       'YearsCodePro', 'CareerSat', 'JobSat', 'MgrIdiot', 'MgrMoney',
       'MgrWant', 'JobSeek', 'LastHireDate', 'LastInt', 'FizzBuzz',
       'JobFactors', 'ResumeUpdate'],axis=1,inplace=True)


#  **Exploratory Data Analysis**

# In[ ]:


data.columns


# In[ ]:


data['Employment'].value_counts()


# In[ ]:


#Pie Chart for the different data of employment records(Freq. depends on type of employment)
data['Employment'].value_counts().plot(kind='pie',autopct="%.02f")


# In[ ]:


data['Gender'].value_counts().head(5)  #Freq. Dist. Table for the different Gender types


# In[ ]:


#Pie chart plot for the Different Genders hyst to give ys an idea of the male dominated nature ofthe data.
data['Gender'].value_counts().plot(kind='pie',autopct="%.02f")


# **Normally, we can analyze the relation between Gender/Programming Language and Gender with out data but our analysis would be inorrect due to the data being male dominant with 91% men and 9% others, so we are going to skip that.**

# In[ ]:


#Freq. Dist. Table of the Languages the developers have worked with(TOP 10 MOST POPULAR)
data['LanguageWorkedWith'].value_counts().head(10)


# In[ ]:


#Pie chart to display the freq. cap of all the TOP 10 most famous languages / lang. combninations
#preffered by the coders of today.
data['LanguageWorkedWith'].value_counts().head(10).plot(kind='pie',autopct="%.02f")


# In[ ]:


#language desired by these people to work with next year
data['LanguageDesireNextYear'].value_counts().head(10)


# In[ ]:


#Pie plot of language desired next year
data['LanguageDesireNextYear'].value_counts().head(10).plot(kind='pie',autopct='%.02f')


# **It is quite significant from these no's that Python is the hot favorite by these developers and has the highest no. of votes for their choice to use and learn next year**

# In[ ]:


#The most preffered medium of social media for these developers.
data['SocialMedia'].value_counts().head(10).plot(kind='bar')
#We get a better understanding of the people we are dealing with from this graph. Developers top 3 preffered Social Media are
#Reddit(dubbed the smart peoples social media) , Youtube(For getting info,learning as well as entertainment), & 
#Whatsapp(For both personal and proffesional Communication). NOTE: Theres also a good chunk of these people who dont use any social media.


# In[ ]:


#Freq Dist. Table for the ease of the survey
data['SurveyEase'].value_counts()


# In[ ]:


#We Can conclude that the survey was considered easy by majority of the survey takers.
data['SurveyEase'].value_counts().plot(kind='pie',autopct="%.02f")


# In[ ]:


#How many years these developers who took the survey have been coding.
data['YearsCode'].value_counts().head(10).plot(kind='bar')


# In[ ]:


#Education Level of the Developers
data['EdLevel'].value_counts().plot(kind='bar')


# **This is a strikingly important feature, previously it was thought proper training from institutions was a necessity to be a good programmer but the trend is shifting at a rapid pace. We notice that a large portion, of our developers have not gone to college or completed any higher degree. Also there are many dropouts among us who are currently developers. Therefore programming can be self taught with effort and dedication.**

# In[ ]:


#Frequency Distribution of the currencies used by our developers.
data['CurrencySymbol'].value_counts().head(10)


# In[ ]:





# In[ ]:


data['CurrencySymbol'].value_counts().head(10).plot(kind='pie',autopct="%.02f")
#Seeing the % of the currencies we get a fair idea about the countries where StackOverflow is widely used, not everyone takes the survey.
#However we can get a rough idea that in USA , Europe and India and UK . These countries host the most no. of the devs who took the survey.


# **Conclusion:
# We can safely conclude that the Stack Overflow Survey data is a good dataset with lots of features to work on but knowingly we choose these features to keep our analysis simple and compact. We got a fair idea about how the survey ease is fine and its okay to keep the same survey in the future with some tweaks. About how Python is in growing Demand and the hottest language of 2020. About how the no. of developers who havent completed higher education are ever increasing (BOTH FOR HOBBY AND PROFFESIONAL AREAS). About how many of our surveyers are Web Developers and about how Geographical Location affects the likelyhood of finding more developers among many other things. I'll write a proper Medium Article later on regarding my Analysis and link it. Hope you liked my analysis. Thanks a lot.**
