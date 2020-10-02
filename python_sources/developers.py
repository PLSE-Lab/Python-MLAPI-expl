#!/usr/bin/env python
# coding: utf-8

# # The Future of Software industry: A look into the current trends

# Softwares are developing at a rapid rate and to cope up with this enormous update demand the software developers also need to develop. And to know about the future of this software we must know the conditions of today's developers. This dataset contains data in various fields for a developer. Let us see what we can extract from this dataset. 

# This is a vast dataset and we will work only on the data of developers for now. To start with I will import all the required libraries that I need to work on this dataset.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS
import collections as cl
from datetime import datetime
import os


# ### The above Librarys are used as because.

# - numpy and pandas to work with the Dataset.
# - seaborn for Visualization of the data.
# - tqdm to track the progression of the loops.
# - matplotlib to plot graphs.
# - wordcloud to generate Wordclouds.
# - DateTime to Print today's Date
# - Os to work and save our outputs.

# Throughout this Code, I will try to find out the answer to the following questions.

# ### Questions!

# - How many Developers are writing code for opensource?
# - How is the number of developers Distributed over the globe?
# - How much money do the Developers get throughout the globe?
# - How much are the Developers satisfied with their jobs throughout the globe?
# - What id the relation between the age of a developer and his work?
# - What are the social media sites that the developer uses?
# - What are the OS platform that the developers use to code and what OS he wants to use in Future?
# - What are the Database Servers that the Developers use to While coding and what Database Servers he wants to use in Future?
# - What are the WebFrames that the Developers uses and what WebFrame he wants to use in Future?
# - Which type of work location is preferred by the Developers?

# In[ ]:


try:
    data = pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')
except:
    data = pd.read_csv('C:/Users/Shakib/CampusX-Files/survey_results_public.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# #### For this Analysis processes, we will be needing a few of the columns of this dataset, Hense we will extract only the required data.

# In[ ]:


data.drop(columns={'Respondent','Hobbyist','Employment','Student','EdLevel','UndergradMajor','EduOther','OrgSize','DevType','YearsCode','Age1stCode','YearsCodePro','JobSat','MgrIdiot','MgrMoney','MgrWant','JobSeek','LastHireDate','LastInt','FizzBuzz','JobFactors','ResumeUpdate','CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','WorkPlan','WorkChallenge','WorkRemote','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','MiscTechWorkedWith','MiscTechDesireNextYear','DevEnviron','OpSys','Containers','BlockchainOrg','BlockchainIs','BetterLife','ITperson','OffOn','Extraversion','ScreenName','SOVisit1st','SOVisitFreq','SOVisitTo','SOFindAnswer','SOTimeSaved','SOHowMuchTime','SOAccount','SOPartFreq','SOJobs','EntTeams','SOComm','WelcomeChange','OpenSource','SONewContent','Trans','Sexuality','Ethnicity','Dependents','SurveyLength','SurveyEase'},inplace = True)


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
# - CareerSat to category type as this contain only 5 type of string.
# - WorkLoc to Category type as 'Ofice' or 'Home'.
# - Age to Integer type as age is alwees a whole number.
# - Gender to Category type as 'Male' or 'Female'

# Quantity of incompleteness issue(almost every column contains empty data so we will try to fill them)
# - MainBranch Will remove all the empty data as Non-Developer.
# - Country We don't have other option as to remove the entry row without Country column.
# - CareerSat We will take a meddle value for empty entry as all satisfied will comment on their satisfaction.
# - WorkWeekHrs to not disturb the observation we will replace the empty entry with mean of that column.
# - Gender we will take the empty entry as Transgender as they sometimes do not state their Gender.

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


# #### Next we will convert CareerSat to integer type as rating out of 5.And for empty value CareerSat we will take a meddle value (3).

# In[ ]:


data['CareerSat'].value_counts()


# In[ ]:


data['CareerSat'] = data['CareerSat'].astype('category')


# #### Next we will convert WorkLoc to Category type as 'Ofice' or 'Home' and we will consider Home as an empty entry as i fill like all the Office people comment their interest.

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


# #### Next we will convet Age to Integer type as age is alwees a whole number and not to disturb the observation.

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


# #### Now it is time to convert Gender to Category type as 'Male' or 'Female' And we will take the empty entry as Transgender as they sometimes do not state their Gender.

# In[ ]:


data['Gender'].value_counts()


# In[ ]:


Gender = []
for i in tqdm(data['Gender']):
    if (i=='Man') or (i== 'Man;Non-binary, genderqueer, or gender non-conforming'):
        Gender.append('Male')
    elif (i=='Woman') or (i=='Woman;Non-binary, genderqueer, or gender non-conforming') or (i=='Woman;Man;Non-binary, genderqueer, or gender non-conforming'):
        Gender.append('Female')
    else:
        Gender.append('Transgender')
data['Gender'] = Gender
data['Gender'] = data['Gender'].astype('category')


# #### Country We don't have other option as to remove the entry row without Country column. Although it is a very little portion of the dataset.

# In[ ]:


Country = data['Country'].value_counts().index
filters = []
for i in tqdm(data['Country']):
    filters.append(i in Country)
    
data = data[filters]


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


try:
    os.makedirs("Output_Graphs")
except FileExistsError:
    print("Folder alrady present")


# ### 1.How many Developers are writing code for opensource?

# #### To answer this question i will plot a graph for all types of coder vs developer or not.

# In[ ]:


haha = data.groupby('MainBranch')['OpenSourcer'].value_counts()
haha = haha.to_frame('Number_of_Developers')
haha = haha.reset_index()
sns.barplot(x='MainBranch',y='Number_of_Developers',hue='OpenSourcer',data=haha).get_figure().savefig('Output_Graphs/Developers_vs_Opensourcer.jpg',dpi=1200,bbox_inches = 'tight')


# ### 2. How is the number of developers Distributed over the globe?

# #### And to answer this Question I will plot a graph country wist with their number of Developers.

# In[ ]:


data_dev = data[data['MainBranch']=='Developer']
data = data_dev
data_plot = data['Country'].value_counts()[:25].reset_index().rename(columns={"index":"Country Name","Country":"Number of Developers"})
sns.barplot(y='Country Name',x='Number of Developers',data = data_plot).get_figure().savefig('Output_Graphs/Country_vs_number_of_developers.jpg',dpi=1200,bbox_inches = 'tight')


# ### 3.How much money do the Developers get throughout the globe?

# #### To answer this question i will plot a bar graph between the name of the country vs average salary they get in USD

# In[ ]:


num = data['Country'].value_counts()[:30]
total = data.groupby('Country')['ConvertedComp'].sum()
data_plot1 = (total/num).sort_values(ascending=False)[:30]
data_plot = data_plot1.reset_index()
data_plot.rename(columns={0:'Average_income_in_USD','index':'Country_Name'},inplace=True)
sns.barplot(y='Country_Name',x='Average_income_in_USD',data = data_plot).get_figure().savefig('Output_Graphs/Salary_of_the_developers.jpg',dpi=1200,bbox_inches = 'tight')


# In[ ]:





# ### 4. How much are the Developers satisfied with their jobs throughout the globe?

# #### To answer this question we will plot a bar-graph between career satisfaction and number of vote in that section.

# In[ ]:


plot_data=data_dev['CareerSat'].value_counts().sort_index().reset_index()
plot_data.rename(columns={'index':'','CareerSat':'Number of Developers'},inplace=True)
sns.barplot(y='',x='Number of Developers',data=plot_data).get_figure().savefig('Output_Graphs/job_satis.jpg',dpi=1200,bbox_inches = 'tight')


# In[ ]:





# ### 5. What id the relation between the age of a developer and his work?

# #### To answer this question i will plot a line graph between age and it's frequency.

# In[ ]:


plot_data = data_dev['Age'].value_counts().sort_index().reset_index()
sns.lineplot(x='index',y='Age',data=plot_data).get_figure().savefig('Output_Graphs/Age_of_developers.jpg',dpi=1200,bbox_inches = 'tight')


# In[ ]:





# ### 6.What are the social media sites that the developer uses?

# #### To answer this question i will plot a bargraph i will plot a between name of social media and their frequency.

# In[ ]:


data_plot = data_dev['SocialMedia'].value_counts().reset_index()
data_plot.rename(columns={'index':'Name of SocialMedia','SocialMedia':'Number of Users'},inplace=True)
sns.barplot(x='Number of Users',y='Name of SocialMedia',data=data_plot).get_figure().savefig('Output_Graphs/Social.jpg',dpi=1200,bbox_inches = 'tight')


# In[ ]:





# ###  7.What are the OS platform that the developers use to code and what OS he wants to use in Future.

# ### 8.What are the Database Servers that the Developers use to While coding and what Database Servers he wants to use in Future?

# ### 9.What are the WebFrames that the Developers uses and what WebFrame he wants to use in Future?

# #### To answer the above 3 question i will draw some wordcloud ans save them.

# In[ ]:


def generate_word_column_for_the_column_of(column,color):
    column_name = column
    os_now_all_word = ''
    for i in data_dev[column_name]:
        try:
            a=i.split(';')
            for j in a:
                os_now_all_word+=' '+ j
        except:
            a=5

    cloud = WordCloud(background_color=color,max_font_size=250,width=960, height=1080).generate(os_now_all_word)

    cloud.to_file("Output_Graphs/" + column_name + '.png')
    plt.imshow(cloud)


# In[ ]:


generate_word_column_for_the_column_of('LanguageWorkedWith','white')


# In[ ]:


generate_word_column_for_the_column_of('LanguageDesireNextYear','black')


# In[ ]:


generate_word_column_for_the_column_of('DatabaseWorkedWith','white')


# In[ ]:


generate_word_column_for_the_column_of('DatabaseDesireNextYear','black')


# In[ ]:


generate_word_column_for_the_column_of('PlatformWorkedWith','white')


# In[ ]:


generate_word_column_for_the_column_of('PlatformDesireNextYear','black')


# In[ ]:


generate_word_column_for_the_column_of('WebFrameWorkedWith','white')


# In[ ]:


generate_word_column_for_the_column_of('WebFrameDesireNextYear','black')


# In[ ]:





# ### 10.Which type of work location is preferred by the Developers?

# #### To answer this question i will plot a pie chat of frequency of office and home candidates.

# In[ ]:


data_plot_number = data_dev['WorkLoc'].value_counts().values
data_plot_name = data_dev['WorkLoc'].value_counts().index 
s = plt.pie(data_plot_number,labels=data_plot_name,autopct='%1.1f%%',)
plt.savefig('Output_Graphs/Location_to_work.jpg',dpi=1200,bbox_inches = 'tight')


# In[ ]:





# In[ ]:





# In[ ]:


print(datetime.now())


# # by MD SHAKIB MONDAL
# # email : - sakibmondal7@gmail.com
