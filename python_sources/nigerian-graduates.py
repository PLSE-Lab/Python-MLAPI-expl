#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # explaining the datasets
# 
# The datasets was culled by Stutern. And the summary of the data origin is as follows
# A total of 5,219 Nigerian graduates completed the survey.
# These graduates completed their degree within the last 5 years (2013 - 2017).
# The survey was live from February 8 through May 15, 2018.
# The survey was hosted using Google Forms and Stutern recruited respondents via email and social media sites.
# To account for graduates in marginalized locations, tracking officers from BudgITCo conducted the offline version in 5 states (Edo, Enugu, Ibadan, Imo and Kaduna State).
# We removed about 600 responses that were incomplete from the offline version of the survey before we arrived at 5,219 total responses.
# Not every question was shown to every respondent, as some questions were specifically for those who are employed or other cases as it may be.

# # Inspirations
# 
# 1. Try to identify patterns from the data
# 2. Predict the employability of a nigerian graduate
# 3. Identify the most important factors that contribute to the employability of the Nigerian Graduate
# 4. compare the employability of the Nigerian graduate from Public Universities to Private Universities
#     * This might increase as time goes on*

# In[ ]:


#for data processing
import pandas as pd
import numpy as np

#for visualizing
from matplotlib import pyplot as plt
import seaborn as sns
#make use of sns beautiful plots
sns.set()


#for MachineLearning
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# # Importing Data

# In[ ]:


file_path= ('/kaggle/input/nigerian-gradaute-report-2018/Nigerian Graduates Destination Survey (2013 -2017) (Responses) - nigerian graduates survey 2018.csv')
data_original = pd.read_csv(file_path)
#make a copy of data you can always come back to incase of mistakes
data = data_original


# In[ ]:


#Lets check the statistical summaries of both the categorical and numerical data
data.head()
data.describe(include = 'all')


# #lets examine the rows individualy
# 1. The names of the columns(features) are too long and needs to reduced to a shorter yet meaningful size
# 2. we can see that about 9 columns have 5219 cells which means they have no null values
# 3. Timestamp has a unique value of 5205, meaning we will have to drop it for the first analysis for easier analysis.
# 4. some data that should have only two answers, have more than two, meaning that there are some answers that are not valid in the category
# 5. also, to avoid data leakage, we will have to exclude some replies that have to do with work experience as it will affect the analysis
# 6. those are the noticeable insights for now, did you see any more?

# In[ ]:


#renaming columns
old_columns = list(data.columns.values)
#in order not to mess up the workspace, you display present column names, then rename them by slicing the list

new_columns=['Time','Gender','Grad_Y','Course','School','Highest_Q','Current_Stat','No_Of_Jobs','NYSC_cert','NYSC_Year',
             'Through_NYSC','FJob_Level','FJob_Role','FJob_sector','FJ_Income_Level','FJ_Required_HQ','Reason_FJ',
             'PJ_level', 'PJ_Role','PJ_Sector','PJ_Income','PJ_Required_HQ','Reason_PJ',
            'Best_Employer','Reason_Best_Employer','Most_PS','Currency','Job_Hours','Most_Important_Qualification',
             'Findout_Job','Worked_For_Employer',
            'Transport_TW','Rent_Buy','CP_Job','CP_Further_Studies','Skills_Prepared']
columns_dict = dict(zip(old_columns,new_columns))
new_data = data.rename(columns=columns_dict)


# 

# In[ ]:


#we call data.head() to confirm that our new_columns has been renamed accordingly
new_data.head()


# In[ ]:


#The columns has been renamed to shorter neater forms, but just in case, lets create a dataframe that can be used as dictionary
#in case we forgot what the columns means
dictionary = pd.DataFrame({'Column Names':[x for x in new_columns],'Meaning':[y for y in old_columns]})


# In[ ]:


#now before we move forward, lets examine each columns and see what explorations we can do.
#lets start with the gender, to examine the balace in the gender of participant
new_data.Gender.unique()


# '''we can see that there are three entries here, which is surprising, but apparently, some of the respondents chose a third option
# prefer not to say.'''

# In[ ]:


#to check if they are moderately/evenly distributed
new_data.Gender.value_counts()


# we discover that there are 2624 male respondents, 2592 female respondents and 3 people who prefered not to say. therefore gender is balanced as the male is 50.22 about percent and female 49.66 percent. Also we can afford to drop the other observations to make the data balanced
# 

# uncomment the bulk of code below to get the prcentage of each Gender class before we dropped 'Prefer not to say'

# In[ ]:


def get_percent_age(new_data):    
    l = list(new_data.Gender.value_counts())
    percent_male = l[0]*100/sum(l)
    percent_female = l[1]*100/sum(l)
    others = l[2]*100/sum(l)
    return percent_male,percent_female,others
#percentages = get_percent_age(new_data)
#print(percentages)


# In[ ]:


#dropping the 'Prefered Not to say'
index = new_data[new_data['Gender']=='Prefer not to say'].index
new_data.drop(index, axis = 0, inplace = True)


# In[ ]:


#checking
new_data.Gender.unique()


# shows that we now have two categories of gender

# # Pre EDA corrections/processing

# In[ ]:


#for Graduation year, lets check if there are any problems
new_data.Grad_Y.unique()


# from the result, we see that the questionnair was for only people who graduated between 2013 and 2017,
# we check for the next feature

# In[ ]:


new_data.Course.value_counts()


# we see that there are 127 unique courses with computer science, Accountancy and Economics being the highest, but that is 
# a pretty high number, we can examine the data closely to see if they are truly unique
# 

# In[ ]:


print(set(new_data.Course[200:500]))


# from here, we can decide to go with courses with more than 10 frequency

# In[ ]:


def pick_top(data,interest):
    courses_counts = data[interest].value_counts() 
    low_courses=[]
    low_courses_index=[]
    for course, count in courses_counts.items():
        if count<10:
            low_courses.append(course)
    for i in low_courses:
        index = data[data[interest]==i].index
        for i in index:
            low_courses_index.append(i)
    return low_courses_index


# In[ ]:


dropped_courses_index= pick_top(new_data, 'Course')
major_courses = new_data.drop(dropped_courses_index, axis=0)
data_copy = major_courses


# # for the schools, we have

# In[ ]:


data_copy.School.value_counts()


# we can see that most of the respondents were from university of lagos, we can take schools with atleast 10 responses to further get a good idea of the data

# In[ ]:


dropped_schools_index = pick_top(data_copy, 'School')
major_schools = data_copy.drop(dropped_schools_index, axis=0)


# Side Note:It will be insightful as we can now see the schools that has the highest employable graduates

# for the Highest Qualifications
# 

# In[ ]:


major_schools.Highest_Q.value_counts()


# we see that 4,145 of the respondents have a bachelor degree, 285 has Master's degree and so on, it will be insightful to see
# Side Note: if higher degrees means higher employability

# current status, which is a very important feature to us,

# In[ ]:


major_schools.Current_Stat.value_counts()


# we have various categories, where working full time is major,1267 part of, people still serving are 790, nd self employed/enterpreneurs are 790
# Side Note:we can derive insight from which schools breed enterpreneurs more, which schools produced the least employed and sort of

# In[ ]:


#for currency
major_schools.Currency.value_counts()


# interestingly, we saw that some people are paid in foreign currency, it will be interesting to know which companies they work for if they are self employed or otherwise
# 

# for job hours

# In[ ]:


major_schools.Job_Hours.isnull().sum()


# we can see that there are about 1635 null values, probably people that are currently unemployed, which we can replace with 0
# 

# In[ ]:


major_schools['Job_Hours'] = major_schools['Job_Hours'].fillna(0)


# next we check what was most important to current employers about as far as  qualification is concerned

# In[ ]:


major_schools.Most_Important_Qualification.value_counts()


# as usual, we can summarize others by getting only ones with more than 10 answers

# In[ ]:


low_qualifications = pick_top(major_schools,'Most_Important_Qualification')
Most_qualifications = major_schools.drop(low_qualifications, axis = 0)


# next we check the medium people found out about their job

# In[ ]:


Most_qualifications.Findout_Job.value_counts()


# we see that most people that are employed heard from their close contacts, family and friends, followed by social media and professional networking sites

# next we check if their courses prepared them for further studies, meaning we can have courses that are most likely to further their studies and therefore also recognise if going for a further study guarantees having a job by checking people with higher than first degree in a course and the employability rate.

# In[ ]:


Most_qualifications.CP_Further_Studies.value_counts()


# from the result we see that most of the answers tends towards the path of agreement

# # EDA and Answering Questions

# We have a lot of questions to be answered by first we have to segment the data to make useful insights. first
# #1. Which schools provide the most employable graduates. Here we can do this on two basis:
#  i. On school to school basis      ii. On privately owned vs Goverment/state owned basis   iii. University vs Polytechnic

# In[ ]:


#for option i, we have to do the following:
#1. you pick the ones that are done with NYSC,as only those could be working, so we drop nysc option, using  the major schools
drop_nysc = major_schools[major_schools.Current_Stat !='Youth Corper (NYSC)']
#so now we have a dataframe containig only people that have completed their NYSC


# In[ ]:


#2. generalise the group into two, employed and unemployed
#next we seperate them into employed and unemployed, we will consider only employed, self employed as well as umemployed
data1=drop_nysc[drop_nysc.Current_Stat == 'Working full time (paid employment)']
data2=drop_nysc[drop_nysc.Current_Stat == 'Unemployed']
data3=drop_nysc[drop_nysc.Current_Stat == 'Self-employed/freelance/entrepreneur']
frames = [data1,data2,data3]
employed_by_school = pd.concat(frames)
employed_by_school


# #Now that we have the dataframe containing people of interest, we can now create a dataframe to help summarize the data we have first we get the names of the schools
# This function returns a dataframe with the names of schools, and statistics of interest

# In[ ]:


def final_table(data):
    '''Returns the final data frame including students employability stats'''
    school_ = data.groupby('School')['Current_Stat'].value_counts()
    #which returns a key/values pairs, we can get the name of school by

    list_schools = list(school_.keys())
    #then we extract only the names of schools
    names_of_school = []
    for i in range(len(list_schools)):
        names_of_school.append(list_schools[i][0])
    name_of_school = set(names_of_school)
    
    #create dataframe for the name of schools
    name_of_schools_df = pd.DataFrame ({'Name_Of_School':[x for x in name_of_school]})
    #sort it alphabetically
    name_of_school_df = name_of_schools_df.sort_values(by='Name_Of_School', ascending=True)
    name_of_school_df = name_of_school_df.reset_index(drop=True)
    
    
    #Below is a list of the numbers for each school, it was computed manually, but there will research to see how it can be
    #automatically computed
    unemployed = [11,2,19,11,19,2,10,13,26,35,4,3,7,7,4,22,4,3,30,4,4,6,16,5,
                  36,8,6,12,2,27,13,18,6,6,15,6,1,2,5,1,6,41,13,22,8,10,10,10,
                  9,4,20,69,3,24,6,25,5,17,11,2,9,25,4,36,4,38,35,5,96,11,19,34,5,6,5,21]
    working=[8,4,19,8,12,4,12,9,3,37,2,10,3,3,2,33,4,5,154,1,4,1,4,6,24,6,3,12,2,29,18,24,
             0,6,15,3,4,4,4,1,2,24,12,31,6,9,5,6,9,3,21,101,6,21,11,43,17,11,6,7,7,26,1,50,
             6,60,53,5,135,3,43,21,4,1,0,10]
    self_employed=[5,3,0,5,10,1,10,1,8,14,3,3,
                   3,2,2,10,4,0,49,4,0,1,9,2,20,8,1,13,4,13,10,11,2,1,3,5,5,2,4,3,3,16,8,19,2,
                   2,0,2,3,0,6,33,2,7,3,15,1,5,5,9,7,16,4,15,3,20,17,4,62,4,64,15,6,2,2,6]
    
    
    
    #Add another column for unemployed, employed, self employed, sum of graduates, percentage unemployed, percentage employed
    Unemployed_df = pd.DataFrame({'Unemployed':[x for x in unemployed]})
    employed_df = pd.DataFrame({'Employed':[x for x in working]})
    self_employed_df = pd.DataFrame({'Self_Employed':[x for x in self_employed]})
   
    #create a list of the dataframes
    dataframes = [name_of_school_df,Unemployed_df,employed_df,self_employed_df]

    #return a concatenated dataframe
    Name_of_schools_df= pd.concat(dataframes,sort=False, axis = 1)
    return Name_of_schools_df


# In[ ]:


final_dataframe= final_table(employed_by_school)
final_dataframe


# this function adds Percentage, total to the dataframe

# In[ ]:


def calculate_summaries(data):
    '''Calculates the percentage as well as the sum total of features'''
    data['Total_Graduates']=data['Unemployed']+data['Employed']+data['Self_Employed']
    data['Percentage_Employed']=(data['Employed']*100)/(data['Unemployed']+data['Employed']+data['Self_Employed'])
    data['Percentage_Unemployed']=(data['Unemployed']*100)/(data['Unemployed']+data['Employed']+data['Self_Employed'])
    data['Percentage_Self_Employed']=(data['Self_Employed']*100)/(data['Unemployed']+data['Employed']+data['Self_Employed'])
    data['Total_Working_Percentage']=data['Percentage_Self_Employed']+data['Percentage_Employed']
    return data


# Running this line of code gives us a summary of everything

# In[ ]:


full_data_frame = calculate_summaries(final_dataframe)
full_data_frame


# we can sort the data by percentage of employability to get idea of the school, or/and by total graduates by checking if they atleast started something(working/self employed/enterpreneur)

# In[ ]:


full_data_frame.groupby(['Name_Of_School','Total_Graduates'])['Total_Working_Percentage'].agg(['max'])


# from the table, we can check for which schools has the most self_employability rate which other insight can you derive from the table?

# To get a much broader outloook, we can catgorize them based on maybe they are state/federal owned or maybe they are private we add a column, indicating maybe they are privately owned or government owned

# Get a copy of the data

# In[ ]:


copy_for_use = full_data_frame


# based on research, we gathered if a school was privately owned or otherwise, from there we have this

# In[ ]:


status = ['Government','Government','Government','Private','Government','Private','Government','Government','Government','Private',
          'Government','Private','Private','Government','Private','Private','Private','Private','Private','Private','Private',
          'Government','Government','Government','Government','Government','Government','Government','Government','Government',
          'Government','Government','Government','Private','Government','Government','Government','Government','Government','Government',
          'Government','Government','Government','Government','Private','Private','Government','Government','Government','Government',
          'Government','Government','Government','Government','Government','Government','Government','Private','Government','Government',
          'Government','Government','Government','Government','Government','Government','Government','Government','Government','Government',
           'Government','Government','Government','Government',
           'Private',
          'Government']


# we add this to a new column, 'Government_or_Private'

# In[ ]:


copy_for_use['Government_or_Private']=status
copy_for_use


# we can them group them by the status and see which schools perform better generally
# 

# In[ ]:


copy_for_use.groupby(['Government_or_Private','Total_Graduates'])['Total_Working_Percentage'].agg(['max'])


# from here we can see that even though they are lower, the precentage employabilty of private schools are mostly higher
# 
# more analysis and visualiztions will be carried out on this part

# I will be updating the notebook with more analysis everyday as part of my data science journey
# I will also appreciate input from anyone pointing out what could be done or what could have been done better
# Thank you for your time

# In[ ]:




