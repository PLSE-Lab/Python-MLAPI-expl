#!/usr/bin/env python
# coding: utf-8

#  # Multifactor General Knowledge Test analysis

# ## Project description

# ### Abstract

# The MGKT measures general knowledge on four scales grouped into two domains. It uses a unique multiple selection question format that allows high measurement precision in comparatively little time compared to multiple-choice questions. This work tries to analyse different trends in the test scores and to provide relevant data visualisation. 

# ### Motivation

# We want to explore trends behind database with general knowledge test with demographic data and to find out possible correlations between general performance on the test and demographic data. The answers can show some dependencies on educational standards in different countries and other possible correlations for different types of test-takers. 

# ### Dataset

# The dataset used for the study is answers to the Multifactor General Knowledge Test with 19218 records, including 32 questions (5 correct answers each, so equivalent to ~160 questions in total), gender, age, native language, country and some other performance information.
# 
# Source of the MGKT Dataset - https://openpsychometrics.org/_rawdata/

# ### Data Preparation and Cleaning

# Data was presented in CSV format with a lot of excessive details and not structured properly. We needed to remove useless for this analysis columns, compute average numbers per user (it was presented as 32x4 columns for each answer), transform the data frame into an appropriate format and prepare it for visual mapping on the map of the world with substituting country data. 

# ###  Research Questions

# * What countries perform relatively better on the test. 
# * Male or female demonstrate better relative performance with general knowledge. 
# * Is there any correlation with participants with native English and non-natives 
# * and their test scores. 
# * Is there any correlation with age of participants and test results. 

# ### Methods

# Data analysis methods focus on strategic approaches to taking raw data, mining for insights that are relevant to primary goals, and drilling down into this information to transform metrics, facts, and figures. Methods used: 
# 1. Preparing the database. 
# 2. Omitting irrelevant for this study information. 
# 3. Leveraging and adjusting the data into relative figures. 
# 4. Data visualization and presentations. 
# 5. Creating an interactive map of the world with live functionality. 

# ## Data preparation and wrangling

# ### Importing Dataset

# In[ ]:


import pandas as pd
import folium
import pycountry
import matplotlib.pyplot as plt



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/mgkt-dataset/data.csv', sep=',')
print(data.columns)
data.head(10)


# ### Understanding Dataset

# The test consisted of 32 questions. They were shown to each subject in a random order. There were four values stored for each question:
# 
# * QXS	The number of right answers selected for this question minus the number of wrong answers.
# * QXA	The exact answers selected for this question.
# * QXE	The time spend on this question is miliseconds.
# * QXI	The position this question in the survey. (X=question ID number)
# 
# There values were collected on a final page:
# * age		in years
# * gender		1=Male, 2=Female, 3=Other
# * engnat		1=English native language, 2=Not native english speaker
# 
# These values were calculated from technical information:
# * country		the user's network location
# * screenw		screen width in pixels
# * screenh		screen height in pixels
# * introelapse	The time spend on the landing page in seconds
# * testelapse	The time spent on the page with the questions in seconds
# * surveyelapse	The time spent on the final page in seconds

# In[ ]:


data['Q2S'].count()


# In[ ]:


data['Q1S'].describe()
#stats about right answers for the Question 1


# ### Cleaning Dataset

# In[ ]:


#columns with exact answers (QxA) are not needed, 
#we have correct answers number for each question withn QxS. 
#We also remove QxI(question order in which it was displayed for the user) 
#and some other irrelevant columns (screen width, time spent oparticular pages, etc.)

cleaned = data.drop(columns=[("Q"+ str(i)+"A") for i in range(1,33)])
cleaned2 = cleaned.drop(columns=[("Q"+ str(i)+"I") for i in range(1,33)])
cleaned3 = cleaned2.drop(columns=["screenw","screenh","introelapse","testelapse","surveyelapse"])


cleaned3.head(5)


# In[ ]:


#We need to make values more readable, so inplace replacing is a good option

cleaned3.gender.replace([1, 2, 3, 0], ["male", "female", "other", "other"], inplace=True)
cleaned3.engnat.replace([1, 2], ["yes", "no"], inplace=True)
cleaned3.head(5)


# In[ ]:


def relative_values(x):
    y = (x +5)*10
    return y

def to_seconds(x):
    y = round(x/1000,1)
    return y


# In[ ]:


relative_values_cleaned = cleaned3

#to make analysis is cleaner and more representative let's transform 'correct' answers for 
#each question to relative (from -5 to 5, totally 10 options) 

for i in range(1,33):
    clmn = "Q"+str(i)+"S"
    relative_values_cleaned[clmn] = relative_values_cleaned[clmn].apply(relative_values)
relative_values_cleaned.head(5)


# In[ ]:


#also let's change time spent for each questions to be represented in seconds, 
#not milliseconds. Plus let's round it as well.

for i in range(1,33):
    clmn = "Q"+str(i)+"E"
    relative_values_cleaned[clmn] = relative_values_cleaned[clmn].apply(to_seconds)

relative_values_cleaned.head(5)


# In[ ]:


#Now we compute average correct score per user
relative_values_cleaned['QxS_avg'] = relative_values_cleaned[[("Q"+ str(i)+"S") for i in range(1,33)]].mean(axis=1)


# In[ ]:


#Now we compute average time spent for the whole questions answered per user
relative_values_cleaned['QxE_avg'] = relative_values_cleaned[[("Q"+ str(i)+"E") for i in range(1,33)]].mean(axis=1)


# In[ ]:


#function to convert 2-letter country codes to 3-letter ones to be used for Folium module

def country3(x):
    try:
        return pycountry.countries.lookup(x).alpha_3
    except Exception:
        pass


# In[ ]:


relative_values_cleaned["country_code"]=relative_values_cleaned["country"].apply(country3)


# In[ ]:


#creating new DF with columns we want to analyse 
aggregated_stats= relative_values_cleaned[['age','gender', 'engnat', 'country_code', 'QxS_avg', 'QxE_avg']]


# In[ ]:


aggregated_stats.head(5)


# In[ ]:


#We need to add new column with the number of participants who took the test per country

aggregated_stats['participants'] = aggregated_stats.groupby(['country_code'])['QxS_avg'].transform('count')
aggregated_stats.head(5)


# ## Data analasys

# ### Number of test takers by country

# In[ ]:


aggregated_stats[['participants', 'country_code']].groupby('country_code', as_index=False)['participants'].mean().sort_values(by='participants', ascending=False)[:10].plot(kind='bar',title="Participants by country (most 10)", x='country_code',y='participants',figsize=(15,8))
plt.show()


# From the figure, it is quite clear the most number of test-takers come from the USA. Top 5 countries include only one non-native English country, i.e. Germany. 

# ###  Average scores for native English participants

# In[ ]:


filter1 = aggregated_stats['participants'] >=100

aggregated_stats[filter1][['QxS_avg', 'country_code']].groupby('country_code', as_index=False)['QxS_avg'].mean().sort_values(by='QxS_avg', ascending=False)[:10].plot(kind='bar',ylim=70,title="Average scores by country (most 10) with more than 100 participants", x='country_code',y='QxS_avg',figsize=(15,8))
plt.show()


# Ireland, Canada and USA are obvious top 3 among all native English test takers.
#  

# ### Average scores for non-native English participants

# In[ ]:


filter1 = aggregated_stats['engnat'] == "no"
filter2 = aggregated_stats['participants'] >=10

aggregated_stats[filter1][filter2][['QxS_avg', 'country_code']].groupby('country_code', as_index=False)['QxS_avg'].mean().sort_values(by='QxS_avg', ascending=False)[:10].plot(kind='bar',ylim=75,title="Average scores by country (most 10) with more than 10 participants, non-native EN speakers", x='country_code',y='QxS_avg',figsize=(15,8))
plt.show()


# Venezuela, Austria and Sweden are the top three among all other non-native English test takers.

# ###   Average scores by age of participants

# In[ ]:


filter1 = aggregated_stats['age'] <= 85
aggregated_stats[filter1][['QxS_avg', 'age']].groupby('age', as_index=False)['QxS_avg'].mean().sort_values(by='age', ascending=False).plot(kind='line',title="Average scores by age of participants", x='age',y='QxS_avg',figsize=(15,8), grid='true')
plt.show()


# Participants of age 30 to 70 get, on average, the same scores for the test. There are some fluctuations after 70, yet to be not so important, taking into account a few numbers of participants 

# ###  Average scores by age for male and females

# In[ ]:


filter1 = aggregated_stats['age'] <= 85
filter_male = aggregated_stats['gender'] =='male'
filter_female = aggregated_stats['gender'] =='female'


ax = plt.gca()

aggregated_stats[filter1][filter_male][['QxS_avg', 'age']].groupby('age', as_index=False)['QxS_avg'].mean().sort_values(by='age', ascending=False).plot(kind='line',title="Average scores by age for male and females", x='age',y='QxS_avg',figsize=(15,8), grid='true', ax=ax)
aggregated_stats[filter1][filter_female][['QxS_avg', 'age']].groupby('age', as_index=False)['QxS_avg'].mean().sort_values(by='age', ascending=False).plot(kind='line', x='age',y='QxS_avg',figsize=(15,8), grid='true',color='red',ax=ax)
plt.show()


# Generally speaking in terms of this test, men perform slightly than women for all ages, except after the age of ~78 (on the figure, blue - men, red - women). 

# ###   Average scores for countries with 100+ test takers

# In[ ]:


#Let's check what countries perform better taking into account their native language is NOT English
value_to_map = "QxS_avg"
filter1 = aggregated_stats['engnat']== "no"
filter2 = aggregated_stats['participants'] >=10


aggregated_stats_with_filters = aggregated_stats[filter1][filter2]

country_geo = '/kaggle/input/python-folio-country-boundaries/world-countries.json'
plot_data = aggregated_stats_with_filters[["country_code",value_to_map]]
map = folium.Map(location=[0, 0], zoom_start=2.0)
map.choropleth(geo_data=country_geo, data=plot_data,
             columns=["country_code", value_to_map],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name="Average rating (correct answers %) by country with filters applied")
map.save('plot_data1.html')

from IPython.display import HTML
HTML('<iframe src=plot_data1.html width=900 height=450></iframe>')


# From this interactive map, we can see the test takers distribution by country and their average scores. 

# ###  Average scores for non-native English on map

# In[ ]:


#Countries with the highest rating AND native English speakers
value_to_map = "QxS_avg"
filter1 = aggregated_stats['participants']>=100

aggregated_stats_with_filters = aggregated_stats[filter1]

country_geo = '/kaggle/input/python-folio-country-boundaries/world-countries.json'
plot_data = aggregated_stats_with_filters[["country_code",value_to_map]]
map = folium.Map(location=[0, 0], zoom_start=2.0)
map.choropleth(geo_data=country_geo, data=plot_data,
             columns=["country_code", value_to_map],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name="Average rating (correct answers %) by country with filters applied")
map.save('plot_data2.html')

from IPython.display import HTML
HTML('<iframe src=plot_data2.html width=900 height=450></iframe>')


# From this interactive map we can see, that non-native takers perform generally on the same level, except for China, Finland, Norway and Mexico. 

# ## Overview

# ### Limitations

# * The data could be incomplete. Missing values, even the lack of a section or a substantial part of the data, could limit its usability. 
# * Participants do not always provide accurate demographic information. 
# * A relatively small number of participants (~19000) 
# * Highly biased in numbers for English native participants (the test was conducted in English only). 
# * No general knowledge test can be culturally unbiased, the MGKT will have the greatest validity for internet users from the United States. 

# ### Conclusions

# * Ireland, Canada and USA are obvious top 3 among all native English test takers. Venezuela, Austria and Sweden are the top three among all other non-native English test takers. 
# * Men perform slightly better than women of all ages, except after the age of ~78.
# * Generally, English native speakers perform better on the test, which is quite obvious due to the test language. 
# * Participants of age 30 to 70 get, on average, the same scores for the test. There are some fluctuations after 70. 

# ### Acknowledgements

# This data was collected through an online general knowledge test. Data collection took place 2017-2018. Users were motivated to take the test to obtain personalized results. This test used a unique question format, where each question was of a type that could have multiple correct answers. In each question, 10 answers were displayed to the respondent and they were told that 5 were correct and to select as many as they knew but not to guess. 
# 
# Source of the MGKT Dataset - https://openpsychometrics.org/_rawdata/

# ### References

# The analysis was performed by the author of this project based on MGKT Dataset only and third party Python modules (Folium, Pandas). 
