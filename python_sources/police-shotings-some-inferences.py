#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing the dataset
# 

# In[ ]:


import seaborn as sns
data = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv", index_col = "id")
data


# # Looking and changing the index to ID
# I choose ID as the indexing column because ID was useless in terms of any particular analysis (on a hindsight I could've just dropped it entirely!)

# In[ ]:


data.index = list(range(len(data.index)))
data.index


# # Very Basic EDA

# Here we see the different types of values stored in each column

# In[ ]:


data.info()


# # Missing Values

# Just a simple check to see which column has how many missing values. I choose this method because it gives a clear picture as to what we could do to handle missing values

# In[ ]:


data.isnull().apply(pd.value_counts)


# Here each column is being checked for the percentage of the missing data in it to decide what could be done for each column

# In[ ]:


missing = [(col_name, ((data[col_name].isnull().sum())/data[col_name].value_counts().sum())*100) for col_name in data.columns if data[col_name].isnull().any() == True]
missing


# As we can see the maximum percentage of missing values is 10% in terms of actual values it amounts to 521 and for a dataset having over 5K rows dropping them would not neccasarlily harm any inferences that we would draw out of this data

# In[ ]:


data.dropna(inplace = True)


# # Date time changes
# 

# It is been done to ensure that we can have a closer look at the months and the years in which the shooting tends to happen

# In[ ]:


data["month"] = pd.to_datetime(data["date"]).dt.month
data["year"] = pd.to_datetime(data["date"]).dt.year
data.drop(columns = "date", inplace = True)


# # Analysis of City and State-wise shootings

# Here I have prepared a list of all the shooting in every city and then I have choosen the cities in which more than 30 shootings have occured to be treated as high shooting cities

# In[ ]:


data.city.unique()
city_crime = data.city.value_counts()
city_high_crime = [(city, city_crime[city]) for city in city_crime.index if city_crime[city] > 30]
city_high_crime


# The same analysis is been done for the states

# In[ ]:


state_crime = data.groupby("state")["name"].count()
state_high_crime = [(state, state_crime[state]) for state in state_crime.sort_values(ascending = False).index if state_crime[state] > 50]
state_high_crime
state_only_high_crime = [state_high_crime[i][0] for i in range(len(state_high_crime))]


# Now comes the meat and the very reason for my analysis to answer the age old question is Police Racist?

# # Racial Profiling of the data
Here we can see the number of people killed VS the Race they belonged
# In[ ]:


race = data["race"].value_counts()
fig, ax = plt.subplots(figsize = (10, 5))
sns.barplot( x = race.index, y = race )
plt.xlabel("Race")
plt.ylabel("Number of Shootings");


# ## What infereneces could be drawn?
# Just by looking at this data we can see the police is anything but racist. 
# 
# But then why was there a recent uprising of #BlackLivesMatter?
# 
# Well the answer is very simple. This data that we have here is biased. There is no problem in the data itself but the total amount or rather the propotion of the population belonging to each Racial Profile is missing. 
# ## What does that mean?
# 
# 
# Well, in a country like America there are more White people than African American. So, we really should not derive any conclusion just on the basis of number. We Have to factor in the population belonging in each racial profile before we can answer the Question is Police Racist?

# So, I was curious about this dataset because of the sheer volume of information it contains. So, I asked more questions to myself and here are the answers that I could find

# ## Is Race a deciding variable?
# 

# Here I plotted race against the threat level ( Too answer are African Americans considered to be a real threat?) and the nature of the person who was killed (i.e. were they Fleeing? How were they Fleeing?)

# In[ ]:



fig, a = plt.subplots(1, 2,figsize = (20, 5))
sns.countplot(data = data, x = "race", hue = "flee", ax = a[0])
a[0].set_xlabel("race vs fleeing")
a[1] = sns.countplot(data  = data, x = "race", hue = "threat_level", ax = a[1])
a[1].set_xlabel("race vs threat level")


# Answer: Now this might be overused throughout this analysis but just looking at these numbers wont really answer the underlying question. But we can get a estimate (Estimate within the people of different racial groups and how many of them were considered dangerous)
# 
# The data doesnt dissapoint. Yes, most of the people were considered dangerous and hence were shot down ( The truth However is yet to be seen #BlackLivesMatter)

# Some more racial profiling. Here I wanted to answer the question of how were each of the indviduals killed and what gender did they belong to. (Gender is non-binary but that is the limitation of this dataset unfortunately)

# ## The role of Gender within each racial profile and how are each individual getting killed based on thier race?
# 
# Does the police like to Taser White people more or The rest of the people (Yes, this was my question and I needed answers)

# In[ ]:



figm, ax = plt.subplots(1, 2, figsize = (20, 5))
sns.countplot(data = data, x ="race", hue = "gender", ax = ax[0])
sns.countplot(data = data, x = "race", hue = "manner_of_death", ax = ax[1])


# Ans: So, no the propotion of the people killed might not seem to be connected to the race but we can see that The number of People Tasered and who were African Americans VS those who were White are very similar 
# 
# (I wonder what does that mean!)

# ## The role of mental illness in the threat level (percived threat level) and The manner they were killed
# 
# YES! I have seen a lot of Mindhunter, so genuinely I was curious to see if Mental Illnesses have any effect on the nature of killings and the threat that they have been seen as

# In[ ]:


fig, ax =plt.subplots(1, 2, figsize = (20, 5))
threat = data[["threat_level", "signs_of_mental_illness"]]
sns.countplot(data = threat, x = "threat_level", hue = "signs_of_mental_illness", ax = ax[0])
sns.countplot(data = data[["manner_of_death", "signs_of_mental_illness"]], x = "manner_of_death", hue = "signs_of_mental_illness")


# So, the realtions i could draw from my analysis was there was no abrupt increase in the number of people who were considerd a threat and they had any underlying mental condition. One thing however I saw that the number of people who were mentally ill were majorly shot down and not tasered (Which is a good thing I guess?)

# ## Distribution of Age 

# So, I wanted to see how many people who were killed were young or old. What was the distribution?

# In[ ]:


plt.subplots(figsize = (10, 5))
sns.distplot( data["age"])


# As it turns out most of the people who were shot were in the age group of 20-40. So, young people are more nefarious.

# # The distribution of Year and month vs the shootings

# This is pretty straightforward. I wanted to see which month was hot and which year in particualr had the most number of deaths

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (20, 5))
sns.countplot(data  = data, x = "month",  ax = ax[0])
sns.countplot(data = data, x = "year",  ax = ax[1])


# # Including the racial profiling into the Mix
# 
# 

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (20, 5), sharey = True)
sns.countplot(data  = data, x = "month",hue = "race" , ax = ax[0])
sns.countplot(data = data, x = "year",hue = "race",  ax = ax[1])


# # The states VS the number of shootings

# Remeber the two lists from early on. Here I try to see which state has the most number of people killed in a graphical manner

# In[ ]:


plt.subplots(figsize = (20, 10))
sns.lineplot(data = state_crime)


# In[ ]:


data_high_state_crime = data[data.state.isin(state_only_high_crime)]


# So, I was trying to see in the states where there was high number of shootings were there any specific racial group being targetted but alas, the lack of the population distribution makes it really hard to answer this question

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (20, 5), sharey = True)
sns.countplot(data = data_high_state_crime, x = "race", ax = ax[0])
sns.countplot(data = data_high_state_crime, x = "race", hue= "threat_level", ax = ax[1])


# In[ ]:





# # Conclusion
# 

# So I tried to answer some of the questions which struck me the moment I saw this data. I am however in very early stages of Data Science, so feel free to let me know what I did wrong and how I can improve
# 
# 
# Thank You!

# In[ ]:




