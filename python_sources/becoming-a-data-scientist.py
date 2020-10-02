#!/usr/bin/env python
# coding: utf-8

# > A 2016 report from McKinsey found that women made up 37% of entry-level roles in tech (compared to 45% in the overall sample), and only 25% advanced to senior management roles. **Just 15% reached the C-suite.** -[World Economic Forum](https://www.weforum.org/agenda/2017/08/women-in-tech-gender-parity/)-
# 
# This statistics lingered. I knew nothing about coding but it sounded serious. Then there was a TedTalk by Reshma Saujani, the founder of Girls Who Code. There was a clear urge: It was more than just fewer women working, it was about setting social expectations on women to dream bigger. 
# 
# A few years later, I sit here with the question: "What are the odds of a 27-yr-old Asian women with a business undergraduate degree becoming a self-taught data scientist?" 
# 
# <img src="https://i.ibb.co/XzqG9G6/non-tech-woman-in-tech.png" alt="non-tech-woman-in-tech" width="80%"><br/>
# Kaggle's 2018 ML & DS Survey Challenge collected data from the responses of the 23,859 participants from 147 different regions. There were three primary questions I asked: 
# * **What are the key differences between those who identify as data scientists from other participants? **
# 
#     *[Step One] Divide data set into two categories according to the question "Are you a data scientist?" - Positive/Negative*
# * **Which subgroup do I belong to?**
# 
#     *[Step Two] Highlight relevant subgroups in both positive/negative data sets*
# * **What would raise my chances of becoming a data scientist? **
# 
#     *[Step Three] Analyze 'positive' data set and understand what skill sets would be required.*
# 

# In[ ]:


#Import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Read data from "multipleChoiceResponses.csv" file
data=pd.read_csv('../input/multipleChoiceResponses.csv', skiprows=[1])
df = pd.DataFrame(data)


# In[ ]:


#Select column 'Q6' and select those who identify as "data scientists"
occupation = df['Q6'].value_counts()
total = occupation.sum()
data_positive = occupation['Data Scientist']
data_negative = total - data_positive

plt.figure(1, figsize=(14,10))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0], aspect=1, title='Are you a Data Scientist?')
plt.axis('equal')
labels = 'Yes (Positive)', 'No (Negative)'
sizes = [data_positive, data_negative]
colors = ['#d9b37c','#B94E8A']
explode = (0.15, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.subplot(the_grid[0, 1], aspect=1, title='If not, what is your occupation?')
num_da = occupation['Data Analyst']
num_se = occupation['Software Engineer']
num_st = occupation['Student']
num_ne = occupation['Not employed']
num_other = (data_negative - (num_da + num_se + num_st + num_ne))
plt.axis('equal')
labels = 'Data Analyst', 'Software Engineer', 'Student', 'Other', 'Not Employed(*)'
sizes = [num_da, num_se, num_st, num_other, num_ne]
colors = ['#FFD7F1', '#D87CA1', '#B94E8A', '#7D156D', '#fff5ff']
explode = (0, 0, 0, 0, 0.4)

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=360)

plt.suptitle('2018 Kaggle ML & DS Survey Challenge Results', fontsize=22)

plt.show()


# <h3>Step One - Summary</h3> <br/> 
# Out of 23,859 participants, **22,900 people** wrote their occupations. Therefore, from this point forward, (n=22,900). <br/>
# Only 4137(**18.1%**) of the written responses answered Yes to the question *"Are you a Data Scientist?"* <br/>
# Among those who answered No, the first majority were students (28%), followed by software engineers (16.7%). <br/>
# 
# 4137 are working professionals who identified as Data Scientists and therefore will represent the population forward.
# 
# What's interesting is that when asked "Do you consider yourself to be a data scientist?", 4684 people answered "Definitely yes". <br/>
# There is discrepancy between **4137** (occupation) and **4684** (self perception). <br/>
# 
# It's likely that there are data scientists unemployed or in other fields looking to transition, but I am going to keep my data sets defined by *occupation* because that is less subjective. 

# In[ ]:


# from now on, n = 22,900
n = df[df['Q6'].notnull()]
# Question: "Are you a Data Scientist"?
# Yes - positive population
pos_pop = n.loc[n['Q6'] == 'Data Scientist']
# No - negative population
neg_pop = n.loc[n['Q6'] != 'Data Scientist'] 


# In[ ]:


agegroup = n['Q2']
age = agegroup.value_counts().sort_index()

sns.set(style="white", context="talk")
sns.color_palette("Set2")

# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
x1 = age.index
y1 = age
sns.barplot(x=x1, y=y1, ax=ax1)
ax1.set_title("Age Group of Participants")
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Number of ppl")
plt.subplots_adjust(hspace = 0.3)

#question - 'are you a data scientist'? - age group
# "no" negative
y2 = neg_pop['Q2'].value_counts().sort_index() 
# "yes" positive
y3 = pos_pop['Q2'].value_counts().sort_index()

p1 = plt.bar(x1,y2,color="#B94E8A")
p2 = plt.bar(x1,y3,color="#d9b37c")
plt.plot(2,1400,'wd', markersize=16)
plt.text(2, 650, '33.09%', horizontalalignment='center', color="w", size=12)
plt.text(1, 300, '17.26%', horizontalalignment='center', color="w", size=12)
plt.text(3, 400, '21.1%', horizontalalignment='center', color="w", size=12)

plt.ylabel('Number of ppl')
plt.legend((p2[0], p1[0]), ('Yes', 'No'))
ax2.axhline(0, color="k", clip_on=False)
plt.title("Are you a Data Scientist?")
plt.show()


# In[ ]:


yes_percentage = []
for x in range(0,12): 
    yes_percentage.append(str(round((y3[x]/4137*100),2)) + "%")
    


# <h4>[Age Group]</h4>
# Out of 2290 survey participants, 4137 (18.1%) people were Data Scientists.<br/>
# 1369 **(33.09%)** Data Scientists were within the age group 25-29. <br/>
# This means about **6% of the total survey participants were 25-29 yr old Data Scientists**. <br/>
# 
# Among those who answered that they were not Data Scientists, <br/>
# 4548 (24.24%) non-Data Scientists were within the age group 25-29. <br/> 
# This was about 20% of the total survey participants. <br/> 

# In[ ]:


#question - 'are you a data scientist'? - gender
# "no" negative
y4 = neg_pop['Q1'].value_counts().sort_index() 
# "yes" positive
y5 = pos_pop['Q1'].value_counts().sort_index()

gender_group = n['Q1']
gender = gender_group.value_counts().sort_index()
x2 = gender.index

plt.figure(1, figsize=(14,10))
the_grid = GridSpec(3, 3)

plt.subplot(the_grid[0, 0], aspect=1, title='Gender of Participants')
plt.axis('equal')
labels = 'Female', 'Male', '', ''
sizes = gender
colors = ['#FFBE00', '#005874', '#E6E6D4', '#E6E6D4']
explode = (0.15,0,0,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, startangle=140)
plt.text(-1,0.08, '16.8%', size=12)
# 16.8% were women

plt.subplot(the_grid[0, 1], aspect=1, title='Gender of Data Scientists')
plt.axis('equal')
sizes = y5
colors = ['#FFBE00', '#005874', '#E6E6D4', '#E6E6D4']
plt.pie(sizes, labels=labels, colors=colors, shadow=True, explode=explode, startangle=130)
plt.text(-0.98,0.2, '16.6%', size=12)

plt.subplot(the_grid[0, 2], aspect=1, title='Gender of Non-Data Scientists')
plt.axis('equal')
sizes = y4
colors = ['#FFBE00', '#005874', '#E6E6D4', '#E6E6D4']
plt.pie(sizes, labels=labels, colors=colors, shadow=True, explode=explode, startangle=130)
plt.text(-0.98, 0.2, '16.9%', size=12)
plt.show()


# <h4>[Gender]</h4> 
# It was remarkably interesting to see how women consistently represented within the 16-17% range of any categorization. <br/>
# 
# Out of 22900 total survey participants, **16.8%** answered 'Female'. <br/>
# Among Data Scientists, **16.6%** answered 'Female'. <br/>
# Among those who were not Data Scientists, **16.9%** answered 'Female'. <br/>
# <img src="https://i.ibb.co/MfHBnyR/non-tech-woman-in-tech-1.png" alt="non-tech-woman-in-tech-1" width="80%">
# The results closely correlate to the McKinsey article about how only **15% of women make it to the C-Suite level in the tech industry**. <br/>
# It's quite astonishing how consistent it is in my opinion.

# In[ ]:


# correlations between undergrad studies and occupation
data_uni = n.groupby(['Q5','Q6'])['Q6'].count().to_frame(name = 'count').reset_index()
# some participants did not answer the question in the survey
#data_uni.fillna('Unknown', inplace=True)
data_scientist = data_uni.loc[data_uni['Q6'] == 'Data Scientist']
data_scientist = data_scientist.sort_values('count', ascending=False)
business = data_uni.loc[data_uni['Q5'] == 'A business discipline (accounting, economics, finance, etc.)']
business = business.sort_values('count', ascending=False)

#plt.figure(1, figsize=(14,10))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 12))
sns.barplot(x="count", y="Q5", data=data_scientist, ax=ax2)
sns.barplot(x="count", y="Q6", data=business, ax=ax1)

ax1.set_ylabel('')    
ax1.set_xlabel('(total=1,760)')
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.invert_xaxis()
ax1.set_title("Business Majors")
ax1.text(122, 0,'Data Scientist (291)', verticalalignment='center',fontsize=14, bbox=dict(facecolor='red'))
ax1.text(118, 1,'Data Analyst (267)', verticalalignment='center',fontsize=14)
ax1.text(142, 3,'Business Analyst (205)', verticalalignment='center',fontsize=14)
ax1.text(120, 14,'Data Engineer (28)', verticalalignment='center',fontsize=14)
ax1.text(168, 18,'DBA/Database Engineer (5)', verticalalignment='center',fontsize=14)
ax1.text(114, 20,'Data Journalist (2)', verticalalignment='center',fontsize=14)

ax2.set_ylabel('')    
ax2.set_xlabel('(total=4,115)')
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_title("Data Scientists")

ax2.text(5, 0,'Computer Science (1403)', verticalalignment='center',fontsize=14)
ax2.text(5, 1,'Mathmatics/Statistics (835)', verticalalignment='center',fontsize=14)
ax2.text(5, 2,'Engineering (645)', verticalalignment='center',fontsize=14)
ax2.text(5, 3,'Physics/Astronomy (339)', verticalalignment='center',fontsize=14)
ax2.text(5, 4,'Business (291)', verticalalignment='center',fontsize=14, bbox=dict(facecolor='red'))


# <h4>Undergraduate Studies</h4>
# My next question was "Who are interested in becoming a data scientist?" Of course, anyone could, but I believe the idiom *birds of a feather flock together* and want to know if there is any correlation between their undergraduate studies and occupation and see where business survey participants placed themselves. I also wanted to know how many business students would self-perceive themselves as data scientists. Again, I truly believe that anyone can become whomever they dream of becoming, but **this is primarily to observe any particular group behaviors/trends.  **
# <img src="https://i.ibb.co/LvHchPM/222.png" alt="222" border="0"><br/>
# Out of 22900 participants, 22,678 answered both survey questions about occupation and undergraduate studies. <br/>
# 1,760 participants studied business in university. 798 of them were working in the realms of data (data scientist/analyst/journalist)
# **That's 45% of business majors.** <br/>
# 291 of the business majors ended up becoming a data scientist. **They are the fifth largest group within the Data Scientist population.** <br/>
# In other words, according to this study, even if the numbers are small, it's not unusual for Business majors to be interested/working as a Data Scientist.

# In[ ]:


# Data Scientists recommend this language for aspiring newcomers
rec_lan = pos_pop['Q18'].value_counts()
x3 = rec_lan.index
str(round((rec_lan[0]/data_positive*100),2))+"% of the data scientists said that they recommend Python for aspiring data scientists to learn first."


# In[ ]:


# how many data scientists are 'self-taught'?
self_taught = pos_pop['Q35_Part_1'].dropna()
self_taught = self_taught.sort_values(ascending=False)
self_taught = self_taught[self_taught >= 50.0]
online_course = pos_pop['Q35_Part_2'].dropna()
online_course = online_course.sort_values(ascending=False)
online_course = online_course[online_course >= 50.0] 

st = self_taught.describe().to_frame().rename(columns={"Q35_Part_1": "Self-taught"})
oc = online_course.describe().to_frame().rename(columns={"Q35_Part_2": "Online Course"})

frames = [st,oc]
pd.concat(frames, axis=1)


# <h4>Self-Taught Data Scientists</h4>
# In one of the survey questions, it asked people "What percentage of your current machine learning/data science training falls under each category?" <br/>
# 
# People's definition will vary regarding the term "Self-taught", but this was my criteria:
# * Their learning is not in a traditional setting (ex: university, work).
# * The specific categories are 'Self-taught' or 'Online-Course.'
# * They spend more than 50% of their time in this particular category.
# 
# I wanted to know how many Data Scientists identify as 'Self-taught'. <br/>
# There were **1048 ** people who fell under my criteria. (min = 50%, max= 100%)
# * 'self_taught' - 612 (mean 63%) 
# * 'online_course' - 436 (mean 61%)
# 
# **That means 1 out of 4 Data Scientists (total: 4137) are "self-taught".** How exciting! 
# 

# In[ ]:


# Asking the Data Scientists - During a typical data science 
# project at work or school, approximately what proportion of your time is devoted to the following? 

plt.figure(1, figsize=(15,10))
plt.subplot(2,3,1) 
ax1 = sns.boxplot(y=pos_pop['Q34_Part_1'], palette="Set3")
ax1.set_ylabel("proportion of time devoted(%)")
ax1.set_xlabel("Gathering data")
plt.subplot(2,3,2)
ax2 = sns.boxplot(y=pos_pop['Q34_Part_2'], palette="Set3")
ax2.set_ylabel("")
ax2.set_xlabel("Cleaning data")
plt.subplot(2,3,3)
ax3 = sns.boxplot(y=pos_pop['Q34_Part_3'], palette="Set3")
ax3.set_ylabel("")
ax3.set_xlabel("Visualizing data")
plt.subplot(2,3,4)
ax4 = sns.boxplot(y=pos_pop['Q34_Part_4'], palette="Set3")
ax4.set_ylabel("proportion of time devoted(%)")
ax4.set_xlabel("Model building/selection")
plt.subplot(2,3,5)
ax5 = sns.boxplot(y=pos_pop['Q34_Part_5'], palette="Set3")
ax5.set_ylabel("")
ax5.set_xlabel("Model production")
plt.subplot(2,3,6)
ax6 = sns.boxplot(y=pos_pop['Q34_Part_6'], palette="Set3")
ax6.set_ylabel("")
ax6.set_xlabel("Finding insights & communicating")


# <h4> [Proportion of Time Spent in a Typical Data Science Project] </h4>
# From the box plots, we can easily tell that **"Cleaning Data"** and **"Model building/selection"** consumed the majority's time. <br/>
# For both, the mean was approximately 20%. <br/>
# However, it is notable to see that 'Model building/selection' has more data variation compared to 'cleaning data'. In other words, it's more spread out.<br/>
# **The takeaway from this analysis is that to become a good data scientist, one should excel at these two skills. **
# 
# 

# <h4>Conclusion</h4>
# This is far from complete, but for now I am content with the findings I gathered from this study. <br/>
# It seemed like I have a good chance of becoming a Data Scientist, so that is encouraging :) <br/>
# I hope you enjoyed my data journey. <br/>
# <img src="https://i.ibb.co/p4vSJPJ/non-tech-woman-in-tech-4.png" alt="non-tech-woman-in-tech-4" width="100%">

# 
