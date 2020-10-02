#!/usr/bin/env python
# coding: utf-8

# # School Subjects Analysis

# <img src="http://www.chicagonista.com/wp-content/uploads/2012/03/veropost.jpg">

# Hello everyone!
# 
# This is simple EDA that I took this week just for practicing some python and analysis skills. I uploaded this dataset that I got from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Student+Performance) to make this analysis.
# 
# It's been a while since I want to do a study about education, get to know how students and schools behave, what they have in common and what we can learn from that.

# ## Summary
# ** 1. [Introduction](#introduction)** <br>
# ** 2. [Loading Data](#loading)** <br>
# ** 3. [Transforming Values and Types](#transforming)** <br>
# ** 4. [Exploratory Data Analysis](#eda)** <br>
# ** 5. [Conclusions](#conclusions)** <br>

# <a id="introduction"></a>
# ## 1. Introduction
# 
# This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. 
# 
# Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). 
# 
# In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades.

# ## Importing Packages

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# <a id="loading"></a>
# ## 2. Loading Data

# In[3]:


path_math = "../input/student-mat.csv"
path_port = "../input/student-por.csv"

math = pd.read_csv(path_math, sep=";")
port = pd.read_csv(path_port, sep=";")


# In[4]:


# Adding the name of the subject

math['subject'] = 'mathematics'
port['subject'] = 'portuguese'


# In[5]:


# Concatenating dataframes

data = pd.concat([math, port])


# ## Getting to Know the Data

# In[6]:


data.head()


# In[7]:


data.describe()


# In[8]:


data.info()


# The good thing about this dataset is that we don't have to fill any missing values. But I still want to make some changes

# <a id="transforming"></a>
# ## 3. Transforming Values and Types
# 
# Some columns have numbers that represent categorical values. I'm going to change the name in some of these columns to make clearer what they mean.
# 
# I'm also going to change the type of the columns labeled numeric when they actually are categories

# In[9]:


data['traveltime'] = data['traveltime'].map({1: '<15m', 2: '15-30m', 3: '30-1h', 4: '>1h'})

data['studytime'] = data['studytime'].map({1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'})


# In[10]:


data[['Medu','Fedu','famrel','goout','Dalc','Walc','health']] = data[['Medu','Fedu','famrel','goout','Dalc','Walc','health']].astype('object')


# In[11]:


data.head()


# <a id="eda"></a>
# ## 4. Exploratory Data Analysis

# In[18]:


plt.figure(figsize=(11,7))
sns.set(font_scale=1.5)
sns.set_style('white')

ax = sns.countplot(y='reason',data=data, order=data['reason'].value_counts().index, palette="Blues_d")

ax.set_yticklabels(('Course Preference', 'Close to Home', 'School Reputation', 'Other'))
plt.ylabel('Reason')
plt.xlabel('Count')
plt.title('Reason For Choosing The School')


# Here we can see that the course of interest is more important for the student when choosing a school

# In[13]:


plt.figure(figsize=(10,8))
sns.set(font_scale=1.5)
sns.set_style('white')

ax = sns.barplot(x='subject', y='G3', hue='sex', data=data, palette=["#ff4949", "#456f95"])

ax.set_xticklabels(('Mathematics', 'Portuguese'))
plt.xlabel('Subject')
plt.ylabel('Final Grade - G3')
plt.title('Comparison of grades between men and women by subject')


# We can see by this chart that men got higher grades in mathematics compared to women, while women got better grades in portuguese. 
# 
# Overall both had better results in portuguese. 

# In[19]:


plt.figure(figsize=(11,7))
sns.set(font_scale=1.4)
sns.set_style('white')

plt.hist(data.G3, alpha=0.8, bins=10)

plt.xlabel('Final Grade - G3')


# Grade goes from 0 to 20. the histogram show us that most of the students got around 10-15.

# In[20]:


plt.figure(figsize=(11,10))
sns.set(font_scale=1.6)
sns.set_style('whitegrid')


sns.boxplot(x='studytime', y='G3', data=data, order=['<2h','2-5h','5-10h','>10h'])

plt.xlabel('Study Time')
plt.ylabel('Final Grade - G3')
plt.title('Final Grades by Study Time')


# We can take a lot of information from these boxplots. First thing we notice (and expected) is that who studied less time, had the lowest grades. We can also see that 2-5h os study time has the same median as <2h but third quartile is bigger.
# 
# There are not so many differences between who studied 5-10 hours and more than 10 hours, of course we can see that the highest grade were only achieved by who studied for the longest period of time.
# 
# And finally we can observe that in these four categories there are some outliers on the 0.0 mark. Probably beacause they didn't attend to the tests or a high number of absences makes the school annul the student's grades.
# 
# Talking about absences, let's check if that correlates with the students' grades.

# In[21]:


plt.figure(figsize=(13,8))
sns.set(font_scale=1.6)
sns.set_style('white')

sns.scatterplot(y='G3', x='absences', hue='traveltime', data=data, s=120,
                palette=["#c0b8b8", "#3498db", "#c0b8b8", "#ec2626"])

plt.xlabel('Absences')
plt.ylabel('Final Grade - G3')
plt.title('Relation Between Grades And Absences by Travel Time')


# Contrary to what we thought, grades and absences are not correlated. Most of the students had between 0 and 20 absences, but the grades still are pretty scattered.
# 
# Another hypothesis was about the time that a student takes from home to school. If a longer distance could afect the grades somehow. For example, less time traveling from home to school, means more time the person has to study. So I highlighted the students that take more and less time to arrive at school. We don't have enough data of the ones who take more than 1 hour, but we can see that the grades of who takes 15 minutes or less are also very scattered.

# ## After all, which school is better?

# In[22]:


plt.figure(figsize=(13,8))
sns.set(font_scale=1.5)
sns.set_style('white')

ax = sns.barplot(x='school', y='G3', hue='subject', data=data, palette=['#80aaff', '#ffd580'])

ax.set_xticklabels(('Gabriel Pereira', 'Mousinho da Silva'))
plt.xlabel('School Name')
plt.ylabel('Final Grade - G3')
plt.title('Comparison Of Grades Between Schools')


# <a id="conclusions"></a>
# ## 5. Conclusions
# 
# This EDA showed us some differences between these two schools, students preferences, some insights that we already thought and others that were not what I was expecting.
# 
# This is not a final analysis, I still want to get more insights from this dataset and make a supervised model to predict a student grade.
# 
# So if you liked this post, don't forget to give me an upvote and comment (omg, I sound like an youtuber haha) what else I could do with this dataset, and if I did something wrong or that could be done better, please let me know!
