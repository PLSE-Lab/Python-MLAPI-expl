#!/usr/bin/env python
# coding: utf-8

# # Data Exploration for CareerVillage.com Challenge
# In this notebook I will explore the data available for the challenge, including basic stats, join patterns (data model image below) and some initial thoughts on analysis. Hope you find it useful.
# 
# Let's start by examining the organization. [CareerVillage.com](https://www.careervillage.org/) is a platform for *students* to ask *questions* related to career and for *professionals* to *answer* these questions. There can be multiple answers per question, as professionals can differ in opinion or they can build on previous answers. Questions can also be *tagged* (e.g. #technology #computer-science), which I guess is what CareerVillage currently uses to *match* questions to specific professionals on their *email* distribution list. Finally, we know a bit about the users:
# * For professionals we have location (city, state), industry, headline, and date they joined the platform
# * For students we know location, school membership, group membership
# 
# Note also that user profiles can also be *tagged* for both professionals and students, similarly both types of users can be part of *groups*. This is not captured in the entity relationship diagram below.

#  ![DB Image](https://i.ibb.co/2PDK5ws/download.png)

# Let's look at the central topic of the challenge - the **questions** that students ask. We have close to 24,000 questions asked in the span of 8 years. The challenge didn't make it clear wheather this is a sampling of questions or the complete set, but what I think will be important is how (if at all) do the questions change over time. In the meantime, here's how the volume of questions breaks down by year. You can see when the platform gained popularity, in 2016 there were over 9,000 questions asked. There is a strange dip in 2017 which I can't yet explain, and of course year 2019 has just begun. 

# In[ ]:


# The usual suspects for data processing and visualization
import pandas as pd 
import seaborn as sns
import matplotlib as plt
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')


# Load the questions data and process dates properly
questions = pd.read_csv('../input/questions.csv', parse_dates=['questions_date_added'])
min_q_date = min(questions['questions_date_added'])
max_q_date = max(questions['questions_date_added'])
print('There were {:,} questions asked between {} and {}'.format(questions.shape[0], min_q_date.strftime('%Y-%m-%d'), max_q_date.strftime('%Y-%m-%d')))

# Plot count of questions accross years
sns.set_style("white")
sns.countplot(x=questions['questions_date_added'].dt.year, data=questions, facecolor='darkorange').set_title('Volume of Questions per Year')
sns.despine();


# Next, let's explore the **answers**. The number of answers in the dataset is almost double the number of questions. This makes sense as a question can have multiple answers linked to it and there are likely some questions that a lot of professionals want to weigh in on. So, this all makes sense. 

# In[ ]:


answers = pd.read_csv('../input/answers.csv', parse_dates=['answers_date_added'])
min_a_date = min(answers['answers_date_added'])
max_a_date = max(answers['answers_date_added'])
print('There were {:,} answers provided between {} and {}'.format(answers.shape[0], min_a_date.strftime('%Y-%m-%d'), max_a_date.strftime('%Y-%m-%d')))

# Plot count of questions accross years
sns.set_style("white")
sns.countplot(x=answers['answers_date_added'].dt.year, data=answers, facecolor='darkorange').set_title('Volume of Answers per Year')
sns.despine();


# To join answers and questions we use the *question_id* as follows. Note that the count of questions dropped down to around 23,000, which means that there is only about 1,000 questions that went unanswered (or roughly 3%). This is a lot lower than I expected, to be honest. So, so far CareerVillage seems to be doing pretty well at getting students questions answered. 

# In[ ]:


q_a = questions.merge(right=answers, how='inner', left_on='questions_id', right_on='answers_question_id')
print('There are {:,} questions that got answered, which is {:.0f}% of all questions.'.format(q_a['questions_id'].nunique(), 100*q_a['questions_id'].nunique()/questions.shape[0]))


# This is work in progress and I hope to continue this data exploration as time permits...
