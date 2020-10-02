#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# * [1. The CareerVillage.org analytics competition](#1.-The-CareerVillage.org-analytics-competition)
# * [2. Exploratory Data Analysis](#2.-Exploratory-Data-Analysis)
#   * [2.1 Students](#2.1-Students)
#   * [2.2 Professionals](#2.2-Professionals)
#    * [2.2.1 Locations](#2.2.1-Locations)
#    * [2.2.2 Industries](#2.2.2-Industries)
#   * [2.3 Questions and answers](#2.3-Questions-and-answers)
#    * [2.3.1 Questions tags](#2.3.1-Questions-tags)
#   * [2.4 Emails](#2.4-Emails)
#   * [2.5 Matches](#2.5-Matches)
# 

# # 1. The CareerVillage.org analytics competition
# 
# CareerVillage.org is a nonprofit that crowdsources career advice for underserved youth. Founded in 2011 in four classrooms in New York City, the platform has now served career advice from 25,000 volunteer professionals to over 3.5M online learners. The platform uses a Q&A style similar to StackOverflow or Quora to provide students with answers to any question about any career.
# 
# In this Data Science for Good challenge, CareerVillage.org, in partnership with Google.org, is inviting you to help recommend questions to appropriate volunteers. To support this challenge, CareerVillage.org has supplied five years of data.
# 
# **Problem Statement**
# The U.S. has almost 500 students for every guidance counselor. Underserved youth lack the network to find their career role models, making CareerVillage.org the only option for millions of young people in America and around the globe with nowhere else to turn.
# 
# To date, 25,000 volunteers have created profiles and opted in to receive emails when a career question is a good fit for them. This is where your skills come in. To help students get the advice they need, the team at CareerVillage.org needs to be able to send the right questions to the right volunteers. The notifications sent to volunteers seem to have the greatest impact on how many questions are answered.
# 
# Your objective: develop a method to recommend relevant questions to the professionals who are most likely to answer them.

# In[ ]:


import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# In[ ]:


print(os.listdir("../input"))


# CareerVillage.org has provided several years of anonymized data and each file comes from a table in their database.
# 
# * answers.csv: Answers are what this is all about! Answers get posted in response to questions. Answers can only be posted by users who are registered as Professionals. However, if someone has changed their registration type after joining, they may show up as the author of an Answer even if they are no longer a Professional.
# * comments.csv: Comments can be made on Answers or Questions. We refer to whichever the comment is posted to as the "parent" of that comment. Comments can be posted by any type of user. Our favorite comments tend to have "Thank you" in them :)
# * emails.csv: Each email corresponds to one specific email to one specific recipient. The frequency_level refers to the type of email template which includes immediate emails sent right after a question is asked, daily digests, and weekly digests.
# * group_memberships.csv: Any type of user can join any group. There are only a handful of groups so far.
# * groups.csv: Each group has a "type". For privacy reasons we have to leave the group names off.
# * matches.csv: Each row tells you which questions were included in emails. If an email contains only one question, that email's ID will show up here only once. If an email contains 10 questions, that email's ID would show up here 10 times.
# * professionals.csv: We call our volunteers "Professionals", but we might as well call them Superheroes. They're the grown ups who volunteer their time to answer questions on the site.
# * questions.csv: Questions get posted by students. Sometimes they're very advanced. Sometimes they're just getting started. It's all fair game, as long as it's relevant to the student's future professional success.
# * school_memberships.csv: Just like group_memberships, but for schools instead.
# * students.csv: Students are the most important people on CareerVillage.org. They tend to range in age from about 14 to 24. They're all over the world, and they're the reason we exist!
# * tag_questions.csv: Every question can be hashtagged. We track the hashtag-to-question pairings, and put them into this file.
# * tag_users.csv: Users of any type can follow a hashtag. This shows you which hashtags each user follows.
# * tags.csv: Each tag gets a name.

# In[ ]:


students = pd.read_csv("../input/students.csv", index_col = "students_id", parse_dates = ['students_date_joined'])
professionals = pd.read_csv('../input/professionals.csv', index_col = "professionals_id", parse_dates = ['professionals_date_joined'])
emails = pd.read_csv("../input/emails.csv", index_col = "emails_id")
questions = pd.read_csv('../input/questions.csv', index_col = "questions_id", parse_dates = ['questions_date_added'])
answers = pd.read_csv('../input/answers.csv', index_col = "answers_id", parse_dates = ["answers_date_added"])
tag_questions = pd.read_csv("../input/tag_questions.csv")
tags = pd.read_csv("../input/tags.csv")
matches = pd.read_csv("../input/matches.csv")



# # 2. Exploratory Data Analysis
# ## 2.1 Students
# Altogether, we have 30,971 unique students in the database with 2 variables per student.

# In[ ]:


students.shape
#students.students_id.nunique()
#students.info()


# In[ ]:


students.head()


# What we see is that most students are from the US and India. The location consists of the city and statename for the US students, and also the country for students outside of the US.

# In[ ]:


students_locations = students.students_location.value_counts().sort_values(ascending=True).tail(20)
students_locations.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Number of students by location", fontsize=20)
plt.xlabel('Number of students', fontsize=12)
plt.show()


# In[ ]:


print("The number of students without specified location is", students.students_location.isna().sum())


# ## 2.2 Professionals
# We have 28,152 professionals in our database with 4 variables.

# In[ ]:


professionals.shape


# In[ ]:


professionals.head()


# ### 2.2.1 Locations
# The Top20 locations with most professionals looks similar when compared to the students locations. However, this time only one of those Top20 locations is in India. Another interesting observation is that "Washington" has no state attached to it. I think this is because Washington DC is not a state, but actually a federal district in the US.

# In[ ]:


prof_locations = professionals.professionals_location.value_counts().sort_values(ascending=True).tail(20)
prof_locations.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Number of professionals by location", fontsize=20)
plt.xlabel('Number of professionals', fontsize=12)
plt.show()


# In[ ]:


print("The number of professionals without specified location is", professionals.professionals_location.isna().sum())


# ### 2.2.2 Industries

# In[ ]:


prof_industry = professionals.professionals_industry.value_counts().sort_values(ascending=True).tail(20)
prof_industry.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Number of professionals by industry", fontsize=20)
plt.xlabel('Number of professionals', fontsize=12)
plt.show()


# The number of distinct inductries is high, which means that the industries do not seem to be standardised well!

# In[ ]:


print("The number of professionals without specified industry is", professionals.professionals_industry.isna().sum())
print('The number of distinct industries is', professionals.professionals_industry.nunique())


# A sample of the professionals_headlines can be found below.

# In[ ]:


professionals.professionals_headline.sample(20)


# ## 2.3 Questions and answers
# Altogether, we have 23,931 questions with 4 variables

# In[ ]:


questions.shape


# In[ ]:


questions.head()


# In[ ]:


print("There are", questions.questions_author_id.nunique(), "unique questions_author_id's, which means that the students who have asked questions asked about 2 questions each on average.")


# Altogether, we have 51,123 answers which 4 variables.

# In[ ]:


answers.shape


# In[ ]:


answers.head()


# In[ ]:


print("There are", answers.answers_author_id.nunique(), "unique answers_author_id's, which means that the professionals who have given answers have given about 5 answers each on average.")


# ### 2.3.1 Questions tags
# To merge the tags with the questions, we need two files. Tag_questions contains the id's of both the questions and the tags.

# In[ ]:


tag_questions.head()


# The tags file contains the tag_name for each tag_id.

# In[ ]:


tags.head()


# After merging the tag names with the tag_questions file, the Top20 of most used question tags is shown below.

# In[ ]:


tag_questions = tag_questions.merge(right=tags, how="left", left_on="tag_questions_tag_id", right_on="tags_tag_id")
tag_questions_groups = tag_questions.tags_tag_name.value_counts().sort_values(ascending=True).tail(20)
tag_questions_groups.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Top20 Question tags", fontsize=20)
plt.xlabel('Number of questions with the tag', fontsize=12)
plt.show()


# ## 2.4 Emails
#  Each email corresponds to one specific email to one specific recipient. The frequency_level refers to the type of email template which includes immediate emails sent right after a question is asked, daily digests, and weekly digests. We have 1.8 million emails in our database.

# In[ ]:


emails.shape


# In[ ]:


emails.head()


# In[ ]:


print("There are", emails.emails_recipient_id.nunique(), "unique email recipients.")


# In[ ]:


emails.emails_frequency_level.replace(["email_notification_daily", "email_notification_immediate", "email_notification_weekly"], ["daily", "immediate", "weekly"], inplace=True)

email_nots = emails.emails_frequency_level.value_counts()

ax = plt.figure()
ax = email_nots.plot.bar(figsize=(10, 8), color='b', width=1, rot = 0)
plt.title("Email notifications", fontsize=20)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.tick_params(axis='x', labelsize=14)
plt.show()


# ## 2.5 Matches
# matches.csv: Each row tells you which questions were included in emails. If an email contains only one question, that email's ID will show up here only once. If an email contains 10 questions, that email's ID would show up here 10 times.
# 
# Altogether, we have 4,3 million matches.

# In[ ]:


matches.shape


# As I am also interested to see how many questions are raised by email type (daily, immediate, weekly), I have joined this info from the email file before showing the head() of the matches below.

# In[ ]:


matches.rename(columns={'matches_email_id': 'emails_id'}, inplace=True)
#The right join ensures that emails without any questions are also added (one entry with matches_question_id is NA)
matches = pd.merge(matches, emails['emails_frequency_level'].reset_index(), on="emails_id", how="right")
matches.head()


# In[ ]:


print(matches[matches.emails_frequency_level == "daily"].matches_question_id.count(), "questions were asked in daily emails")
print(matches[matches.emails_frequency_level == "immediate"].matches_question_id.count(), "questions were asked in immediate emails")
print(matches[matches.emails_frequency_level == "weekly"].matches_question_id.count(), "questions were asked in weekly emails")


# Of the large number of daily emails, the vast majority has 3 questions. As there is a huge drop in the number of questions per daily email after 3 questions per mail, I have split the graph of the number of questions in daily mails into two figures (please notice the very different scales!).The number of daily emails with 4 questions is still 4,000+. Small numbers of daily emails have a very large number of questions, and the the maximum number of questions found in a daily email is 90 (one email).
# 
# We can also see that almost all immediate emails have exactly one question (never more than one question, and sometimes no questions).
# 
# The graph of the weekly emails shows a strange pattern. Most weekly emails contain 19 questions (why this number?). Few weekly emails have more than 20 questions (the small amounts are not visible on the graph).
# 
# **Overall, the largest chunks within the total of about 1,800,000 emails are: almost 1,000,000 daily emails with three questions (about 3 million questions), about 300,000 daily emails with one question, and also about 300,000 immediate emails with one question. In addition, the about 230,000 questions are raised in the relatively small number of 12,000 weekly emails with 19 questions**.

# In[ ]:


fig = plt.figure(figsize=(20,15))
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

matches_count =  matches.groupby(['emails_id', 'emails_frequency_level']).count()['matches_question_id'].reset_index()
daily_counts = matches_count[matches_count.emails_frequency_level == "daily"].matches_question_id.value_counts().sort_index()
immediate_counts = matches_count[matches_count.emails_frequency_level == "immediate"].matches_question_id.value_counts().sort_index()
weekly_counts = matches_count[matches_count.emails_frequency_level == "weekly"].matches_question_id.value_counts().sort_index()[:31]

ax1 = plt.subplot(grid[0, :1])
ax1 = daily_counts[:4].plot.bar(color='b', width=1, rot=0)
plt.title("Daily emails part 1", fontsize=20)
ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('Number of emails', fontsize=14)
plt.xlabel('Number of questions per mail', fontsize=14)

ax11 = plt.subplot(grid[0, 1:])
ax11 = daily_counts[4:31].plot.bar(color='b', width=1, rot=0)
plt.title("Daily emails part 2", fontsize=20)
ax11.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('Number of emails', fontsize=14)
plt.xlabel('Number of questions per mail', fontsize=14)

ax2 = plt.subplot(grid[1, :1])
ax2 = immediate_counts.plot.bar(color='b', width=1, rot=0)
plt.title("Immediate emails", fontsize=20)
ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('Number of emails', fontsize=14)
plt.xlabel('Number of questions per mail', fontsize=14)

ax3 = plt.subplot(grid[1, 1:])
ax3 = weekly_counts.plot.bar(color='b', width=1, rot=0)
plt.title("Weekly emails", fontsize=20)
ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('Number of emails', fontsize=14)
plt.xlabel('Number of questions per mail', fontsize=14)
plt.show()


# **This is work in progress.Please stay tuned!**
