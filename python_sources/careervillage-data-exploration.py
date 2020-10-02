#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# CareerVillage.org is a nonprofit that crowdsources career advice for underserved youth. Founded in 2011 in four classrooms in New York City, the platform has now served career advice from 25,000 volunteer professionals to over 3.5M online learners. The platform uses a Q&A style similar to StackOverflow or Quora to provide students with answers to any question about any career.
# 
# ### Problem
# 
# CareerVillage.org, in partnership with Google.org, requests a method for recommending questions to appropriate volunteers
# 
# The U.S. has almost 500 students for every guidance counselor. Underserved youth lack the network to find their career role models, making CareerVillage.org the only option for millions of young people in America and around the globe with nowhere else to turn.
# 
# To date, 25,000 volunteers have created profiles and opted in to receive emails when a career question is a good fit for them. This is where your skills come in. To help students get the advice they need, the team at CareerVillage.org needs to be able to send the right questions to the right volunteers. The notifications sent to volunteers seem to have the greatest impact on how many questions are answered.
# 
# ### Data
#                                            
# 
# #### Reviewing the Data 
# Before starting the review proces, I import the relevant libraries for the review process
# 

# #### Importing relevant libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import wordcloud


# ### Revieweing the Students DataFrame

# In[ ]:


students = pd.read_csv('../input/students.csv')
students.head()


# checking when was the lastest and the earliest student entry

# In[ ]:


print('Latest:',students.students_date_joined.max(), '\n' + 'Earliest:',students.students_date_joined.min())


# Since the first few values are NaN, I will be checking how many null values as location we have and how many non null 

# In[ ]:


print('not null: ', students.students_location.notnull().sum(), '\nnull:', students.students_location.isnull().sum())
print (round(students.students_location.isnull().sum()/students.students_location.notnull().sum() *100, 2) , '% of the data is null')


# **getting the top 20 locations**

# In[ ]:


students.students_location.value_counts().head(20)


# So far, we can see that the data is not entered in the same format. the top two locations are shown as follows:
# 
# 
# 
# | Data 	| Format 	|
# |---------------------------------	|----------------------	|
# | New York, New York	| City, State 	|
# | Bengaluru, Karnataka, India 	| City, State, Country 	|
# 
# From my first glance at the data, I am asuming that location data wihtout the country is from the US
# 
# 
# To visualize the location frequency:

# In[ ]:


student_location = students.students_location.value_counts().head(20)
student_location.plot.barh(figsize=(10,10), legend=True)
plt.title('Top Locations of students\n',fontsize='16')
plt.ylabel('Location',fontsize='12')
plt.xlabel('Frequency',fontsize='12')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


pros = pd.read_csv("../input/professionals.csv")
pros.head()


# In[ ]:


print('Latest:',pros.professionals_date_joined.max(), '\n' + 'Earliest:',pros.professionals_date_joined.min())


# from the above, we can see that the professionals joined a couple of months before the first student
# 
# To check how many data is missing

# In[ ]:


print('Location: \nnot null: ', pros.professionals_location.notnull().sum(), '\nnull:', pros.professionals_location.isnull().sum())
print (round(pros.professionals_location.isnull().sum()/pros.professionals_location.notnull().sum() *100 , 2), '% of the data is null')
print('Industry: \nnot null: ', pros.professionals_industry.notnull().sum(), '\nnull:', pros.professionals_industry.isnull().sum())
print (round(pros.professionals_industry.isnull().sum()/pros.professionals_industry.notnull().sum() *100, 2) , '% of the data is null')
print('Headline: \nnot null: ', pros.professionals_headline.notnull().sum(), '\nnull:', pros.professionals_headline.isnull().sum())
print (round(pros.professionals_headline.isnull().sum()/pros.professionals_headline.notnull().sum() *100, 2) , '% of the data is null')


# 
# Now to see what is the most frequent location, industry, and headline of the professionals

# In[ ]:


pros.professionals_location.value_counts().head(20)


# from the top 20, it looks like all the professionals are located in the US. The format of location is still there 
# 
# | Data 	| Format 	|
# |---------------------------------	|----------------------	|
# | New York, New York	| City, State 	|
# | Washington 	| State 	|
# 
# It also looks like the majority of the locations of both students and professionals is New York, New York
# 
# To visualize the location frequency:

# In[ ]:


pros_location = pros.professionals_location.value_counts().head(20)
pros_location.plot.barh(figsize=(10,10), legend=True)
plt.title('Top Locations of Professionals\n',fontsize='16')
plt.ylabel('Location',fontsize='12')
plt.xlabel('Frequency',fontsize='12')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


pros.professionals_industry.value_counts().head(20)


# looks like the top industries are mostly wihtin STEM industries. The top 3 are all tech industries, and considering we have 25576 non-null data, this means that they represent 25.6% of the total industries, that without counting the less frequent industries such as 'Computer Hardware' 
# 
# To visualize the industry frequency:

# In[ ]:


pros_industry = pros['professionals_industry'].value_counts().head(20)
pros_industry.plot.barh(figsize=(10,10), legend=True)
plt.title('Top Industries of Professionals\n',fontsize='16')
plt.ylabel('Location',fontsize='12')
plt.xlabel('Frequency',fontsize='12')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


pros.professionals_headline.value_counts().head(20)


# I am asuming that '--' indicates a blank value, if we consider the amount of '--' and NaNs in the headliners columns, it means the majority of the headlines are not entered

# In[ ]:


pros_headline = pros.professionals_headline.value_counts().head(20)
pros_headline.plot.barh(figsize=(10,10), legend=True)
plt.title('Top Headlines of Professionals\n',fontsize='16')
plt.ylabel('Location',fontsize='12')
plt.xlabel('Frequency',fontsize='12')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


qs = pd.read_csv("DataFiles/questions.csv")
qs.head()


# In[ ]:


ans = pd.read_csv("DataFiles/answers.csv")
ans.head()


# In[ ]:


emails = pd.read_csv("DataFiles/emails.csv")
emails.head(0)


# In[ ]:


matches = pd.read_csv("DataFiles/matches.csv")
matches.head(0)


# In[ ]:


tags = pd.read_csv("DataFiles/tags.csv")
tags.head(0)


# In[ ]:


tag_users = pd.read_csv("DataFiles/tag_users.csv")
tag_users.head(0)


# In[ ]:


tag_qs = pd.read_csv("DataFiles/tag_questions.csv")
tag_qs.head(0)


# In[ ]:


groups = pd.read_csv("DataFiles/groups.csv")
groups.head(0)


# In[ ]:


group_m = pd.read_csv("DataFiles/group_memberships.csv")
group_m.head(0)


# In[ ]:


school = pd.read_csv("DataFiles/school_memberships.csv")
school.head(0)


# In[ ]:


commnets = pd.read_csv("DataFiles/comments.csv")
commnets.head(0)


# In[ ]:




