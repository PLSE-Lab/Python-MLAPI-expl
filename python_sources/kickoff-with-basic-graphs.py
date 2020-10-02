#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# This is a playground for me to have some basic plots here to explore first, also for other people to check up some basic distributions here. I'll also put some notes of my observation. I'll have my analysis in another kernel after finishing here.

# In[ ]:


# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# import dataset
free_from = pd.read_csv('../input/freeFormResponses.csv')
free_from.columns = free_from.iloc[0]
free_from = free_from.iloc[1:]


# In[ ]:


free_from.head()


# In[ ]:


multiple = pd.read_csv('../input/multipleChoiceResponses.csv')
multiple.columns = multiple.iloc[0]
multiple = multiple.iloc[1:]


# In[ ]:


multiple.head()


# In[ ]:


sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='What is your gender? - Selected Choice',data=multiple,
              order = multiple['What is your gender? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('What is your gender?')


# Over 80% of the people here are Male. 

# In[ ]:


sns.countplot(y='What is your age (# years)?',data=multiple,
              order = multiple['What is your age (# years)?'].value_counts().index)
plt.ylabel('')
plt.title('What is your age (# years)?')


# Most of the Kaggler are between 18 and 34 years old, especially 22 to 29.

# In[ ]:


sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',data=multiple,
             order = multiple['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().index)
plt.ylabel('')
plt.title('What is the highest level of formal education that you have attained or plan to attain within the next 2 years?')


# Based on this graph, I think most of the Kaggler between 22 to 29 years old are students and fresh graduates who are practising and preparing themselves here for their first job. As a lot of articles saying, Kaggle is a great place for people who want to be a data scientist or data analyst to have some experience to meet the requirements of the positions. Me, myself, also made a lot of kernels to be my side projects and portfolio. And they do get me some offers and a great job recently.

# In[ ]:


sns.set(rc={'figure.figsize':(12,12)})
sns.countplot(y='In which country do you currently reside?',data=multiple,
             order = multiple['In which country do you currently reside?'].value_counts().index)
plt.ylabel('')
plt.title('In which country do you currently reside?')


# We can see that the amount of American and Indian are way bigger than that of others. Well, if you stay here for long enough, you can see there are really some great Kagglers from these two countries with some prestigiuos Kagglers from the other countries. In my opinion, I saw some outstanding Kaggler, like [Janio Alexander Bachmann](https://www.kaggle.com/janiobachmann), [Bojan Tunguz](https://www.kaggle.com/tunguz), [Erik Bruin](https://www.kaggle.com/erikbruin), [Sang-eon Park](https://www.kaggle.com/caicell), and [Siddharth Yadav](https://www.kaggle.com/thebrownviking20) are all my favorite. They all come from different corner around the world.

# In[ ]:


sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='Which best describes your undergraduate major? - Selected Choice',data=multiple,
             order = multiple['Which best describes your undergraduate major? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('Which best describes your undergraduate major?')


# Well, as we known, most people own a CS degree here. However, I frequently saw outstanding analysis or models made by people with a Physics degree or a Linguistics degree. As a Business major person, I think it has a significant impact on the way I think and how I analyze my data. In the last summer, I was a Product Manager Intern at HTC VIVE. I cowork with people all over the world. And it empowered my capability of defining questions, managin to meet business requirements, and convey how I solve the other people's problems. So, in my opinion, I think people with different degrees and different backgrounds all have their unique way to think.      
#    
#      
# Data Scientist Nanodegree of Udacity said there are three elements of being a data scientist: Statistics, Computer Science, and own domain. I can't agree more. I think how a person be a special and outstanding data scientist is to apply his or her own domain to solve the problem with data science in his or her own way.

# In[ ]:


sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(y='Select the title most similar to your current role (or most recent title if retired): - Selected Choice',data=multiple,
             order = multiple['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('Select the title most similar to your current role (or most recent title if retired)')


# Most of the people on Kaggle are students. I guess it's because Kaggle is really good for learning and earning experience to be prepared for jobs.  The following three are just classic. A lot of Data Analyst, Scientist and Engineer practice code and join machine learning competitions here. We all know that.

# In[ ]:


sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',data=multiple,
             order = multiple['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().index)
plt.ylabel('')
plt.title('What is the highest level of formal education that you have attained or plan to attain within the next 2 years?')


# In[ ]:


sns.countplot(y='In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice',data=multiple,
             order = multiple['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('In what industry is your current employer/contract (or your most recent employer if retired)?')


# I'm not surprised that a great amount of Kaggler have Academia or Education backgrounds. A lot of kernels I read are really good for learning and well-organized. I love them. And that's why there are so many students here.

# In[ ]:


sns.set(rc={'figure.figsize':(7,3)})
sns.countplot(y='How many years of experience do you have in your current role?',data=multiple,
             order = multiple['How many years of experience do you have in your current role?'].value_counts().index)
plt.ylabel('')
plt.title('How many years of experience do you have in your current role?')


# As I mentioned, I think a lot of students and fresh graduates like me accumulate their works and practice here to be prepared for their first data science job. 

# In[ ]:


sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(y='What is your current yearly compensation (approximate $USD)?',data=multiple,
             order = multiple['What is your current yearly compensation (approximate $USD)?'].value_counts().index)
plt.ylabel('')
plt.title('What is your current yearly compensation (approximate $USD)?')


# Well, I believer there are more people with more salaries here. I believe those people are hiding in those prefer not to disclose haha

# In[ ]:


# unstack Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years?


# In[ ]:


df = multiple.loc[:,"Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Jupyter/IPython":"Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Other - Text"]


# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:


s = df.astype(bool).sum(axis=0)


# In[ ]:


s.index


# In[ ]:


s = s.rename(lambda x: x.replace("Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - ", ''))
s = s.rename(lambda x: x.replace("Selected Choice - ", ''))
s = s.rename(lambda x: x.replace("Other - ", ''))


# In[ ]:


sns.barplot(s.index, s.values, alpha=0.8)
plt.xticks(rotation=90)
plt.title('IDE distribution')


# I think it's not surprised that a lot of users here use Jupyter notebook. And quiet some people use Rstudio as well. However, it's pretty surprised to me that most people use Text and Notepad++. But, to be honestly, I personally use these two tools to edit my codes every now and then to replace all or check some syntax

# In[ ]:


sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(y='What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice',data=multiple,
             order = multiple['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('What programming language would you recommend an aspiring data scientist to learn first?')


# Since this community is for data science, we can see that Python is dominant here, while followed by R and SQL but way less than the Python users.

# In[ ]:


sns.countplot(y='What specific programming language do you use most often? - Selected Choice',data=multiple,
             order = multiple['What specific programming language do you use most often? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('What specific programming language do you use most often?')


# In[ ]:


df = multiple.loc[:,"What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python":"What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other"]


# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:


s = df.astype(bool).sum(axis=0)


# In[ ]:


s = s.rename(lambda x: x.replace("What programming languages do you use on a regular basis?", ''))
s = s.rename(lambda x: x.replace("(Select all that apply)", ''))
s = s.rename(lambda x: x.replace(" - Selected Choice - ", ''))
s = s.rename(lambda x: x.replace("Other - ", ''))


# In[ ]:


sns.barplot(s.index, s.values, alpha=0.8)
plt.xticks(rotation=90)
plt.title('IDE distribution')


# In[ ]:




