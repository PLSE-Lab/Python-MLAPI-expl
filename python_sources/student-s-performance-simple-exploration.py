#!/usr/bin/env python
# coding: utf-8

# 

# ![](http://blog.edmentum.com/sites/blog.edmentum.com/files/styles/blog_image/public/images/Measure.jpg?itok=-HpJXeUt)

# This is a short and simple analysis of the student's performance dataset. Hope you will like it !

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly 
import plotly.offline as py
import plotly.graph_objs as go

init_notebook_mode(connected=True)

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/StudentsPerformance.csv")
data.shape


# Let's look into the data...

# In[ ]:


data.head()


# Checking for null values...

# In[ ]:


data.isna().sum()


# No NULL values! What can be better than this!!

# # Gender Distribution

# Let's see the gender distribution of the respondants..

# In[ ]:


data['gender'].value_counts()


# In[ ]:


fig = plt.figure(figsize = (15,10))
ax = sns.barplot(y = data['gender'].value_counts(),x = data['gender'].value_counts().index)
plt.xlabel("gender")
plt.ylabel("count")
plt.show()


# 1. Male students --> 51.8% 
# 1. Female students -->48.2%   
# Dataset  is balanced!!

# # Parent's Education

# In[ ]:


edu = data['parental level of education'].value_counts()


# In[ ]:



fig = {
  "data": [
    {
      "values": edu.values,
      "labels": edu.index,
      "domain": {"x": [0.1, 1]},
      "name": "Full Population",
      "hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    },
       
    ],
  "layout": {
        "title":"Parent's Education",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Full Population",
                "x": 0.55,
                "y": 0.5
            },
             
        ]
    }
}
py.iplot(fig, filename='donut')


# As expected, only *5.9%* of the parents have Master's  and *11.8% *have Bachelor's degree.

# # Exam Scores

# ![](http://png.pngtree.com/element_origin_min_pic/16/09/07/1457cfb6fdc628b.jpg)

# # Let's talk about MATHS!!

# In[ ]:


fig = plt.figure(figsize = (10,6))
grouped = data.groupby('gender')['math score'].mean().reset_index()
grouped.columns = ['gender','math_score']

a = sns.barplot(x = 'gender',y = 'math_score', data = grouped)
for index, row in grouped.iterrows():
    if index == 1:
        index+=3
    a.text(row.name,40, round(row.math_score,3), color='black', ha="center")


# Average Maths score of Boys is 68.7/100  whereas of Girls is 63.6/100 . Hmmm..

# # What about READING section ?

# In[ ]:


fig = plt.figure(figsize = (10,6))
grouped = data.groupby('gender')['reading score'].mean().reset_index()
grouped.columns = ['gender','reading_score']

a = sns.barplot(x = 'gender',y = 'reading_score', data = grouped)
for index, row in grouped.iterrows():
    a.text(row.name,40, round(row.reading_score,3), color='black', ha="center")


# Girls are holding the position with average score of 72.6/100 whereas boys on average scored 65.4/100.

# # Who is good at WRITING ?

# In[ ]:


fig = plt.figure(figsize = (10,6))
grouped = data.groupby('gender')['writing score'].mean().reset_index()
grouped.columns = ['gender','writing_score']

a = sns.barplot(x = 'gender',y = 'writing_score', data = grouped)
for index, row in grouped.iterrows():
    a.text(row.name,40, round(row.writing_score,3), color='black', ha="center")


# Again Girls on the top with average score of 72.4/100 leaving behind boys with average score of 63.3/100.

# # Overall Score

# No wonder, girls remain on the top!!

# In[ ]:


fig = plt.figure(figsize = (10,6))
data['avg_score'] = (data['math score'] + data['reading score'] + data['writing score'])/3

grouped = data.groupby('gender')['avg_score'].mean().reset_index()
grouped.columns = ['gender','avg_score']

a = sns.barplot(x = 'gender',y = 'avg_score', data = grouped)
for index, row in grouped.iterrows():
    a.text(row.name,40, round(row.avg_score,3), color='black', ha="center")


# # Test Preparation and Results

# In[ ]:


fig = plt.figure(figsize = (10,6))
test_prep = data.groupby('test preparation course')['avg_score'].mean().reset_index()
test_prep.columns = ['test_preparation_course','avg_score']
a = sns.barplot(x = 'test_preparation_course',y = 'avg_score', data = test_prep)
for index, row in test_prep.iterrows():
    a.text(row.name,40, round(row.avg_score,3), color='black', ha="center")


# As expected, students who completed the test preparation course scored higher than others. On average those who completed test preparation course scord 72.6/100
# and those who didn't scored 65.0/100 .

# In[ ]:


test_prep = data.groupby('gender')['test preparation course'].value_counts()
male_test_prep = test_prep['male'][1]/test_prep['male'].values.sum()
female_test_prep = test_prep['female'][1]/test_prep['female'].values.sum()
print('%age of Male population who completed preparation for test---->',round(male_test_prep,3)*100)
print('%age of Female population who completed preparation for test---->',round(female_test_prep,3)*100)


# This is interesting! 36.1% boys completed the test preparation course which is more than girls, but still scored less !!

# # Does parents education impacts performance of children in school?

# In[ ]:


fig = plt.figure(figsize = (12,12))
edu_score = data.groupby('parental level of education')['avg_score'].mean().sort_values(ascending =False).reset_index()
edu_score.columns = ['parental_level_of_education','score']

a = sns.barplot(x = 'parental_level_of_education',y = 'score', data = edu_score)
for index, row in edu_score.iterrows():
    a.text(row.name,40, round(row.score,3), color='black', ha="center")


# As per data **YES**, it has a large impact!
# Students whose parents have Master's degree scored 73.5/100 which is highest among others!

# What can be the reasons? 
# 

# Let me test a silly hypothesis. Is it that more educated parents made their child complete the test preparation course ?

# In[ ]:


p_edu = data.groupby('parental level of education')['test preparation course'].value_counts()
p_edu


# In[ ]:


deg_dict = data['parental level of education'].value_counts().to_dict()
deg_dict


# In[ ]:


temp = pd.DataFrame(columns= ['parent_level_of_education','percentage_students_who_completed_preparation'])
i = 0
for index, row in p_edu.iteritems():
    if index[1] != 'none':
        temp.loc[i,:] = index[0],row/deg_dict[index[0]]
        i+=1
temp = temp.sort_values(by = 'percentage_students_who_completed_preparation',ascending = False).reset_index(drop = True)
temp


# In[ ]:


fig = plt.figure(figsize = (12,8))
ax = sns.barplot(x = 'parent_level_of_education',y = 'percentage_students_who_completed_preparation', data = temp)
for index, row in temp.iterrows():
    ax.text(row.name,0.2, round(row.percentage_students_who_completed_preparation,3), color='black', ha="center")


# Ohh! This is not something I thought.

# So the hypothesis was actually wrong. And their seems no straightforward relation between education of parents and completion of test preparation course by their children .

# # Lunch

# ![](https://rlv.zcache.es/pegatina_redonda_emoji_divertido_hace_frente_a_amarillo_delicioso-reee1250d7424440288a9fe9e2d9985df_v9waf_8byvr_630.jpg?view_padding=%5B285%2C0%2C285%2C0%5D)

# In[ ]:


data['lunch'].value_counts()


# In[ ]:


lunch_score = data.groupby('lunch')['avg_score'].mean().sort_values(ascending = False).reset_index()
lunch_score


# In[ ]:


fig = plt.figure(figsize = (14,8))
a = sns.barplot(x = 'lunch',y = 'avg_score', data = lunch_score)
for index, row in lunch_score.iterrows():
    a.text(row.name,40, round(row.avg_score,3), color='black', ha="center")


# Children taking standard lunch perform better than others as shown by their average marks.

# **Education of parents and lunch they provide to their children ..**

# In[ ]:


food = data.groupby('parental level of education')['lunch'].value_counts()
food


# In[ ]:



temp1 = pd.DataFrame(columns= ['parent_level_of_education','standard_lunch'])
i = 0
for index, row in food.iteritems():
    if index[1] != 'free/reduced':
        temp1.loc[i,:] = index[0],row/deg_dict[index[0]]
        i+=1
temp1 = temp1.sort_values(by = 'standard_lunch',ascending = False).reset_index(drop = True)
temp1


# Parents with higher education seems more busy !!

# In[ ]:


fig = plt.figure(figsize = (14,8))
a = sns.barplot(x = 'parent_level_of_education',y = 'standard_lunch', data = temp1)
for index, row in temp1.iterrows():
    a.text(row.name,.4, round(row.standard_lunch,3), color='black', ha="center")


# That's it for now!

# **Any Suggestions are most welcome!!**

# **Thanks for going through it, please upvote if you found it interesting.**

# In[ ]:




