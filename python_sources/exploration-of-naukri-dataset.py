#!/usr/bin/env python
# coding: utf-8

# **Naukri Analysis**
# 
# This analysis will try to dig deep into the Naukri dataset to get insights on the hiring activity..
# 
# I am here to brush up my DA & DV skills after a long time, therefore this would be a beginner level data analysis and moves slowly to advanced level, consistency is the key.

# In[ ]:


#importing favourites;)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #viz.
import matplotlib.pyplot as plt #viz.

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
       input =  os.path.join(dirname, filename)


# In[ ]:


df = pd.read_csv(input)


# In[ ]:


df.head()


# In[ ]:


df.shape
#There are 30000 rows and 11 columns in the data set


# In[ ]:


print(df.columns)
#column names


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.drop(['Uniq Id','Crawl Timestamp'],axis=1,inplace=True)
#dropping the identifier fields as it wont be much helpful for out analysis, since we are not going to drilldown to an ID level analysis


# In[ ]:


title = df['Job Title'].value_counts()[:20]
title.plot(kind='barh') 
#top 20 job titles


# In[ ]:


location = df['Location'].value_counts()[:10]
location.plot(kind = 'barh',color = "red")


# In[ ]:


salary = df['Job Salary'].value_counts()[:25]

salary.plot(kind='barh')
#most of the salaries are not disclosed by recruiters, many are not properly formatted


# In[ ]:


skills = df['Key Skills'].value_counts()[:10]

skills.plot(kind='barh',color = "green")
#Top 10 skills combinations


# In[ ]:


industry = df["Industry"].value_counts()[:10]
industry.plot(kind = "barh",color = "purple")
#Top 10 industries


# In[ ]:


exp = df["Job Experience Required"].value_counts()[:15]
exp.plot(kind = "barh",color = "yellow")
#Most of the jobs require 2-5yrs of experience


# In[ ]:


#splitting key skills column based on delimiter to visualise the top skill
ne_skill = df['Key Skills'].str.split("|", n = 15, expand = True) 


# In[ ]:


#Adding top 5 skills
df['newskill1']=ne_skill[0]
df['newskill2']=ne_skill[1]
df['newskill3']=ne_skill[2]
df['newskill4']=ne_skill[3]
df['newskill5']=ne_skill[4]


# In[ ]:


df.head()


# In[ ]:


ne_skill


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
wc1 = df['newskill1'].values 

wordcloud = WordCloud(max_font_size=100, max_words=30000, background_color="white").generate(str(wc1))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


wc2 = df['newskill2'].values
wordcloud2 = WordCloud(max_font_size=100,max_words=30000,background_color = "white").generate(str(wc2))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()


# In[ ]:


wc3 = df['newskill3'].values
wordcloud3 = WordCloud(max_font_size = 100,max_words=30000,background_color = "white").generate(str(wc3))
plt.imshow(wordcloud3)
plt.axis("off")
plt.show()


# In[ ]:


wc4 = df['newskill4'].values
wordcloud4 = WordCloud(max_font_size = 100,max_words=30000,background_color = "white").generate(str(wc4))
plt.imshow(wordcloud4)
plt.axis("off")
plt.show()


# In[ ]:


wc5 = df['newskill5'].values
wordcloud5 = WordCloud(max_font_size = 100,max_words=30000,background_color = "white").generate(str(wc5))
plt.imshow(wordcloud5)
plt.axis("off")
plt.show()


# This is th combined wordcloud with the one fat df ne_skill that we have created to spilt by delimiter

# In[ ]:


neskill = ne_skill.values 

wordcloude = WordCloud(max_font_size=100, max_words=30000, background_color="white").generate(str(neskill))

plt.imshow(wordcloude)
plt.axis("off")
plt.show()
#Looks like none, security, software, media and digital tops the chart


# In[ ]:


df["Functional Area"].head()

#looks like functional area is having multiple values, we are going to split it


# In[ ]:


fun_area = df["Functional Area"].str.split(",", n = 5, expand = True) 


# In[ ]:


fun_area.head(20)

df['functional_1']=fun_area[0]
df['functional_2']=fun_area[1]
df['functional_3']=fun_area[2]
df['functional_4']=fun_area[3]
df['functional_5']=fun_area[4]


# In[ ]:


fun_1 = df['functional_1'].values 

fun1_wc = WordCloud(max_font_size=100, max_words=30000, background_color="white").generate(str(fun_1))

plt.imshow(fun1_wc)
plt.axis("off")
plt.show()


# In[ ]:


fun_2 = df['functional_2'].values 

fun2_wc = WordCloud(max_font_size=100, max_words=30000, background_color="white").generate(str(fun_2))

plt.imshow(fun2_wc)
plt.axis("off")
plt.show()


# In[ ]:


fun_3 = df['functional_3'].values 

fun3_wc = WordCloud(max_font_size=100, max_words=30000, background_color="white").generate(str(fun_3))

plt.imshow(fun3_wc)
plt.axis("off")
plt.show()


# In[ ]:


#Heatmap of locations 
counter_loc = df['Location'].value_counts()
counter_loc.head()

import pandas as pd
df2 = pd.DataFrame({'nb_people':[4986,3318,2431,2144,1751], 'group':["Bengaluru", "Mumbai", "Pune", "Hyderabad","Gurgaon"] })
squarify.plot(sizes=df2['nb_people'], label=df2['group'], color=["red","green","blue", "grey","orange"],alpha=.5 )
plt.axis('off')
plt.show()


# Index(['Uniq Id', 'Crawl Timestamp', 'Job Title', 'Job Salary',
#        'Job Experience Required', 'Key Skills', 'Role Category', 'Location',
#        'Functional Area', 'Industry', 'Role'],
#       dtype='object')
