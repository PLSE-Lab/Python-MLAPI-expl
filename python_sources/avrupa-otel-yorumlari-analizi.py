#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #basic plot library
import seaborn as sns #for visualition


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# * importing data

# In[ ]:


df=pd.read_csv("../input/Hotel_Reviews.csv") 


# * first contact with data 

# In[ ]:


df.info() #looking for info 
df.columns


# In[ ]:


df.head(10) #first ten


# In[ ]:


df.corr() # as we can see, numbers sometimes can be messy, so lets check it with seaborns heatmap


# In[ ]:


f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,ax=ax) 


# First Review with heatmap;
# as we can see here Total_Number_of_Reviews is positive correlated with Additional_Number_of_Scoring 
# by looking this data we can understand that if hotels get more reviews it gets more scoring points and usually people give positive reviews
# we can slightly say the reviewers score and the avarage_Score is 0.36 correlated 
# just in case lets try to figure out with scatter plot
# 
# 

# In[ ]:


df.plot(kind='scatter',x='Reviewer_Score',y='Average_Score',alpha=0.2,color='red',figsize=(13,13))
plt.xlabel('Reviewer Score')
plt.ylabel('Average Score')
plt.show()


# lets look deep inside of Average scores with histogram plot

# In[ ]:


df.Average_Score.plot(kind='hist',bins=60,figsize=(13,13))
plt.show()


# as we can see average score is usually between 7.9 and 9.1 so we can work with bigger then 7.9 and lower then 9.1

# In[ ]:


df_tight=df[(df.Average_Score>=7.9) & (df.Average_Score<=9.1)]
df_tight.info()


# In[ ]:


text = ""
for i in range(df_tight.shape[0]):
    text = " ".join([text,df_tight["Reviewer_Nationality"].values[i]])

from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white', width=1200, height=600, max_font_size=90, max_words=40).generate(text)
wordcloud.recolor(random_state=312)
plt.imshow(wordcloud)
plt.title("Wordcloud for countries ")
plt.axis("off")
plt.show()


# * Lets find the best 10 hotels in europe
# 

# In[ ]:


df[df.Average_Score >= 5.][['Hotel_Name','Average_Score','Total_Number_of_Reviews']].drop_duplicates().sort_values(by ='Average_Score',ascending = False)[:10]


# Till now what we found;
# * Most of the ratings for hotels are between 7.9 to 9.1
# * Most of the people are from United Kingdom.
# * Top 10 famous Hotels in Europe are(on the basis of number of reviews)
