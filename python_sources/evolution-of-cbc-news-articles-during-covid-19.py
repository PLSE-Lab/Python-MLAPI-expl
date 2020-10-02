#!/usr/bin/env python
# coding: utf-8

# # Evolution of CBC new articles

# ## Goal
# 
# To understand how the main focus of CBC news articles has evolved during this COVID -19 affected time period.
# 
# ## Methods
# 
# 1. Analysing the trend with word count of the articles
# 2. Analysing the keywords of articles every month using wordcloud
# 
# ## Results
# It was interesting to observe how the articles has moved from "*what is coronavirus?*" to "*social distancing*" as solution
# 
# Welcome to any feedback!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#  # Data Cleaning 

# In[ ]:


df = pd.read_csv('/kaggle/input/cbc-news-coronavirus-articles-march-26/news.csv', delimiter=',')
df.head()


# Adding extra features

# In[ ]:


df = df.iloc[:, 1:]
df.rename(columns={"publish_date":"publish_timestamp"},inplace=True)
df.publish_timestamp = pd.to_datetime(df.publish_timestamp)
df["title_length"] = df["title"].apply(lambda x : len(x.strip().split()) )
df["description_length"] = df["description"].apply(lambda x : len(x.strip().split()) )
df["text_length"] = df["text"].apply(lambda x : len(x.strip().split()) )
df["publish_date"] = df.publish_timestamp.dt.date
df["publish_year"] = df.publish_timestamp.dt.year
df.head()


# In[ ]:


df.info()


# Let's remove duplicates

# In[ ]:


df.drop_duplicates(inplace=True)
df.info()


# # Analysing the trend with word count

# In order to analyse the trend, we need to understand the timeperiod of these articles

# In[ ]:


df.publish_year.value_counts()


# Looks like we have few articles before 2020. Let's take articles from 2020 alone for our analysis.

# In[ ]:


df_2020 = df[df.publish_year == 2020]


# In[ ]:


df_2020[["title_length","description_length","text_length"]].describe()


# The length of article's *text* has high standard deviation. Let's check the trend with respect to length of the articles

# In[ ]:


df_2020_text_length = df_2020.groupby("publish_date")["text_length"].mean().reset_index()


# In[ ]:


lr = LinearRegression()
lr.fit(df_2020_text_length.index.values.reshape(-1,1), df_2020_text_length.text_length.values.reshape(-1,1) )
predictedLine = lr.predict(df_2020_text_length.index.values.reshape(-1,1))


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
                x=df_2020_text_length.publish_date,
                y=df_2020_text_length.text_length,
                name="text_length",
                line_color='deepskyblue',
                opacity=0.8))

fig.add_trace(go.Scatter(
                x=df_2020_text_length.publish_date,
                y=predictedLine.flatten(),
                name="Trend line",
                line_color='red',
                opacity=0.8))

# Use date string to set xaxis range
fig.update_layout(title_text="Trend on the length of the articles", yaxis_title="word_count")
fig.show()


# Eventhough there are some spikes on the article length, general trend looks almost consistent.

# # Evolution of the articles

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re


# Some words might be quite common and straightforward with respect to certain use cases. Let's remove those kind of words in this usecase

# In[ ]:


usecase_specific_words = ["good","morning","said","says","news", "will","monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
print("Total extra stopwords : ",len([STOPWORDS.add(word) for word in usecase_specific_words]))
stopwords = set(STOPWORDS)


# In[ ]:


df_2020["text"] = df_2020.text.apply(lambda x : x.lower())


# In[ ]:


def word_cloud(month_no, month):
    words_list = df_2020[df_2020.publish_timestamp.dt.month.astype(int) == month_no].text.values.tolist()
    words = " ".join([ re.sub(r"[^a-zA-Z0-9]+", ' ', word.strip()).strip()  for word in " ".join(words_list).split(" ") ]).split(" ")
    wordcloud = WordCloud(
                        background_color='white',
                        stopwords=stopwords,
                        max_font_size=40, 
                        random_state=37,
                        max_words=200
                     ).generate(" ".join(words))
    print(month+" - Total number of words - "+str(len(words)))
    plt.figure(figsize=(14,15))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    


# ### January wordcloud

# In[ ]:


word_cloud(1, "January")


# The articles in January are primarily discussing around the corona virus origin, symptoms and the first case reported at Toronto.  

# ### February Wordcloud

# In[ ]:


word_cloud(2, "February")


# In February, the highlight of words like 
# * quarantine - indicates that the government's action agains corona
# * diamond princess related terms indicate  - the return of 137 Canadians from the Diamon Princess cruise.

# ### March Wordcloud

# In[ ]:


word_cloud(3, "March")


# Articles from March indicates the importance of social distancing, effects of community spread and facilities at hospital
# 

# # Conclusion
# * We could see the evolution of importance in news articles over the time period
# * The dataset size got reduced because of duplicates ( 3500 to 2786 )

# In[ ]:




