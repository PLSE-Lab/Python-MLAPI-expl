#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# * In this kernel, we will investigate youtube trends.
# 
# <br>Content:
# 1. [Loading Data](#1)
# 1. [Basic skimming on our data](#2)
# 1. [How long usually a Video Can Trend in Different Countries](#3)
# 1. [Number of Youtube Trending Videos in 6 countries](#4)
# 1. [How many likes, dislikes, views and comments get by different countries](#5)
# 1. [ Users like videos from which CATEGORY the most?](#6)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sns
import json
# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> <br>
# # Loading Datas

# In[ ]:


#reading datas
catrends = pd.read_csv("../input/youtube-new/CAvideos.csv")
detrends = pd.read_csv("../input/youtube-new/DEvideos.csv")
frtrends = pd.read_csv("../input/youtube-new/FRvideos.csv")
gbtrends = pd.read_csv("../input/youtube-new/GBvideos.csv")
intrends = pd.read_csv("../input/youtube-new/INvideos.csv")
ustrends = pd.read_csv("../input/youtube-new/USvideos.csv")

generaltrends_full = pd.concat([catrends,detrends,frtrends,gbtrends,intrends,ustrends])


# In[ ]:





# <a id="2"></a> <br>
# ## Basic skimming on our data

# In[ ]:


#general analysis about datas and comparing
catrends.info()
intrends.info()
frtrends.info()

#we see that there is no NaN data


# In[ ]:


ustrends.head()


# ### Setting datas what we need
# ### Escaping from unnecessary columns actually

# In[ ]:


catrends = catrends[["trending_date","title","channel_title","publish_time","views","likes","dislikes","comment_count","category_id","video_id"]]
ustrends = ustrends[["trending_date","title","channel_title","publish_time","views","likes","dislikes","comment_count","category_id","video_id"]]
detrends = detrends[["trending_date","title","channel_title","publish_time","views","likes","dislikes","comment_count","category_id","video_id"]]
frtrends = frtrends[["trending_date","title","channel_title","publish_time","views","likes","dislikes","comment_count","category_id","video_id"]]
intrends = intrends[["trending_date","title","channel_title","publish_time","views","likes","dislikes","comment_count","category_id","video_id"]]
gbtrends = gbtrends[["trending_date","title","channel_title","publish_time","views","likes","dislikes","comment_count","category_id","video_id"]]


# In[ ]:


ustrends.trending_date.unique()


# In[ ]:


#detecting how much trend data per a day
dataperday=0
trenddays = [int(each.replace(".","")) for each in ustrends["trending_date"]]
for i in trenddays:
    if(i==171411):
        dataperday+=1
    else:
        break
print(dataperday)
#we see that there are 200 content in a day.


# <a id="3"></a> <br>
# ## How long usually a video can trend in different countries

# In[ ]:


ustrends.head()


# In[ ]:


ustrends.title.value_counts().values.mean()


# In[ ]:


# times to stay in trend averages
ustrenddates = ustrends.title.value_counts().values.mean()
catrenddates = catrends.title.value_counts().values.mean()
detrenddates = detrends.title.value_counts().values.mean()
frtrenddates = frtrends.title.value_counts().values.mean()
gbtrenddates = gbtrends.title.value_counts().values.mean()
intrenddates = intrends.title.value_counts().values.mean()

labels = ['United States','Canada','Germany','France','United Kingdom','India']
pielist = [ustrenddates,catrenddates,detrenddates,frtrenddates,gbtrenddates,intrenddates]

#figure

fig = {
    'data' : 
    [
        {
            "values": pielist,
          "labels": labels,
          "domain": {"x": [0, .5]},
          "name": "General Stay in trend rates",
          "hoverinfo":"label+percent+name",
          "hole": .25,
          "type": "pie"
        },
    ],

    'layout' :
    {
        "title":"Trends According to Countries",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": True,  # Title's arrow
              "text": "Average Trend Days",
                "x": 0.20,
                "y": 1
    },
        ]
    }
}
iplot(fig)


# ### Observation
# 
# According to the pie plot, it can be seen that United Kingdom at top of the enduring trend list follow by US and India. Unlike that 3 countries, Canada, Germany and France have very few video can last long in trending.

# <a id="4"></a> <br>
# ## Number of Youtube Trending Videos in 6 countries

# In[ ]:


#Adding Country column manually
catrends["Country"] = "Canada"
detrends["Country"] = "Germany"
frtrends["Country"] = "France"
gbtrends["Country"] = "United Kingdom"
intrends["Country"] = "India"
ustrends["Country"] = "United States"
#sum of data of all countries
generaltrends = pd.concat([catrends,detrends,frtrends,gbtrends,intrends,ustrends])
generaltrends


# In[ ]:


generaltrends.groupby(['Country']).count()


# In[ ]:


labels = generaltrends.groupby(['Country']).count().index
datasize = generaltrends.groupby(['Country']).count()["title"]


# import graph objects as "go"
import plotly.graph_objs as go

#create trace1
trace1 = go.Bar(
            x = labels,
            y = datasize,
            name = "Number of Trend Videos",
            marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
)

data = [trace1]
layout= go.Layout(barmode="group")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
                


# <a id="5"></a> <br>
# ## How many likes, dislikes, views and comments get by different countries?

# In[ ]:


# trends.likes.value_counts().sum() is number of liked videos
# ustrends.likes.sum()    sum of all likes
countrylikerate = []
countrydislikerate = []
countryviewrate = []
countrycommentrate = []
countrylist = list(generaltrends.Country.unique())
for i in countrylist:
    likes = generaltrends.likes[generaltrends.Country ==i] #like numbers for each country
    likerates = sum(likes)/len(likes)
    countrylikerate.append(likerates)
    
for j in countrylist:
    dislikes = generaltrends.dislikes[generaltrends.Country ==j] #like numbers for each country
    dislikerates = sum(dislikes)/len(dislikes)
    countrydislikerate.append(dislikerates)

for k in countrylist:
    views = generaltrends.views[generaltrends.Country ==k] #like numbers for each country
    viewrates = sum(views)/len(views)
    countryviewrate.append(viewrates)
for l in countrylist:
    comments = generaltrends.comment_count[generaltrends.Country ==l] #like numbers for each country
    commentrates = sum(comments)/len(comments)
    countrycommentrate.append(commentrates)


#creating traces
trace1 = go.Bar(
                x = countrylist,
                y = countrylikerate,
                name = "Like Rates For Trend Videos",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = countrylist)

trace2 = go.Bar(
                x = countrylist,
                y = countrydislikerate,
                name = "Dislike Rates For Trend Videos",
                marker = dict(color = 'rgba(25, 174, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = countrylist)


trace3 = go.Bar(
                x = countrylist,
                y = countryviewrate,
                name = "View Rates For Trend Videos",
                marker = dict(color = 'rgba(255, 174, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = countrylist)


trace4 = go.Bar(
                x = countrylist,
                y = countrycommentrate,
                name = "Comment Rates For Trend Videos",
                marker = dict(color = 'rgba(25, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = countrylist)


data = [trace1,trace2,trace3,trace4]
layout = go.Layout(barmode = "group")
fig = go.Figure(data=data,layout = layout)
iplot(fig)


# ### Observation
# According to bar plot, we observe that in UK, People genererally watch and like videos unlike their low population according to USA and India. They watch youtube videos million times. Of course this data is coming from trend videos so we deduce that people watch same contents simultaneously. UK citizens can be manipulated easily by popular youtube content creators.

# <a id="6"></a> <br>
# # Users like videos from which CATEGORY the most?

# In[ ]:


#inserting categories from json files
generaltrends["category_id"] = generaltrends["category_id"].astype(str)
generaltrends_full['category_id'] = generaltrends_full['category_id'].astype(str)  # we store original data


category_id = {}

with open('../input/youtube-new/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        category_id[category['id']] = category['snippet']['title']
        
generaltrends.insert(4, 'category', generaltrends['category_id'].map(category_id))
generaltrends_full.insert(4, 'category', generaltrends_full['category_id'].map(category_id))


# In[ ]:


#From Canada 

labels = generaltrends.category[generaltrends["Country"] == "Canada"].value_counts().index
data = generaltrends.category[generaltrends["Country"] == "Canada"].value_counts().values

#visualization
plt.figure(figsize=(15,8))
ax = sns.barplot(y=labels,x=data)
plt.grid()
plt.title("Youtube Viewing Categories of Canada",color="r")
plt.xlabel("View value of Categories",color="gray")
plt.ylabel("Categories",color="gray")


# In[ ]:


#From Germany

labels = generaltrends.category[generaltrends["Country"] == "Germany"].value_counts().index
data = generaltrends.category[generaltrends["Country"] == "Germany"].value_counts().values

#visualization
plt.figure(figsize=(15,8))
ax = sns.barplot(y=labels,x=data)
plt.grid()
plt.title("Youtube Viewing Categories of Germany",color="r")
plt.xlabel("View value of Categories",color="gray")
plt.ylabel("Categories",color="gray")


# In[ ]:


#From France

labels = generaltrends.category[generaltrends["Country"] == "France"].value_counts().index
data = generaltrends.category[generaltrends["Country"] == "France"].value_counts().values

#visualization
plt.figure(figsize=(15,8))
ax = sns.barplot(y=labels,x=data)
plt.grid()
plt.title("Youtube Viewing Categories of France",color="r")
plt.xlabel("View value of Categories",color="gray")
plt.ylabel("Categories",color="gray")


# In[ ]:


#From United Kingdom 

labels = generaltrends.category[generaltrends["Country"] == "United Kingdom"].value_counts().index
data = generaltrends.category[generaltrends["Country"] == "United Kingdom"].value_counts().values

#visualization
plt.figure(figsize=(15,8))
ax = sns.barplot(y=labels,x=data)
plt.grid()
plt.title("Youtube Viewing Categories of United Kingdom",color="r")
plt.xlabel("View value of Categories",color="gray")
plt.ylabel("Categories",color="gray")


# In[ ]:


#From India

labels = generaltrends.category[generaltrends["Country"] == "India"].value_counts().index
data = generaltrends.category[generaltrends["Country"] == "India"].value_counts().values

#visualization
plt.figure(figsize=(15,8))
ax = sns.barplot(y=labels,x=data)
plt.grid()
plt.title("Youtube Viewing Categories of India",color="r")
plt.xlabel("View value of Categories",color="gray")
plt.ylabel("Categories",color="gray")


# In[ ]:


#From United States

labels = generaltrends.category[generaltrends["Country"] == "United States"].value_counts().index
data = generaltrends.category[generaltrends["Country"] == "United States"].value_counts().values

#visualization
plt.figure(figsize=(15,8))
ax = sns.barplot(y=labels,x=data)
plt.grid()
plt.title("Youtube Viewing Categories of United States of America",color="r")
plt.xlabel("View value of Categories",color="gray")
plt.ylabel("Categories",color="gray")


# In[ ]:




